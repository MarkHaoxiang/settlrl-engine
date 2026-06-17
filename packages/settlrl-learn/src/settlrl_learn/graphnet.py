"""A configurable graph net over the board, the major architecture levers as
config knobs so an ablation is a config sweep, not a rewrite.

Design stance (and how player/board invariance is maintained): the net carries
**no absolute positional encoding** -- a rotated board is the same game, so the
strategic signal lives in the node/edge *features* (production, ownership,
ports), not in a vertex index. Every operation here is symmetric over nodes
(message passing, attention via segment-softmax, the global node's pooled
update, the readout aggregators) and reads ownership relatively (own vs. other),
so the output is invariant under the board's symmetry group and the player
relabeling -- the same contracts ``tests/test_architectures.py`` enforces.

Levers (``GraphNetConfig``):

- ``conv`` -- ``"mpnn"`` (message MLP + sum aggregation, count-sensitive) vs
  ``"gat"`` (GATv2 dynamic attention, Brody et al. 2022);
- ``norm`` -- ``"none"`` / ``"layer"`` (per-node) / ``"graph"`` (GraphNorm,
  Cai et al. 2021: normalise across nodes with a learnable mean-shift);
- ``global_node`` -- a virtual global node seeded from the global features and
  updated from a pooled summary each layer (O(N) long-range, no O(N^2) attention);
- ``readout`` -- ``"mean"`` vs ``"multi"`` (mean ++ max ++ sum: ``sum`` keeps the
  *count* signal -- how many settlements/cities are mine -- that ``mean`` washes
  out, the PNA argument, Corso et al. 2020);
- ``jk`` -- jumping-knowledge: pool every layer's node state, not just the last
  (multi-scale, dodges over-smoothing);
- ``layers`` / ``width`` / ``heads`` -- depth/capacity. Non-recurrent: each layer
  has its own weights.
"""

from __future__ import annotations

from typing import NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
from jaxtyping import Array, Float
from settlrl_engine.board.layout import N_VERTICES
from settlrl_engine.board.state import KeyScalar

from settlrl_learn.graph import RECEIVERS, SENDERS, Sample


class GraphNetConfig(NamedTuple):
    width: int = 64
    layers: int = 3  # message-passing layers (each with its own weights)
    head_depth: int = 2  # readout MLP hidden layers
    conv: str = "mpnn"  # "mpnn" | "gat"
    heads: int = 4  # attention heads (gat)
    norm: str = "layer"  # "none" | "layer" | "graph"
    residual: bool = True
    global_node: bool = True
    readout: str = "multi"  # "mean" | "sum" | "multi"
    jk: bool = False


def _aggr(messages: Float[Array, "e w"]) -> Float[Array, "v w"]:
    """Sum a per-edge message into its receiver node (count-sensitive)."""
    return cast(Array, jraph.segment_sum(messages, RECEIVERS, num_segments=N_VERTICES))


class _GraphNorm(eqx.Module):
    """Normalise each feature across nodes with a learnable mean-shift ``alpha``
    (Cai et al. 2021). ``alpha`` lets the layer keep some of the graph-mean,
    which a plain instance-norm would discard on regular graphs."""

    scale: Array
    shift: Array
    alpha: Array

    def __init__(self, width: int) -> None:
        self.scale = jnp.ones((width,))
        self.shift = jnp.zeros((width,))
        self.alpha = jnp.ones((width,))

    def __call__(self, x: Float[Array, "v w"]) -> Float[Array, "v w"]:
        mean = x.mean(axis=0, keepdims=True)
        centred = x - self.alpha * mean
        var = centred.var(axis=0, keepdims=True)
        return centred * jax.lax.rsqrt(var + 1e-5) * self.scale + self.shift


_Norm = eqx.nn.LayerNorm | _GraphNorm


def _make_norm(norm: str, width: int) -> _Norm | None:
    if norm == "layer":
        return eqx.nn.LayerNorm(width)
    if norm == "graph":
        return _GraphNorm(width)
    return None


def _apply_norm(norm_mod: _Norm | None, x: Float[Array, "v w"]) -> Array:
    if norm_mod is None:
        return x
    if isinstance(norm_mod, eqx.nn.LayerNorm):
        return jax.vmap(norm_mod)(x)  # per-node over the feature axis
    return norm_mod(x)  # GraphNorm spans the node axis itself


class _Layer(eqx.Module):
    msg: eqx.nn.MLP | None  # mpnn message function
    att_w: eqx.nn.Linear | None  # gat: W over [h_s, h_r, e]
    att_a: Array | None  # gat: attention vector per head
    val_w: eqx.nn.Linear | None  # gat: value projection of the sender
    node: eqx.nn.MLP  # node update over [h, aggregate, global?]
    glob: eqx.nn.MLP | None  # virtual global-node update
    norm: _Norm | None
    cfg: GraphNetConfig = eqx.field(static=True)

    def __init__(self, key: KeyScalar, cfg: GraphNetConfig) -> None:
        w = cfg.width
        g_in = w if cfg.global_node else 0
        keys = jax.random.split(key, 5)
        if cfg.conv == "gat":
            assert w % cfg.heads == 0, "width must divide heads"
            d = w // cfg.heads
            self.msg = None
            self.att_w = eqx.nn.Linear(3 * w, cfg.heads * d, key=keys[0])
            self.att_a = jax.random.normal(keys[1], (cfg.heads, d)) * 0.1
            self.val_w = eqx.nn.Linear(w, cfg.heads * d, key=keys[2])
        else:
            self.msg = eqx.nn.MLP(3 * w + g_in, w, w, 1, key=keys[0])
            self.att_w = self.val_w = None
            self.att_a = None
        self.node = eqx.nn.MLP(2 * w + g_in, w, w, 1, key=keys[3])
        self.glob = eqx.nn.MLP(3 * w, w, w, 1, key=keys[4]) if cfg.global_node else None
        self.norm = _make_norm(cfg.norm, w)
        self.cfg = cfg

    def _aggregate(
        self, h: Float[Array, "v w"], e: Float[Array, "e w"], g: Array
    ) -> Float[Array, "v w"]:
        hs, hr = h[SENDERS], h[RECEIVERS]
        if self.cfg.conv == "gat":
            assert self.att_w is not None and self.val_w is not None
            assert self.att_a is not None
            d = self.cfg.width // self.cfg.heads
            feat = jnp.concatenate([hs, hr, e], axis=-1)  # (E, 3w)
            proj = jax.vmap(self.att_w)(feat).reshape(-1, self.cfg.heads, d)
            score = (jax.nn.leaky_relu(proj) * self.att_a).sum(-1)  # (E, heads) GATv2
            alpha = jraph.segment_softmax(score, RECEIVERS, num_segments=N_VERTICES)
            value = jax.vmap(self.val_w)(hs).reshape(-1, self.cfg.heads, d)
            msg = (alpha[..., None] * value).reshape(-1, self.cfg.width)
        else:
            assert self.msg is not None
            parts = [hs, hr, e]
            if self.cfg.global_node:
                parts.append(jnp.broadcast_to(g, (hs.shape[0], g.shape[0])))
            msg = jax.vmap(self.msg)(jnp.concatenate(parts, axis=-1))
        return _aggr(msg)

    def __call__(
        self, h: Float[Array, "v w"], e: Float[Array, "e w"], g: Array
    ) -> tuple[Float[Array, "v w"], Array]:
        agg = self._aggregate(h, e, g)
        parts = [h, agg]
        if self.cfg.global_node:
            parts.append(jnp.broadcast_to(g, (h.shape[0], g.shape[0])))
        delta = jax.vmap(self.node)(jnp.concatenate(parts, axis=-1))
        h = h + delta if self.cfg.residual else delta
        h = _apply_norm(self.norm, h)
        if self.glob is not None:
            summary = jnp.concatenate([g, h.mean(0), h.max(0)])
            g = g + self.glob(summary)
        return h, g


class GraphNet(eqx.Module):
    node_enc: eqx.nn.Linear
    edge_enc: eqx.nn.Linear
    glob_enc: eqx.nn.Linear
    layers: tuple[_Layer, ...]
    head: eqx.nn.MLP
    cfg: GraphNetConfig = eqx.field(static=True)

    def __init__(self, key: KeyScalar, *, out_dim: int, cfg: GraphNetConfig) -> None:
        from settlrl_learn.graph import EDGE_DIM, GLOBAL_DIM, NODE_DIM

        w = cfg.width
        keys = jax.random.split(key, 4 + cfg.layers)
        self.node_enc = eqx.nn.Linear(NODE_DIM, w, key=keys[0])
        self.edge_enc = eqx.nn.Linear(EDGE_DIM, w, key=keys[1])
        self.glob_enc = eqx.nn.Linear(GLOBAL_DIM, w, key=keys[2])
        self.layers = tuple(_Layer(keys[4 + i], cfg) for i in range(cfg.layers))
        per_pool = w * (3 if cfg.readout == "multi" else 1)
        pooled = per_pool * (cfg.layers if cfg.jk else 1)
        self.head = eqx.nn.MLP(pooled + w, out_dim, w, cfg.head_depth, key=keys[3])
        self.cfg = cfg

    def _pool(self, h: Float[Array, "v w"]) -> Array:
        if self.cfg.readout == "mean":
            return h.mean(0)
        if self.cfg.readout == "sum":
            return h.sum(0)
        return jnp.concatenate([h.mean(0), h.max(0), h.sum(0)])  # multi (PNA-style)

    def __call__(self, s: Sample) -> Float[Array, "out"]:
        h = jax.vmap(self.node_enc)(s.nodes)
        # undirected edges are mirrored in `s.edges`; encode once, share per layer.
        e = jax.vmap(self.edge_enc)(s.edges)
        g = self.glob_enc(s.glob)
        pools = []
        for layer in self.layers:
            h, g = layer(h, e, g)
            if self.cfg.jk:
                pools.append(self._pool(h))
        readout = jnp.concatenate(pools) if self.cfg.jk else self._pool(h)
        return self.head(jnp.concatenate([readout, g]))


# Named presets: ``base`` is plain message passing + mean readout (the closest to
# the legacy ``gnn``); each other flips one lever, plus a stacked ``full``.
PRESETS: dict[str, GraphNetConfig] = {
    "gn_base": GraphNetConfig(
        conv="mpnn", norm="none", global_node=False, readout="mean"
    ),
    "gn_multi": GraphNetConfig(
        conv="mpnn", norm="none", global_node=False, readout="multi"
    ),
    "gn_norm": GraphNetConfig(
        conv="mpnn", norm="layer", global_node=False, readout="multi"
    ),
    "gn_graphnorm": GraphNetConfig(
        conv="mpnn", norm="graph", global_node=False, readout="multi"
    ),
    "gn_global": GraphNetConfig(
        conv="mpnn", norm="layer", global_node=True, readout="multi"
    ),
    "gn_gat": GraphNetConfig(
        conv="gat", norm="layer", global_node=True, readout="multi"
    ),
    "gn_jk": GraphNetConfig(
        conv="mpnn", norm="layer", global_node=True, readout="multi", jk=True
    ),
    "gn_full": GraphNetConfig(
        conv="gat", norm="layer", global_node=True, readout="multi", jk=True
    ),
}
