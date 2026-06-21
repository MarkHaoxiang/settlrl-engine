"""Probe: do the GNNs actually reason about longest *road*, or just count edges?

Longest road is a longest-trail (NP-hard in general) problem, usually hard for a
GNN whose receptive field is its depth. Yet `gn_*` hit R^2 0.99 on the greedy
self-play `road` target. Hypothesis: greedy roads are near-simple paths, where
longest-road == edge-count, so a sum-pooling readout gets it for free without any
path reasoning. The hard cases are where edge-count >> longest-road:

- **branchy** networks (a bush of 15 edges whose longest trail is short);
- **broken** trails (a long path an opponent settlement cuts in the middle).

This generates a controlled mix (simple paths / branchy / broken, edge counts
3..15), labels each with the engine's exact `longest_road_length`, and trains
`gn_base` at several depths. The tell: on the *hard* subset (|count - longest|
>= 2), does the prediction track the true longest road, or the edge count?

    uv run python experiments/0003_neural_board_architectures/road_probe.py
"""

from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from settlrl_engine.board import make_board
from settlrl_engine.board.layout import EDGE_V, N_EDGES, N_VERTICES
from settlrl_engine.mechanics.longest_road import longest_road_length
from settlrl_learn.nn.graph import Sample, board_sample
from settlrl_learn.nn.graphnet import PRESETS, GraphNet

# vertex -> incident (edge, other vertex), host-side.
_EV = np.asarray(EDGE_V)
_ADJ: list[list[tuple[int, int]]] = [[] for _ in range(N_VERTICES)]
for _e, (_a, _b) in enumerate(_EV):
    _ADJ[int(_a)].append((_e, int(_b)))
    _ADJ[int(_b)].append((_e, int(_a)))


def _grow(rng: np.random.Generator, n_edges: int, branchy: bool) -> set[int]:
    """A connected player-0 road set of ``n_edges`` edges. ``branchy`` extends
    from a random network vertex (a bush, short longest trail); otherwise it
    extends the last tip (a near-simple path, longest trail ~ edge count)."""
    start = int(rng.integers(N_VERTICES))
    nodes: set[int] = {start}
    edges: set[int] = set()
    tip = start
    while len(edges) < n_edges:
        base = int(rng.choice(list(nodes))) if branchy else tip
        opts = [(e, o) for (e, o) in _ADJ[base] if e not in edges]
        if not opts:  # stuck: jump to any network vertex with a free edge
            free = [v for v in nodes if any(e not in edges for e, _ in _ADJ[v])]
            if not free:
                break
            base = int(rng.choice(free))
            opts = [(e, o) for (e, o) in _ADJ[base] if e not in edges]
        e, o = opts[int(rng.integers(len(opts)))]
        edges.add(e)
        nodes.add(o)
        tip = o
    return edges


def _sample(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One board's (edge_road, vertex_owner) for a random structure type."""
    n_edges = int(rng.integers(3, 16))
    kind = rng.integers(3)  # 0 path, 1 branchy, 2 broken path
    edges = _grow(rng, n_edges, branchy=(kind == 1))
    er = np.zeros((N_EDGES,), np.uint8)
    er[list(edges)] = 1
    vo = np.zeros((N_VERTICES,), np.uint8)
    if kind == 2 and edges:  # cut the trail: opponent settlements at interior nodes
        verts = list({int(v) for e in edges for v in _EV[e]})
        k = max(1, len(verts) // 5)
        for v in rng.choice(verts, size=min(k, len(verts)), replace=False):
            vo[int(v)] = 2
    return er, vo


def _build(n: int, seed: int) -> tuple[Sample, np.ndarray, np.ndarray]:
    """``n`` boards: featurized Sample, exact longest-road label, edge count."""
    rng = np.random.default_rng(seed)
    layout, state = make_board(batch_size=n, n_players=2, seed=seed)
    pairs = [_sample(rng) for _ in range(n)]
    edge_road = jnp.asarray(np.stack([p[0] for p in pairs]))
    vertex_owner = jnp.asarray(np.stack([p[1] for p in pairs]))
    state = state._replace(edge_road=edge_road, vertex_owner=vertex_owner)
    samples = jax.jit(jax.vmap(lambda lo, st: board_sample(lo, st, jnp.int32(0))))(
        layout, state
    )
    road = jax.jit(
        jax.vmap(
            lambda st: longest_road_length(st.edge_road, st.vertex_owner, jnp.int32(0))
        )
    )(state)
    return (
        cast(Sample, jax.device_get(samples)),
        np.asarray(road, np.float32),
        (np.asarray(edge_road).sum(1)),
    )


def _standardize(tr: Sample, va: Sample) -> tuple[Sample, Sample]:
    def f(field: str) -> tuple[np.ndarray, np.ndarray]:
        x = getattr(tr, field)
        mu = x.mean(0, keepdims=True)
        sd = np.maximum(x.std(0, keepdims=True), 1e-3)
        return ((x - mu) / sd).astype(np.float32), (
            (getattr(va, field) - mu) / sd
        ).astype(np.float32)

    a: dict[str, Any] = {}
    b: dict[str, Any] = {}
    for fld in ("nodes", "edges", "glob", "engineered"):
        a[fld], b[fld] = f(fld)
    return Sample(**a), Sample(**b)


def _fit(depth: int, tr: Sample, ytr: np.ndarray, va: Sample, seed: int) -> np.ndarray:
    cfg = PRESETS["gn_base"]._replace(width=64, layers=depth, head_depth=2)
    model = GraphNet(jax.random.key(seed), out_dim=1, cfg=cfg)
    opt = optax.adamw(1e-3, weight_decay=1e-4)
    st = opt.init(eqx.filter(model, eqx.is_inexact_array))
    xtr = jax.tree.map(jnp.asarray, tr)
    y = jnp.asarray(ytr)

    def loss(m: Any, xs: Any, yy: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((jax.vmap(m)(xs)[:, 0] - yy) ** 2)

    @eqx.filter_jit
    def step(m: Any, s: Any, xs: Any, yy: jnp.ndarray) -> Any:
        ll, g = eqx.filter_value_and_grad(loss)(m, xs, yy)
        up, s = opt.update(g, s, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, up), s, ll

    n, bs = y.shape[0], 256
    rng = np.random.default_rng(seed)
    for _ in range(60):
        order = rng.permutation(n)
        for i in range(0, n - bs + 1, bs):
            idx = jnp.asarray(order[i : i + bs])
            model, st, _ = step(
                model, st, jax.tree.map(lambda x, j=idx: x[j], xtr), y[idx]
            )

    @eqx.filter_jit
    def _predict(m: Any, xs: Any) -> jnp.ndarray:
        return cast(jnp.ndarray, jax.vmap(m)(xs)[:, 0])

    return np.asarray(_predict(model, jax.tree.map(jnp.asarray, va)))


def _r2(pred: np.ndarray, y: np.ndarray) -> float:
    sse = float(np.sum((pred - y) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2)) + 1e-9
    return 1.0 - sse / sst


def main() -> None:
    tr, ytr, _ctr = _build(12000, seed=0)
    va, yva, cva = _build(3000, seed=1)
    tr, va = _standardize(tr, va)
    hard = np.abs(cva - yva) >= 2  # edge-count and longest road disagree
    print(
        f"val n={len(yva)}  longest: mean {yva.mean():.2f} max {yva.max():.0f}  "
        f"hard (|count-longest|>=2): {hard.mean():.0%}"
    )
    print(f"{'depth':>5} {'R2 all':>7} {'R2 hard':>8} {'pred|hard':>10} "
          f"{'true|hard':>10} {'count|hard':>11}")  # fmt: skip
    for depth in (1, 2, 3, 5):
        pred = _fit(depth, tr, ytr, va, seed=0)
        print(
            f"{depth:>5} {_r2(pred, yva):>7.3f} {_r2(pred[hard], yva[hard]):>8.3f} "
            f"{pred[hard].mean():>10.2f} {yva[hard].mean():>10.2f} "
            f"{cva[hard].mean():>11.2f}"
        )


if __name__ == "__main__":
    main()
