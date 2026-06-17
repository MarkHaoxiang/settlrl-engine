"""Board -> graph featurization for the architecture benchmark.

The board graph has fixed topology (54 vertices, 72 edges; the senders/receivers
never change shape), so a sample carries only per-node, per-edge and global
*features* — the static incidence lives here as module constants. Perspective is
one player: ownership and the global player summary are relative (own vs. the
best opponent), so the same featurization serves any seat. Computed on the true
board (this benchmark measures representation capacity, not belief handling).

Three readouts of the same position are produced per sample so architectures can
be compared on a level field:

- ``nodes`` / ``edges`` / ``glob`` — the graph (GNN, DeepSet, flat-MLP all read
  this);
- ``engineered`` — the hand-tuned :mod:`settlrl_learn.features` vector, the
  baseline a learned representation must beat.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from settlrl_agents.internal.feature_engineering import tile_pips
from settlrl_engine.board.layout import (
    EDGE_V,
    N_EDGES,
    N_TILES,
    N_VERTICES,
    PORT_V,
    TILE_V,
    BoardLayout,
)
from settlrl_engine.board.resources import N_RESOURCES
from settlrl_engine.board.state import (
    CITY,
    SETTLEMENT,
    BoardState,
    GamePhase,
    IntScalar,
)
from settlrl_engine.mechanics.common import player_total_vp
from settlrl_learn.features import FEATURE_DIM
from settlrl_learn.features import features as engineered_features

# --- static topology (shared by every sample) ---

# Board edges are undirected; the message-passing graph uses both directions.
_edge_np = np.asarray(EDGE_V)
SENDERS = jnp.asarray(np.concatenate([_edge_np[:, 0], _edge_np[:, 1]]))
RECEIVERS = jnp.asarray(np.concatenate([_edge_np[:, 1], _edge_np[:, 0]]))
N_DIR_EDGES = 2 * N_EDGES

# vertex-tile incidence (V, T): vertex v touches tile t.
_inc = np.zeros((N_VERTICES, N_TILES), dtype=np.float32)
for _t, _vs in enumerate(np.asarray(TILE_V)):
    _inc[_vs, _t] = 1.0
VERTEX_TILE = jnp.asarray(_inc)

# vertex-port incidence (V, P): vertex v sits on port p.
_pinc = np.zeros((N_VERTICES, PORT_V.shape[0]), dtype=np.float32)
for _p, _vs in enumerate(np.asarray(PORT_V)):
    _pinc[_vs, _p] = 1.0
VERTEX_PORT = jnp.asarray(_pinc)


class Sample(NamedTuple):
    """One featurized position. ``nodes``/``edges``/``glob`` are the graph;
    ``engineered`` the hand-tuned baseline vector."""

    nodes: Float[Array, f"v={N_VERTICES} node_f"]
    edges: Float[Array, f"e={N_DIR_EDGES} edge_f"]
    glob: Float[Array, "global_f"]
    engineered: Float[Array, f"feat={FEATURE_DIM}"]


def _node_features(
    layout: BoardLayout, state: BoardState, p: IntScalar
) -> Float[Array, f"v={N_VERTICES} node_f"]:
    owner = state.vertex_owner.astype(jnp.int32)
    mine = owner == p + 1
    occupied = owner > 0
    owner_oh = jnp.stack([~occupied, mine, occupied & ~mine], axis=1).astype(
        jnp.float32
    )
    building = jnp.stack(
        [state.vertex_type == SETTLEMENT, state.vertex_type == CITY], axis=1
    ).astype(jnp.float32)

    pips = tile_pips(layout.tile_number) * (jnp.arange(N_TILES) != state.robber)
    res_oh = jax.nn.one_hot(layout.tile_resource.astype(jnp.int32) % N_RESOURCES, 5)
    node_prod = VERTEX_TILE @ (res_oh * pips[:, None])  # (V, 5)
    robber_adj = (VERTEX_TILE[:, state.robber] > 0).astype(jnp.float32)[:, None]

    alloc = layout.port_allocation.astype(jnp.int32)
    port_res_oh = jax.nn.one_hot(alloc % N_RESOURCES, 5) * (alloc < 5)[:, None]
    v_2to1 = jnp.clip(VERTEX_PORT @ port_res_oh, 0.0, 1.0)  # (V, 5)
    v_3to1 = jnp.clip(VERTEX_PORT @ (alloc == 5).astype(jnp.float32), 0.0, 1.0)[:, None]

    return jnp.concatenate(
        [owner_oh, building, node_prod, robber_adj, v_2to1, v_3to1], axis=1
    )


def _edge_features(
    state: BoardState, p: IntScalar
) -> Float[Array, f"e={N_DIR_EDGES} edge_f"]:
    owner = state.edge_road.astype(jnp.int32)
    mine = owner == p + 1
    built = owner > 0
    one_dir = jnp.stack([~built, mine, built & ~mine], axis=1).astype(jnp.float32)
    return jnp.concatenate([one_dir, one_dir], axis=0)  # mirror both directions


def _global_features(
    layout: BoardLayout, state: BoardState, p: IntScalar
) -> Float[Array, "global_f"]:
    held = state.player_resources.astype(jnp.float32).sum(axis=0)
    bank = 19.0 - held
    phase = jax.nn.one_hot(state.phase.astype(jnp.int32), len(GamePhase))

    players = jnp.arange(state.n_players)
    vps = jax.vmap(lambda q: player_total_vp(state, q))(players).astype(jnp.float32)
    hands = state.player_resources.astype(jnp.float32).sum(axis=1)
    other = players != p
    opp_max_vp = jnp.max(jnp.where(other, vps, -jnp.inf))
    opp_hand = jnp.sum(jnp.where(other, hands, 0.0))

    scalars = jnp.asarray(
        [
            state.dev_deck.astype(jnp.float32).sum(),
            state.has_rolled.astype(jnp.float32),
            tile_pips(layout.tile_number)[state.robber],
            vps[p],
            hands[p],
            state.dev_hand[p].astype(jnp.float32).sum(),
            state.knights_played[p].astype(jnp.float32),
            (state.longest_road_owner == p).astype(jnp.float32),
            (state.largest_army_owner == p).astype(jnp.float32),
            (state.current_player.astype(jnp.int32) == p).astype(jnp.float32),
            opp_max_vp,
            opp_hand,
            jnp.float32(state.n_players),
        ],
        dtype=jnp.float32,
    )
    return jnp.concatenate([bank, scalars, phase])


def board_sample(layout: BoardLayout, state: BoardState, p: IntScalar) -> Sample:
    """Featurize one position from ``p``'s perspective into a :class:`Sample`."""
    return Sample(
        nodes=_node_features(layout, state, p),
        edges=_edge_features(state, p),
        glob=_global_features(layout, state, p),
        engineered=engineered_features(layout, state, p),
    )


def _dims() -> tuple[int, int, int]:
    from settlrl_engine.board import make_board

    layout, state = make_board(batch_size=1, n_players=2)
    one = jax.tree.map(lambda x: x[0], (layout, state))
    s = jax.eval_shape(board_sample, one[0], one[1], jnp.int32(0))
    return int(s.nodes.shape[1]), int(s.edges.shape[1]), int(s.glob.shape[0])


NODE_DIM, EDGE_DIM, GLOBAL_DIM = _dims()
"""Feature widths (the topology dims ``N_VERTICES`` / ``N_DIR_EDGES`` are fixed)."""
