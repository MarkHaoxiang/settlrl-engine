"""Board featurization: one game's position as a flat vector for one player.

The vector concatenates the player's own block, the elementwise max and mean
over the opponents' blocks, and a global block — so its width is independent
of the seated player count. Like the heuristic, it is only ever computed on
concrete (sampled) worlds, so reading "hidden" fields stays honest.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from settlrl_agents.internal.feature_engineering import tile_pips, vertex_pips
from settlrl_engine.board.layout import (
    EDGE_V,
    N_TILES,
    N_VERTICES,
    PORT_V,
    TILE_V,
    BoardLayout,
)
from settlrl_engine.board.state import (
    CITY,
    MAX_SETTLEMENTS,
    SETTLEMENT,
    BoardState,
    GamePhase,
    IntScalar,
)
from settlrl_engine.mechanics.common import player_total_vp


def _player_block(
    layout: BoardLayout, state: BoardState, q: IntScalar
) -> Float[Array, "block"]:
    """One player's standing: production, holdings, expansion, awards."""
    own_vertex = state.vertex_owner == q + 1
    settlements = (own_vertex & (state.vertex_type == SETTLEMENT)).sum()
    cities = (own_vertex & (state.vertex_type == CITY)).sum()
    own_road = state.edge_road == q + 1
    roads = own_road.sum()

    # Production per resource, robber-aware, city double.
    pips = tile_pips(layout.tile_number) * (jnp.arange(N_TILES) != state.robber)
    weight = (
        (own_vertex[TILE_V] * state.vertex_type[TILE_V]).sum(axis=1).astype(jnp.float32)
    )
    per_res = (
        jnp.zeros((5,), jnp.float32)
        .at[layout.tile_resource.astype(jnp.int32) % 5]
        .add(pips * weight)
    )

    res = state.player_resources[q].astype(jnp.float32)
    hand = res.sum()

    # Best settlement spot reachable right now (the heuristic's expansion term).
    occ = state.vertex_owner > 0
    u, v = EDGE_V[:, 0], EDGE_V[:, 1]
    nb_occ = jnp.zeros((N_VERTICES,), bool).at[u].max(occ[v]).at[v].max(occ[u])
    touched = jnp.zeros((N_VERTICES,), bool).at[u].max(own_road).at[v].max(own_road)
    spot = ~occ & ~nb_occ & touched & (settlements < MAX_SETTLEMENTS)
    best_spot = jnp.max(jnp.where(spot, vertex_pips(layout.tile_number), 0.0))

    port_alloc = layout.port_allocation.astype(jnp.int32)
    on_port = (state.vertex_owner[PORT_V] == q + 1).any(axis=1)
    ports_2to1 = (
        jnp.zeros((5,), bool).at[port_alloc % 5].max(on_port & (port_alloc < 5))
    )
    port_3to1 = jnp.any(on_port & (port_alloc == 5))

    return jnp.concatenate(
        [
            jnp.asarray(
                [
                    player_total_vp(state, q),
                    settlements,
                    cities,
                    roads,
                    hand,
                    jnp.maximum(hand - 7.0, 0.0),
                    state.dev_hand[q].astype(jnp.float32).sum(),
                    state.knights_played[q],
                    (per_res > 0).sum(),
                    best_spot,
                    jnp.any(spot),
                    state.longest_road_owner == q,
                    state.largest_army_owner == q,
                    port_3to1,
                ],
                dtype=jnp.float32,
            ),
            res,
            state.dev_hand[q].astype(jnp.float32),
            per_res,
            ports_2to1.astype(jnp.float32),
        ]
    )


def _global_block(layout: BoardLayout, state: BoardState) -> Float[Array, "block"]:
    """Position-wide facts shared by every seat."""
    held = state.player_resources.astype(jnp.float32).sum(axis=0)
    phase = jax.nn.one_hot(state.phase.astype(jnp.int32), len(GamePhase))
    return jnp.concatenate(
        [
            19.0 - held,  # bank per resource
            jnp.asarray(
                [
                    state.dev_deck.astype(jnp.float32).sum(),
                    state.has_rolled,
                    tile_pips(layout.tile_number)[state.robber],
                ],
                dtype=jnp.float32,
            ),
            phase,
        ]
    )


def features(
    layout: BoardLayout, state: BoardState, player: IntScalar
) -> Float[Array, "features"]:
    """The position from ``player``'s seat: own block, opponent max and mean
    blocks, and the global block, concatenated (width is player-count
    invariant)."""
    players = jnp.arange(state.n_players)
    blocks = jax.vmap(lambda q: _player_block(layout, state, q))(players)
    own = blocks[player]
    other = players != player
    other_max = jnp.max(jnp.where(other[:, None], blocks, -jnp.inf), axis=0)
    other_mean = jnp.sum(jnp.where(other[:, None], blocks, 0.0), axis=0) / jnp.maximum(
        other.sum(), 1
    )
    return jnp.concatenate([own, other_max, other_mean, _global_block(layout, state)])


def _feature_dim() -> int:
    from settlrl_engine.board import make_board

    layout, state = make_board(batch_size=1, n_players=2)
    shape = jax.eval_shape(
        features,
        jax.tree.map(lambda x: x[0], layout),
        jax.tree.map(lambda x: x[0], state),
        jnp.int32(0),
    )
    return int(shape.shape[0])


FEATURE_DIM = _feature_dim()
"""Width of :func:`features` (player-count invariant)."""
