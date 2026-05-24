from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, UInt8

from catan_engine.board import N_EDGES, N_VERTICES
from catan_engine.resources import N_PLAYERS, N_RESOURCES

VertexOwnerArray = UInt8[Array, f"batch vertices={N_VERTICES}"]  # 0=none, 1-4=player
VertexTypeArray = UInt8[Array, f"batch vertices={N_VERTICES}"]   # 0=none, 1=settlement, 2=city
EdgeRoadArray = UInt8[Array, f"batch edges={N_EDGES}"]           # 0=none, 1-4=player
RobberArray = UInt8[Array, "batch"]                              # tile index; batch is variable
PlayerResourcesArray = UInt8[Array, f"batch players={N_PLAYERS} resources={N_RESOURCES}"]
VictoryPointsArray = UInt8[Array, f"batch players={N_PLAYERS}"]


class BoardState(NamedTuple):
    """Mutable game state on the board."""

    vertex_owner: jax.Array
    vertex_type: jax.Array
    edge_road: jax.Array
    robber: jax.Array
    player_resources: jax.Array  # (batch, N_PLAYERS, N_RESOURCES)
    victory_points: jax.Array    # (batch, N_PLAYERS)


def make_board_state(batch_size: int = 1) -> BoardState:
    B = batch_size
    return BoardState(
        vertex_owner=jnp.zeros((B, N_VERTICES), dtype=jnp.uint8),
        vertex_type=jnp.zeros((B, N_VERTICES), dtype=jnp.uint8),
        edge_road=jnp.zeros((B, N_EDGES), dtype=jnp.uint8),
        robber=jnp.zeros((B,), dtype=jnp.uint8),
        player_resources=jnp.zeros((B, N_PLAYERS, N_RESOURCES), dtype=jnp.uint8),
        victory_points=jnp.zeros((B, N_PLAYERS), dtype=jnp.uint8),
    )
