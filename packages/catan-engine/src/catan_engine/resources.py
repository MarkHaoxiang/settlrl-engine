import jax
import jax.numpy as jnp
from jaxtyping import Array, UInt8

N_PLAYERS = 4
N_RESOURCES = 5  # Tile indices 0-4: SHEEP, WHEAT, WOOD, BRICK, ORE (excludes DESERT)
BANK_INITIAL = 19  # starting count of each resource type in the bank

# Build costs, in Tile resource order [sheep, wheat, wood, brick, ore].
ROAD_COST: tuple[int, ...] = (0, 0, 1, 1, 0)
SETTLEMENT_COST: tuple[int, ...] = (1, 1, 1, 1, 0)
CITY_COST: tuple[int, ...] = (0, 2, 0, 0, 3)

PlayerResourcesArray = UInt8[
    Array, f"batch players={N_PLAYERS} resources={N_RESOURCES}"
]


def compute_bank_resources(player_resources: jax.Array) -> jax.Array:
    """Return remaining bank stock for each resource.

    Args:
        player_resources: uint8 array of shape (batch, players, resources).

    Returns:
        uint8 array of shape (batch, resources).
    """
    held = player_resources.sum(axis=-2, dtype=jnp.uint8)
    return (jnp.full_like(held, BANK_INITIAL) - held).astype(jnp.uint8)
