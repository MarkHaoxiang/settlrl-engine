from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, UInt8

from catan_engine.dev_cards import DEV_CARD_COUNTS, N_DEV_CARD_TYPES
from catan_engine.layout import NO_INDEX, N_EDGES, N_VERTICES
from catan_engine.resources import N_PLAYERS, N_RESOURCES

# Per-player building stock (standard Catan).
MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4

# Victory points needed to win.
VICTORY_POINTS_TO_WIN = 10

# Player convention: players are 0-indexed (0..N_PLAYERS-1) for current_player,
# player_resources, victory_points and the dev-card arrays. The board occupancy
# arrays (vertex_owner, edge_road) instead store player + 1, reserving 0 for
# "empty". longest_road_owner / largest_army_owner store a 0-indexed player or
# NO_INDEX for "unclaimed".

VertexOwnerArray = UInt8[Array, f"batch vertices={N_VERTICES}"]  # 0=none, 1-4=player
VertexTypeArray = UInt8[
    Array, f"batch vertices={N_VERTICES}"
]  # 0=none, 1=settlement, 2=city
EdgeRoadArray = UInt8[Array, f"batch edges={N_EDGES}"]  # 0=none, 1-4=player
RobberArray = UInt8[Array, "batch"]  # tile index; batch is variable
PlayerResourcesArray = UInt8[
    Array, f"batch players={N_PLAYERS} resources={N_RESOURCES}"
]
VictoryPointsArray = UInt8[Array, f"batch players={N_PLAYERS}"]


class GamePhase(IntEnum):
    """The step of the turn/game that decides which actions are available."""

    SETUP_SETTLEMENT = 0  # place a free starting settlement
    SETUP_ROAD = 1  # place the road next to it
    ROLL = 2  # must roll dice (may play a Knight first)
    DISCARD = 3  # players with >7 cards discard half (after a 7)
    MOVE_ROBBER = 4  # current player moves robber and steals
    MAIN = 5  # build / trade / play dev card / end turn
    GAME_OVER = 6  # a player has reached VICTORY_POINTS_TO_WIN

    def __str__(self) -> str:
        return (
            "SETUP_SETTLEMENT",
            "SETUP_ROAD",
            "ROLL",
            "DISCARD",
            "MOVE_ROBBER",
            "MAIN",
            "GAME_OVER",
        )[self]


class BoardState(NamedTuple):
    """Mutable game state for a batch of games.

    Players are 0-indexed except in vertex_owner / edge_road (player + 1, 0=empty).
    victory_points holds *building* points only (settlement=1, city=2); Longest
    Road, Largest Army and hidden Victory Point cards are added on top when a
    total is needed (see catan_engine.rules.player_total_vp).
    """

    # -- Board occupancy ----------------------------------------------------
    vertex_owner: jax.Array  # (batch, N_VERTICES)
    vertex_type: jax.Array  # (batch, N_VERTICES)
    edge_road: jax.Array  # (batch, N_EDGES)
    robber: jax.Array  # (batch,) tile index
    player_resources: jax.Array  # (batch, N_PLAYERS, N_RESOURCES)
    victory_points: jax.Array  # (batch, N_PLAYERS) building points only

    # -- Development cards --------------------------------------------------
    dev_deck: jax.Array  # (batch, N_DEV_CARD_TYPES) remaining in draw pile
    dev_hand: jax.Array  # (batch, N_PLAYERS, N_DEV_CARD_TYPES) held, unplayed
    knights_played: jax.Array  # (batch, N_PLAYERS) cumulative, for Largest Army

    # -- Turn / flow --------------------------------------------------------
    phase: jax.Array  # (batch,) GamePhase
    current_player: jax.Array  # (batch,) 0-indexed
    turn_number: jax.Array  # (batch,) main-game turn counter (uint16)
    setup_index: jax.Array  # (batch,) 0..2*N_PLAYERS placement counter
    dice_roll: jax.Array  # (batch,) 0 = not rolled this turn, else 2..12
    has_rolled: jax.Array  # (batch,) flag
    dev_played: jax.Array  # (batch,) flag - a dev card played this turn
    dev_bought: jax.Array  # (batch, N_DEV_CARD_TYPES) bought this turn (unplayable)
    free_roads: jax.Array  # (batch,) free roads owed (Road Building)
    pending_discard: jax.Array  # (batch, N_PLAYERS) cards still owed after a 7

    # -- Awards -------------------------------------------------------------
    longest_road_owner: jax.Array  # (batch,) player or NO_INDEX
    largest_army_owner: jax.Array  # (batch,) player or NO_INDEX
    longest_road_len: jax.Array  # (batch,) length held by longest_road_owner

    # -- Randomness ---------------------------------------------------------
    key: jax.Array  # (batch,) PRNG keys for dice rolls and steals


def make_board_state(batch_size: int = 1, key: jax.Array | None = None) -> BoardState:
    B = batch_size
    key = key if key is not None else jax.random.key(0)
    none = jnp.full((B,), NO_INDEX, dtype=jnp.uint8)
    deck = jnp.broadcast_to(
        jnp.array(DEV_CARD_COUNTS, dtype=jnp.uint8), (B, N_DEV_CARD_TYPES)
    ).copy()
    return BoardState(
        vertex_owner=jnp.zeros((B, N_VERTICES), dtype=jnp.uint8),
        vertex_type=jnp.zeros((B, N_VERTICES), dtype=jnp.uint8),
        edge_road=jnp.zeros((B, N_EDGES), dtype=jnp.uint8),
        robber=jnp.zeros((B,), dtype=jnp.uint8),
        player_resources=jnp.zeros((B, N_PLAYERS, N_RESOURCES), dtype=jnp.uint8),
        victory_points=jnp.zeros((B, N_PLAYERS), dtype=jnp.uint8),
        dev_deck=deck,
        dev_hand=jnp.zeros((B, N_PLAYERS, N_DEV_CARD_TYPES), dtype=jnp.uint8),
        knights_played=jnp.zeros((B, N_PLAYERS), dtype=jnp.uint8),
        phase=jnp.full((B,), GamePhase.SETUP_SETTLEMENT, dtype=jnp.uint8),
        current_player=jnp.zeros((B,), dtype=jnp.uint8),
        turn_number=jnp.zeros((B,), dtype=jnp.uint16),
        setup_index=jnp.zeros((B,), dtype=jnp.uint8),
        dice_roll=jnp.zeros((B,), dtype=jnp.uint8),
        has_rolled=jnp.zeros((B,), dtype=jnp.uint8),
        dev_played=jnp.zeros((B,), dtype=jnp.uint8),
        dev_bought=jnp.zeros((B, N_DEV_CARD_TYPES), dtype=jnp.uint8),
        free_roads=jnp.zeros((B,), dtype=jnp.uint8),
        pending_discard=jnp.zeros((B, N_PLAYERS), dtype=jnp.uint8),
        longest_road_owner=none,
        largest_army_owner=none,
        longest_road_len=jnp.zeros((B,), dtype=jnp.uint8),
        key=jax.random.split(key, B),
    )
