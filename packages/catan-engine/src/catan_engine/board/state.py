from enum import IntEnum
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, Key, Num, UInt8

from catan_engine.board.dev_cards import (
    DEV_CARD_COUNTS,
    N_DEV_CARD_TYPES,
    DevCardDeckArray,
    PlayerDevCardHandArray,
    PlayerPlayedKnightsArray,
)
from catan_engine.board.layout import N_EDGES, N_VERTICES
from catan_engine.board.resources import N_PLAYERS, N_RESOURCES, PlayerResourcesArray

# Per-player building stock (standard Catan).
MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4

# vertex_type encoding: 0 = empty, then a settlement is upgraded to a city.
SETTLEMENT = 1
CITY = 2

# Victory points needed to win.
VICTORY_POINTS_TO_WIN = 10

# Sentinel for an unclaimed award: longest_road_owner / largest_army_owner hold a
# 0-indexed player, or this value when no one qualifies (a uint8 out of player
# range).
NO_INDEX = 255

# Player convention: players are 0-indexed (0..N_PLAYERS-1) for current_player,
# player_resources, victory_points and the dev-card arrays. The board occupancy
# arrays (vertex_owner, edge_road) instead store player + 1, reserving 0 for
# "empty". longest_road_owner / largest_army_owner store a 0-indexed player or
# NO_INDEX for "unclaimed".

# jaxtyping aliases for the BoardState arrays. Every field carries a leading
# variable `batch` axis; all remaining axes are fixed board/game constants, so
# the shapes are fully known in advance. Per-vertex/-edge/-player/-dev-card
# arrays reuse the canonical aliases from layout / resources / dev_cards.
VertexOwnerArray = UInt8[Array, f"batch vertices={N_VERTICES}"]  # 0=none, 1-4=player
VertexTypeArray = UInt8[
    Array, f"batch vertices={N_VERTICES}"
]  # 0=none, 1=settlement, 2=city
EdgeRoadArray = UInt8[Array, f"batch edges={N_EDGES}"]  # 0=none, 1-4=player
VictoryPointsArray = UInt8[Array, "batch players"]
PlayerDiscardArray = UInt8[Array, "batch players"]  # cards still owed
TradeCountsArray = UInt8[Array, f"batch resources={N_RESOURCES}"]  # pending offer
# Per-game scalars: phase, current_player, robber tile, counters, flags, awards.
GameScalarArray = UInt8[Array, "batch"]
KeyArray = Key[Array, "batch"]

# Single-game (un-batched) aliases. The rule modules (placement / awards / dice /
# robber / setup / trade / development) run on one game at a time -- under
# jax.vmap the leading `batch` axis is stripped -- so they annotate their array
# params with these batch-free shapes. tests/conftest.py installs a jaxtyping
# import hook that turns these annotations into enforced runtime shape/dtype
# checks (each call and each jit trace) for those modules.
EdgeRoadVec = UInt8[Array, f"edges={N_EDGES}"]
VertexOwnerVec = UInt8[Array, f"vertices={N_VERTICES}"]
VertexTypeVec = UInt8[Array, f"vertices={N_VERTICES}"]
PlayerMaskVec = Bool[Array, "players"]
PlayerU8Vec = UInt8[Array, "players"]  # one game's row of a (batch, players) array
TradeCountsVec = UInt8[Array, f"resources={N_RESOURCES}"]
U8Scalar = UInt8[Array, ""]  # a single uint8 state field (one game's GameScalarArray)
IntScalar = Int[Array, ""]  # a single int index / count (player, vertex, roll, ...)
BoolScalar = Bool[Array, ""]  # a single legality / flag
KeyScalar = Key[Array, ""]  # a single PRNG key


class GamePhase(IntEnum):
    """The step of the turn/game that decides which actions are available."""

    SETUP_SETTLEMENT = 0  # place a free starting settlement
    SETUP_ROAD = 1  # place the road next to it
    ROLL = 2  # must roll dice (may play a Knight first)
    DISCARD = 3  # players with >7 cards discard half (after a 7)
    MOVE_ROBBER = 4  # current player moves robber and steals
    MAIN = 5  # build / trade / play dev card / end turn
    TRADE_RESPONSE = 6  # the proposed-to player accepts or rejects a trade
    # Reserved/unused: the engine never assigns this phase. Game-over is detected
    # out-of-band by the victory-point total (>= VICTORY_POINTS_TO_WIN) in
    # env/batched.py, not by a phase transition.
    GAME_OVER = 7

    def __str__(self) -> str:
        return (
            "SETUP_SETTLEMENT",
            "SETUP_ROAD",
            "ROLL",
            "DISCARD",
            "MOVE_ROBBER",
            "MAIN",
            "TRADE_RESPONSE",
            "GAME_OVER",
        )[self]


class BoardState(NamedTuple):
    """Mutable game state for a batch of games.

    Players are 0-indexed except in vertex_owner / edge_road (player + 1, 0=empty).
    victory_points holds *building* points only (settlement=1, city=2); Longest
    Road, Largest Army and hidden Victory Point cards are added on top when a
    total is needed (see catan_engine.mechanics.action.player_total_vp).

    The per-player arrays are sized to the seated player count (2..N_PLAYERS),
    read back via the ``n_players`` property; every game in a batch seats the
    same number of players.
    """

    # -- Board occupancy ----------------------------------------------------
    vertex_owner: VertexOwnerArray
    vertex_type: VertexTypeArray
    edge_road: EdgeRoadArray
    robber: GameScalarArray  # tile index
    player_resources: PlayerResourcesArray
    victory_points: VictoryPointsArray  # building points only

    # -- Development cards --------------------------------------------------
    dev_deck: DevCardDeckArray  # remaining in draw pile
    dev_hand: PlayerDevCardHandArray  # held, unplayed
    knights_played: PlayerPlayedKnightsArray  # cumulative, for Largest Army

    # -- Turn / flow --------------------------------------------------------
    phase: GameScalarArray  # GamePhase
    current_player: GameScalarArray  # 0-indexed
    setup_index: GameScalarArray  # 0..2*n_players placement counter
    dice_roll: GameScalarArray  # 0 = not rolled this turn, else 2..12
    has_rolled: GameScalarArray  # flag
    dev_played: GameScalarArray  # flag - a dev card played this turn
    dev_bought: DevCardDeckArray  # bought this turn (unplayable)
    free_roads: GameScalarArray  # free roads owed (Road Building)
    pending_discard: PlayerDiscardArray  # cards still owed after a 7
    trade_partner: GameScalarArray  # proposed-to player, or NO_INDEX
    trade_give: TradeCountsArray  # per-resource counts the proposer gives
    trade_receive: TradeCountsArray  # per-resource counts the proposer asks for

    # -- Awards -------------------------------------------------------------
    longest_road_owner: GameScalarArray  # player or NO_INDEX
    largest_army_owner: GameScalarArray  # player or NO_INDEX
    longest_road_len: GameScalarArray  # length held by longest_road_owner

    # -- Randomness ---------------------------------------------------------
    key: KeyArray  # PRNG keys for dice rolls and steals

    @property
    def n_players(self) -> int:
        """Seated players (2..N_PLAYERS), read off the player axis (static)."""
        return self.victory_points.shape[-1]


def to_u8(x: Num[Array, "*s"]) -> UInt8[Array, "*s"]:
    """Saturating cast to uint8 (clip to ``[0, 255]``)."""
    return jnp.clip(x, 0, 255).astype(jnp.uint8)


def tree_select(mask: BoolScalar, a: BoardState, b: BoardState) -> BoardState:
    """Per-leaf ``where(mask, a, b)`` over two single-game states.

    The branchless-application primitive for the action layer: an action always
    computes its candidate next state, then commits it only where legal.
    """
    return cast(
        BoardState, jax.tree_util.tree_map(lambda x, y: jnp.where(mask, x, y), a, b)
    )


def make_board_state(
    batch_size: int = 1, key: KeyScalar | None = None, n_players: int = N_PLAYERS
) -> BoardState:
    if not 2 <= n_players <= N_PLAYERS:
        raise ValueError(f"n_players must be in [2, {N_PLAYERS}], got {n_players}")
    B, P = batch_size, n_players
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
        player_resources=jnp.zeros((B, P, N_RESOURCES), dtype=jnp.uint8),
        victory_points=jnp.zeros((B, P), dtype=jnp.uint8),
        dev_deck=deck,
        dev_hand=jnp.zeros((B, P, N_DEV_CARD_TYPES), dtype=jnp.uint8),
        knights_played=jnp.zeros((B, P), dtype=jnp.uint8),
        phase=jnp.full((B,), GamePhase.SETUP_SETTLEMENT, dtype=jnp.uint8),
        current_player=jnp.zeros((B,), dtype=jnp.uint8),
        setup_index=jnp.zeros((B,), dtype=jnp.uint8),
        dice_roll=jnp.zeros((B,), dtype=jnp.uint8),
        has_rolled=jnp.zeros((B,), dtype=jnp.uint8),
        dev_played=jnp.zeros((B,), dtype=jnp.uint8),
        dev_bought=jnp.zeros((B, N_DEV_CARD_TYPES), dtype=jnp.uint8),
        free_roads=jnp.zeros((B,), dtype=jnp.uint8),
        pending_discard=jnp.zeros((B, P), dtype=jnp.uint8),
        trade_partner=none,
        trade_give=jnp.zeros((B, N_RESOURCES), dtype=jnp.uint8),
        trade_receive=jnp.zeros((B, N_RESOURCES), dtype=jnp.uint8),
        longest_road_owner=none,
        largest_army_owner=none,
        longest_road_len=jnp.zeros((B,), dtype=jnp.uint8),
        key=jax.random.split(key, B),
    )
