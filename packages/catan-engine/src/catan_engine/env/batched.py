"""RL environment entry point -- a *batched* PettingZoo-AEC-style Catan env.

Two layers live here:

1. The thin functional interface ``step`` / ``available`` -- ``jit(vmap(...))``
   over the ``lax.switch`` dispatchers in ``action`` -- which apply one
   ``(ActionType, ActionParams)`` action per game across a whole batch and
   return the new ``BoardState`` plus ``(batch,)`` ``ActionResult`` codes.

2. ``BatchedCatanEnv`` -- a stateful environment that wraps that core in the
   PettingZoo `AEC API <https://pettingzoo.farama.org/api/aec/>`_, adapted to
   run ``batch_size`` games in parallel. Catan is turn-based, so the AEC model
   (one agent acts at a time) fits directly: the batch axis is independent
   games and the active agent in lane ``b`` is that game's ``current_player``.

Batched adaptations of the AEC surface (documented per attribute below):

- ``agent_selection`` is a ``(B,)`` int array (the acting player per lane), not
  a single agent id -- different lanes may be on different players' turns. In
  the DISCARD phase it points at the next player who still owes cards.
- ``rewards`` / ``terminations`` / ``truncations`` are ``(B, N_PLAYERS)`` arrays
  rather than ``{agent: value}`` dicts.
- ``infos`` is a single dict of batched arrays (the acting agent varies per
  lane), carrying the action mask under ``"action_mask"`` per AEC convention.
- ``observe(agent)`` returns that player's *partial* view across all lanes
  (own hand / dev cards in full; only public counts for opponents).
- Terminated lanes **auto-reset**: a finished game is immediately replaced with
  a fresh random board so the batch stays fully active for rollouts. The
  returned observation is the reset game's; ``rewards`` / ``terminations``
  reflect the terminal transition that just occurred.

Spaces are described with the lightweight ``Discrete`` / ``Box`` descriptors
below (the package deliberately avoids a hard ``gymnasium`` dependency); a
caller can wrap them in real ``gymnasium`` spaces if needed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp

from catan_engine.mechanics.action import (
    ActionParams,
    ActionResult,
    ActionType,
    ActionTypeArray,
    Mask,
    N_ACTION_TYPES,
    ResultCode,
    _build_city_avail,
    _build_road_avail,
    _build_settlement_avail,
    _buy_dev_avail,
    _end_turn_avail,
    _knight_avail,
    _maritime_avail,
    _monopoly_avail,
    _move_robber_avail,
    _road_building_avail,
    _roll_avail,
    _setup_road_avail,
    _setup_settlement_avail,
    _yop_avail,
    action_available,
    apply_action,
    player_total_vp,
)
from catan_engine.board import Board
from catan_engine.board.dev_cards import N_DEV_CARD_TYPES
from catan_engine.board.layout import (
    N_EDGES,
    N_PORTS,
    N_TILES,
    N_VERTICES,
    BoardLayout,
    desert_tile,
    make_layout,
)
from catan_engine.board.resources import (
    BANK_INITIAL,
    N_PLAYERS,
    N_RESOURCES,
    compute_bank_resources,
)
from catan_engine.board.state import (
    VICTORY_POINTS_TO_WIN,
    BoardState,
    GamePhase,
    make_board_state,
)

__all__ = [
    "ActionParams",
    "ActionResult",
    "ActionType",
    "N_ACTION_TYPES",
    "BatchedCatanEnv",
    "Box",
    "Discrete",
    "Observation",
    "step",
    "available",
]

# ---------------------------------------------------------------------------
# Functional core: one batched (ActionType, ActionParams) action per game.
# ---------------------------------------------------------------------------
_step = jax.jit(jax.vmap(apply_action, in_axes=(0, 0, 0, 0)))
_available = jax.jit(jax.vmap(action_available, in_axes=(0, 0, 0, 0)))


def step(
    board: Board, action_type: ActionTypeArray, params: ActionParams
) -> tuple[BoardState, ResultCode]:
    """Apply one (batched) action per game; return (new state, ActionResult codes)."""
    new_state, result = _step(board[0], board[1], action_type, params)
    return new_state, result


def available(board: Board, action_type: ActionTypeArray, params: ActionParams) -> Mask:
    """``(batch,)`` legality mask for the chosen action per game (no state change)."""
    return cast(Mask, _available(board[0], board[1], action_type, params))


# ---------------------------------------------------------------------------
# Batched derived quantities used by the environment.
# ---------------------------------------------------------------------------
Observation = dict[str, jax.Array]

# Static parameter domains for the legality sweeps.
_VERTEX_DOM = jnp.arange(N_VERTICES, dtype=jnp.int32)
_EDGE_DOM = jnp.arange(N_EDGES, dtype=jnp.int32)
_TILE_DOM = jnp.arange(N_TILES, dtype=jnp.int32)
_RES_DOM = jnp.arange(N_RESOURCES, dtype=jnp.int32)
_VICTIM_DOM = jnp.arange(-1, N_PLAYERS, dtype=jnp.int32)  # -1 = steal from no one


def _total_vp_single(state: BoardState) -> jax.Array:
    """Total VP (buildings + awards + VP cards) for every player in one game."""
    players = jnp.arange(N_PLAYERS, dtype=jnp.int32)
    return jax.vmap(lambda p: player_total_vp(state, p))(players)


_total_vp_b = jax.jit(jax.vmap(_total_vp_single))


def _agent_selection_single(state: BoardState) -> jax.Array:
    """Acting player for one game: the discarder during DISCARD, else current."""
    owes = state.pending_discard > 0
    discarder = jnp.argmax(owes).astype(jnp.int32)
    in_discard = state.phase == jnp.uint8(GamePhase.DISCARD)
    return jnp.where(in_discard, discarder, state.current_player.astype(jnp.int32))


_agent_selection_b = jax.jit(jax.vmap(_agent_selection_single))


# Single-game legality cores by parameter shape (see action.py).
_IndexAvail = Callable[[BoardLayout, BoardState, jax.Array], jax.Array]
_PairAvail = Callable[[BoardLayout, BoardState, tuple[jax.Array, jax.Array]], jax.Array]


def _action_type_mask_single(layout: BoardLayout, state: BoardState) -> jax.Array:
    """Per-action-type legality for the acting player in one game (any params)."""

    def any_idx(avail: _IndexAvail, dom: jax.Array) -> jax.Array:
        return jnp.any(jax.vmap(lambda i: avail(layout, state, i))(dom))

    def any_robber(avail: _PairAvail) -> jax.Array:
        # Legal if some (tile, victim) pair -- including victim == -1 -- is valid.
        return jnp.any(
            jax.vmap(
                lambda t: jnp.any(
                    jax.vmap(lambda v: avail(layout, state, (t, v)))(_VICTIM_DOM)
                )
            )(_TILE_DOM)
        )

    def any_two_res(avail: _PairAvail) -> jax.Array:
        return jnp.any(
            jax.vmap(
                lambda a: jnp.any(
                    jax.vmap(lambda b: avail(layout, state, (a, b)))(_RES_DOM)
                )
            )(_RES_DOM)
        )

    # Discard enumerates a resource vector; reduce to its precondition instead.
    discard = (state.phase == jnp.uint8(GamePhase.DISCARD)) & jnp.any(
        state.pending_discard > 0
    )

    flags = [
        any_idx(_setup_settlement_avail, _VERTEX_DOM),
        any_idx(_setup_road_avail, _EDGE_DOM),
        _roll_avail(layout, state, None),
        discard,
        any_robber(_move_robber_avail),
        any_idx(_build_road_avail, _EDGE_DOM),
        any_idx(_build_settlement_avail, _VERTEX_DOM),
        any_idx(_build_city_avail, _VERTEX_DOM),
        _buy_dev_avail(layout, state, None),
        any_robber(_knight_avail),
        _road_building_avail(layout, state, None),
        any_two_res(_yop_avail),
        any_idx(_monopoly_avail, _RES_DOM),
        any_two_res(_maritime_avail),
        _end_turn_avail(layout, state, None),
    ]
    return jnp.stack(flags)  # (N_ACTION_TYPES,) bool, in ActionType order


_action_type_mask_b = jax.jit(jax.vmap(_action_type_mask_single))


_BatchedMask = Callable[[BoardLayout, BoardState], jax.Array]


def _index_mask_factory(avail: _IndexAvail, n: int) -> _BatchedMask:
    """Batched ``(B, n)`` legality sweep over a single index parameter."""
    dom = jnp.arange(n, dtype=jnp.int32)

    def single(layout: BoardLayout, state: BoardState) -> jax.Array:
        return jax.vmap(lambda i: avail(layout, state, i))(dom)

    return cast(_BatchedMask, jax.jit(jax.vmap(single)))


def _robber_tile_mask_factory(avail: _PairAvail) -> _BatchedMask:
    """Batched ``(B, N_TILES)`` mask: a tile is legal if some victim choice works."""

    def single(layout: BoardLayout, state: BoardState) -> jax.Array:
        return jax.vmap(
            lambda t: jnp.any(
                jax.vmap(lambda v: avail(layout, state, (t, v)))(_VICTIM_DOM)
            )
        )(_TILE_DOM)

    return cast(_BatchedMask, jax.jit(jax.vmap(single)))


# ActionType -> batched legality sweep over that action's primary index domain.
# (Multi-parameter / parameterless actions are absent; use ``action_mask`` /
# ``available`` for those.)
_INDEX_MASKS = {
    ActionType.SETUP_SETTLEMENT: _index_mask_factory(
        _setup_settlement_avail, N_VERTICES
    ),
    ActionType.SETUP_ROAD: _index_mask_factory(_setup_road_avail, N_EDGES),
    ActionType.BUILD_ROAD: _index_mask_factory(_build_road_avail, N_EDGES),
    ActionType.BUILD_SETTLEMENT: _index_mask_factory(
        _build_settlement_avail, N_VERTICES
    ),
    ActionType.BUILD_CITY: _index_mask_factory(_build_city_avail, N_VERTICES),
    ActionType.PLAY_MONOPOLY: _index_mask_factory(_monopoly_avail, N_RESOURCES),
    ActionType.MOVE_ROBBER: _robber_tile_mask_factory(_move_robber_avail),
    ActionType.PLAY_KNIGHT: _robber_tile_mask_factory(_knight_avail),
}


# ---------------------------------------------------------------------------
# Lightweight space descriptors (no gymnasium dependency).
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Discrete:
    """A discrete space of ``n`` values ``{0, ..., n - 1}``."""

    n: int


@dataclasses.dataclass(frozen=True)
class Box:
    """A bounded integer array space (shape excludes the leading batch axis)."""

    shape: tuple[int, ...]
    dtype: str
    low: int
    high: int


Space = Discrete | Box


# ---------------------------------------------------------------------------
# The batched AEC environment.
# ---------------------------------------------------------------------------
_BATCH = jnp.int32  # dtype alias for readability


class BatchedCatanEnv:
    """A batch of Catan games behind a (batched) PettingZoo-AEC interface.

    Args:
        batch_size: number of games run in parallel (the leading array axis).
        seed: PRNG seed for the initial boards and auto-reset randomness.
        reward: ``"sparse"`` (+1 to the winner on the terminal step, 0 else) or
            ``"vp_delta"`` (each player's change in total VP this step).
        auto_reset: when True (default), a lane that terminates is replaced with a
            fresh game on the same step; when False, terminated lanes freeze (and
            keep reporting ``terminations`` True) for callers that manage episode
            boundaries themselves (e.g. the single-game AEC wrapper).

    The action consumed by :meth:`step` is the engine's
    ``(action_type, ActionParams)`` pair with a leading batch axis -- one action
    per lane, applied to that lane's acting player. See module docstring for the
    batched adaptations of the AEC attributes.
    """

    metadata = {"name": "catan_batched_aec_v0"}

    def __init__(
        self,
        batch_size: int = 1,
        seed: int = 0,
        reward: str = "sparse",
        auto_reset: bool = True,
    ) -> None:
        if reward not in ("sparse", "vp_delta"):
            raise ValueError(f"reward must be 'sparse' or 'vp_delta', got {reward!r}")
        self.batch_size = batch_size
        self.reward_mode = reward
        self.auto_reset = auto_reset
        self.possible_agents = [f"player_{i}" for i in range(N_PLAYERS)]
        self.agents = list(self.possible_agents)
        self.num_agents = N_PLAYERS
        self._seed = seed
        self.reset(seed)

    # -- AEC lifecycle ----------------------------------------------------

    def reset(self, seed: int | None = None, options: object = None) -> None:
        """Start a fresh batch of games (AEC ``reset``; returns ``None``).

        Read the starting position with :meth:`observe` / :meth:`last`.
        """
        if seed is not None:
            self._seed = seed
        self._key = jax.random.key(self._seed)
        self._key, k_layout, k_state = jax.random.split(self._key, 3)
        self._layout = make_layout(self.batch_size, key=k_layout)
        self._state = make_board_state(self.batch_size, key=k_state)
        self._state = self._state._replace(
            robber=desert_tile(self._layout.tile_resource)
        )
        B, P = self.batch_size, N_PLAYERS
        self._reward = jnp.zeros((B, P), dtype=jnp.float32)
        self._terminations = jnp.zeros((B, P), dtype=jnp.bool_)
        self._truncations = jnp.zeros((B, P), dtype=jnp.bool_)
        self._result = jnp.full((B,), int(ActionResult.SUCCESS), dtype=jnp.int32)
        self._vps = cast(jax.Array, _total_vp_b(self._state))
        self.agents = list(self.possible_agents)

    def step(self, action_type: ActionTypeArray, params: ActionParams) -> None:
        """Apply one action per lane to its acting player (AEC ``step``).

        ``action_type`` is a ``(B,)`` int array of :class:`ActionType` codes and
        ``params`` an :class:`ActionParams` with batched leaves. Terminated lanes
        auto-reset; the resulting reward / termination reflect the transition
        that just happened, while the next observation is the reset game's.
        """
        at = jnp.asarray(action_type, dtype=jnp.int32)
        new_state, result = _step(self._layout, self._state, at, params)
        vps_after = cast(jax.Array, _total_vp_b(new_state))
        done_lane = jnp.any(vps_after >= VICTORY_POINTS_TO_WIN, axis=1)  # (B,)

        self._reward = self._compute_reward(self._vps, vps_after, done_lane)
        self._terminations = jnp.broadcast_to(
            done_lane[:, None], (self.batch_size, N_PLAYERS)
        )
        self._truncations = jnp.zeros((self.batch_size, N_PLAYERS), dtype=jnp.bool_)
        self._result = result

        self._layout, self._state = self._auto_reset(self._layout, new_state, done_lane)
        self._vps = cast(jax.Array, _total_vp_b(self._state))

    def last(
        self,
    ) -> tuple[Observation, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        """``(obs, reward, termination, truncation, info)`` for the acting agents.

        Each quantity is gathered per lane for that lane's ``agent_selection``.
        """
        sel = self.agent_selection
        rows = jnp.arange(self.batch_size)
        obs = self._obs_for(sel)
        reward = self._reward[rows, sel]
        termination = self._terminations[rows, sel]
        truncation = self._truncations[rows, sel]
        return obs, reward, termination, truncation, self.infos

    def close(self) -> None:
        """No resources to release."""

    def render(self, lane: int = 0) -> str:
        """One-line textual status for ``lane`` (full rendering lives in tests)."""
        s = self._state
        phase = GamePhase(int(s.phase[lane]))
        return (
            f"[lane {lane}] {phase}  player={int(s.current_player[lane]) + 1}  "
            f"dice={int(s.dice_roll[lane])}  vp={[int(x) for x in self._vps[lane]]}"
        )

    # -- AEC attributes ---------------------------------------------------

    @property
    def agent_selection(self) -> jax.Array:
        """``(B,)`` int array of the acting player per lane (batched AEC)."""
        return cast(jax.Array, _agent_selection_b(self._state))

    @property
    def rewards(self) -> jax.Array:
        """``(B, N_PLAYERS)`` reward from the last :meth:`step`."""
        return self._reward

    @property
    def terminations(self) -> jax.Array:
        """``(B, N_PLAYERS)`` per-lane game-over flags (broadcast across players)."""
        return self._terminations

    @property
    def truncations(self) -> jax.Array:
        """``(B, N_PLAYERS)`` truncation flags (always False -- no time limit)."""
        return self._truncations

    @property
    def infos(self) -> dict[str, jax.Array]:
        """Batched info: action mask, acting player, current player, last result."""
        return {
            "action_mask": self.action_mask(),
            "agent_selection": self.agent_selection,
            "current_player": self._state.current_player.astype(jnp.int32),
            "result": self._result,
        }

    @property
    def board(self) -> Board:
        """The underlying batched ``(BoardLayout, BoardState)``."""
        return self._layout, self._state

    # -- Spaces -----------------------------------------------------------

    def action_space(self, agent: object = None) -> dict[str, Space]:
        """Descriptor of the ``(action_type, ActionParams)`` action (per lane)."""
        return {
            "action_type": Discrete(N_ACTION_TYPES),
            "idx": Discrete(max(N_VERTICES, N_EDGES)),  # vertex/edge/tile/resource
            "target": Discrete(N_PLAYERS),  # victim / receive / 2nd resource
            "resources": Box((N_RESOURCES,), "int32", 0, BANK_INITIAL),  # Discard
        }

    def observation_space(self, agent: object = None) -> dict[str, Space]:
        """Descriptor of one lane's observation (see :meth:`observe`)."""
        return {
            "tile_resource": Box((N_TILES,), "uint8", 0, 5),
            "tile_number": Box((N_TILES,), "uint8", 0, 12),
            "port_allocation": Box((N_PORTS,), "uint8", 0, 5),
            "vertex_owner": Box((N_VERTICES,), "uint8", 0, N_PLAYERS),
            "vertex_type": Box((N_VERTICES,), "uint8", 0, 2),
            "edge_road": Box((N_EDGES,), "uint8", 0, N_PLAYERS),
            "robber": Discrete(N_TILES),
            "victory_points": Box((N_PLAYERS,), "uint8", 0, VICTORY_POINTS_TO_WIN),
            "knights_played": Box((N_PLAYERS,), "uint8", 0, 14),
            "hand_size": Box((N_PLAYERS,), "int32", 0, 255),
            "dev_card_count": Box((N_PLAYERS,), "int32", 0, 25),
            "longest_road_owner": Discrete(N_PLAYERS + 1),
            "largest_army_owner": Discrete(N_PLAYERS + 1),
            "longest_road_len": Discrete(N_EDGES + 1),
            "bank": Box((N_RESOURCES,), "uint8", 0, BANK_INITIAL),
            "phase": Discrete(len(GamePhase)),
            "current_player": Discrete(N_PLAYERS),
            "dice_roll": Discrete(13),
            "has_rolled": Discrete(2),
            "self": Discrete(N_PLAYERS),
            "self_resources": Box((N_RESOURCES,), "uint8", 0, BANK_INITIAL),
            "self_dev_hand": Box((N_DEV_CARD_TYPES,), "uint8", 0, 25),
            "self_pending_discard": Discrete(256),
        }

    # -- Observations -----------------------------------------------------

    def observe(self, agent: int | str) -> Observation:
        """Partial observation from ``agent``'s point of view, across all lanes.

        ``agent`` may be a player index (0..N_PLAYERS-1) or ``"player_i"``. The
        observer sees its own hand / dev cards in full but only public counts for
        opponents.
        """
        me = self._agent_index(agent)
        sel = jnp.full((self.batch_size,), me, dtype=jnp.int32)
        return self._obs_for(sel)

    def action_mask(self) -> jax.Array:
        """``(B, N_ACTION_TYPES)`` -- which action types the acting player can use."""
        return cast(jax.Array, _action_type_mask_b(self._layout, self._state))

    def available_indices(self, action_type: int | ActionType) -> jax.Array:
        """``(B, D)`` legality over the primary index of an index-parameterized action.

        Supported: SETUP_SETTLEMENT / BUILD_SETTLEMENT / BUILD_CITY (vertices),
        SETUP_ROAD / BUILD_ROAD (edges), PLAY_MONOPOLY (resources), and
        MOVE_ROBBER / PLAY_KNIGHT (tiles, legal if some victim choice works).
        Other action types have no single index domain -- use :meth:`action_mask`
        or :func:`available`.
        """
        at = ActionType(int(action_type))
        fn = _INDEX_MASKS.get(at)
        if fn is None:
            raise ValueError(
                f"{at.name} has no single primary-index domain; "
                "use action_mask() or available()"
            )
        return fn(self._layout, self._state)

    # -- Internals --------------------------------------------------------

    def _agent_index(self, agent: int | str) -> int:
        if isinstance(agent, str):
            return self.possible_agents.index(agent)
        if not 0 <= agent < N_PLAYERS:
            raise ValueError(f"agent index {agent} out of range [0, {N_PLAYERS})")
        return agent

    def _obs_for(self, sel: jax.Array) -> Observation:
        """Observation with per-lane ``self`` index ``sel`` (``(B,)`` int array)."""
        layout, state = self._layout, self._state
        res = state.player_resources
        self_res = jnp.take_along_axis(res, sel[:, None, None], axis=1)[:, 0, :]
        self_dev = jnp.take_along_axis(state.dev_hand, sel[:, None, None], axis=1)[
            :, 0, :
        ]
        self_pending = jnp.take_along_axis(state.pending_discard, sel[:, None], axis=1)[
            :, 0
        ]
        return {
            # public board
            "tile_resource": layout.tile_resource,
            "tile_number": layout.tile_number,
            "port_allocation": layout.port_allocation,
            "vertex_owner": state.vertex_owner,
            "vertex_type": state.vertex_type,
            "edge_road": state.edge_road,
            "robber": state.robber,
            # public player info
            "victory_points": state.victory_points,
            "knights_played": state.knights_played,
            "hand_size": res.astype(jnp.int32).sum(axis=2),
            "dev_card_count": state.dev_hand.astype(jnp.int32).sum(axis=2),
            "longest_road_owner": state.longest_road_owner,
            "largest_army_owner": state.largest_army_owner,
            "longest_road_len": state.longest_road_len,
            "bank": compute_bank_resources(res),
            # turn / flow
            "phase": state.phase,
            "current_player": state.current_player,
            "dice_roll": state.dice_roll,
            "has_rolled": state.has_rolled,
            # private (observer)
            "self": sel,
            "self_resources": self_res,
            "self_dev_hand": self_dev,
            "self_pending_discard": self_pending,
        }

    def _compute_reward(
        self, vps_before: jax.Array, vps_after: jax.Array, done_lane: jax.Array
    ) -> jax.Array:
        if self.reward_mode == "vp_delta":
            return (vps_after - vps_before).astype(jnp.float32)
        # `done_lane` is the row-any of `winners`, so `winners & done_lane[:, None]`
        # equals `winners` (no row can be a winner unless its lane is done).
        winners = vps_after >= VICTORY_POINTS_TO_WIN
        return winners.astype(jnp.float32)

    def _auto_reset(
        self, layout: BoardLayout, state: BoardState, done_lane: jax.Array
    ) -> tuple[BoardLayout, BoardState]:
        if not self.auto_reset or not bool(jnp.any(done_lane)):
            return layout, state
        self._key, k_layout, k_state = jax.random.split(self._key, 3)
        fresh_layout = make_layout(self.batch_size, key=k_layout)
        fresh_state = make_board_state(self.batch_size, key=k_state)
        fresh_state = fresh_state._replace(
            robber=desert_tile(fresh_layout.tile_resource)
        )
        new_layout = BoardLayout(
            *(
                _where_lane(done_lane, f, c)
                for f, c in zip(fresh_layout, layout, strict=True)
            )
        )
        fields = {}
        for name in BoardState._fields:
            f, c = getattr(fresh_state, name), getattr(state, name)
            fields[name] = (
                _select_key(done_lane, f, c)
                if name == "key"
                else _where_lane(done_lane, f, c)
            )
        return new_layout, BoardState(**fields)


def _where_lane(mask: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """Per-lane ``where``: pick ``a`` where ``mask`` (``(B,)``) is set, else ``b``."""
    m = mask.reshape((mask.shape[0],) + (1,) * (a.ndim - 1))
    return jnp.where(m, a, b)


def _select_key(mask: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """``_where_lane`` for typed PRNG key arrays (select on the raw key data)."""
    ad, bd = jax.random.key_data(a), jax.random.key_data(b)
    m = mask.reshape((mask.shape[0],) + (1,) * (ad.ndim - 1))
    return cast(jax.Array, jax.random.wrap_key_data(jnp.where(m, ad, bd)))
