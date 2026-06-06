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
- ``rewards`` / ``terminations`` / ``truncations`` are ``(B, n_players)`` arrays
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
import functools
from typing import Literal, cast

import jax
import jax.numpy as jnp

from catan_engine.mechanics.action import (
    ActionParams,
    ActionResult,
    ActionType,
    N_ACTION_TYPES,
    _ATYPE_J,
    _IDX_J,
    _INDEX_MASKS,
    _N_FLAT,
    _TARGET_J,
    _flat_available_b,
    _flat_available_for,
    action_available,
    apply_action,
    flat_legality,
)
from catan_engine.mechanics.common import (
    ActionTypeArray,
    Mask,
    ResultCode,
    agent_selection_single,
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
_step = jax.jit(jax.vmap(apply_action, in_axes=(0, 0, 0, 0, 0)))
_available = jax.jit(jax.vmap(action_available, in_axes=(0, 0, 0, 0)))


def step(
    board: Board, action_type: ActionTypeArray, params: ActionParams
) -> tuple[BoardState, ResultCode]:
    """Apply one (batched) action per game; return (new state, ActionResult codes).

    Computes the per-action legality once (via the switch-based
    :func:`action_available`, exact for any params) and hands it to ``apply_action``
    -- no branch recomputes avail. ``BatchedCatanEnv`` instead reads legality from
    its cached flat-legality sweep; this functional entry stays self-validating for
    callers (tests / the reference oracle) that pass arbitrary actions.
    """
    available = _available(board[0], board[1], action_type, params)
    new_state, result = _step(board[0], board[1], action_type, params, available)
    return new_state, result


def available(board: Board, action_type: ActionTypeArray, params: ActionParams) -> Mask:
    """``(batch,)`` legality mask for the chosen action per game (no state change)."""
    return cast(Mask, _available(board[0], board[1], action_type, params))


# ---------------------------------------------------------------------------
# Batched derived quantities used by the environment.
# ---------------------------------------------------------------------------
Observation = dict[str, jax.Array]


def _total_vp_single(state: BoardState) -> jax.Array:
    """Total VP (buildings + awards + VP cards) for every player in one game."""
    players = jnp.arange(state.n_players, dtype=jnp.int32)
    return jax.vmap(lambda p: player_total_vp(state, p))(players)


_total_vp_b = jax.jit(jax.vmap(_total_vp_single))


_agent_selection_b = jax.jit(jax.vmap(agent_selection_single))


def _random_action_single(
    avail_flat: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample a uniformly-random *legal* flat action for one game.

    The single-game core behind :meth:`BatchedCatanEnv.random_actions`: takes the
    lane's cached flat legality mask ``avail_flat`` (``(N_FLAT,)``) and the argmax
    of uniform noise over its legal entries. Returns the chosen action's
    ``(action_type, idx, target)`` straight from the static flat table. The mask
    is read from the env's cache (computed once per step), not recomputed here.
    """
    # Random legal action: score legal moves with uniform noise and take the
    # argmax (illegal scored -1, so only picked if the lane has no legal move).
    noise = jax.random.uniform(key, (_N_FLAT,))
    chosen = jnp.argmax(jnp.where(avail_flat, noise, -1.0))
    return _ATYPE_J[chosen], _IDX_J[chosen], _TARGET_J[chosen]


@jax.jit
def _random_actions_b(
    avail_flat: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample a random legal action per lane: ``(action_type, idx, target)``.

    Splits ``key`` per lane *inside* the trace (so the split fuses into the XLA
    graph rather than dispatching eagerly every step) and maps
    :func:`_random_action_single` over the batch. ``avail_flat`` is the cached
    ``(B, N_FLAT)`` flat-legality sweep -- no avail is recomputed here.
    """
    keys = jax.random.split(key, avail_flat.shape[0])
    return jax.vmap(_random_action_single, in_axes=(0, 0))(avail_flat, keys)


@jax.jit
def _type_mask_from_flat(avail_flat: jax.Array) -> jax.Array:
    """``(B, N_ACTION_TYPES)`` per-action-type legality, reduced from the cache.

    OR the cached ``(B, N_FLAT)`` flat mask over each action type's rows (its
    concrete moves): an action type is legal iff some concrete move of that type
    is. This is the single legality source -- no separate per-type avail sweep.
    """
    b = avail_flat.shape[0]
    return jnp.zeros((b, N_ACTION_TYPES), jnp.bool_).at[:, _ATYPE_J].max(avail_flat)


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
class BatchedCatanEnv:
    """A batch of Catan games behind a (batched) PettingZoo-AEC interface.

    Args:
        batch_size: number of games run in parallel (the leading array axis).
        seed: PRNG seed for the initial boards and auto-reset randomness.
        n_players: players seated per game (2..4, default 4; same for every
            lane). Sizes the per-player axis of the state, observations, and
            ``rewards`` / ``terminations`` / ``truncations``.
        reward: ``"sparse"`` (+1 to the winner on the terminal step, 0 else) or
            ``"vp_delta"`` (each player's change in total VP this step).
        auto_reset: when True (default), a lane that terminates is replaced with a
            fresh game on the same step; when False, terminated lanes freeze (and
            keep reporting ``terminations`` True) for callers that manage episode
            boundaries themselves (e.g. the single-game AEC wrapper).
        number_placement: ``"random"`` (default) shuffles the number tokens
            uniformly; ``"spiral"`` lays them in the rulebook's variable set-up
            spiral (tournament-style balanced boards). Applied to the initial
            boards and to auto-reset replacements.

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
        number_placement: Literal["random", "spiral"] = "random",
        n_players: int = N_PLAYERS,
    ) -> None:
        if reward not in ("sparse", "vp_delta"):
            raise ValueError(f"reward must be 'sparse' or 'vp_delta', got {reward!r}")
        if number_placement not in ("random", "spiral"):
            raise ValueError(
                "number_placement must be 'random' or 'spiral', "
                f"got {number_placement!r}"
            )
        if not 2 <= n_players <= N_PLAYERS:
            raise ValueError(f"n_players must be in [2, {N_PLAYERS}], got {n_players}")
        self.batch_size = batch_size
        self.reward_mode = reward
        self.auto_reset = auto_reset
        self.number_placement = number_placement
        self.n_players = n_players
        self.possible_agents = [f"player_{i}" for i in range(n_players)]
        self.agents = list(self.possible_agents)
        self.num_agents = n_players
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
        self._layout = make_layout(
            self.batch_size, key=k_layout, number_placement=self.number_placement
        )
        self._state = make_board_state(
            self.batch_size, key=k_state, n_players=self.n_players
        )
        self._state = self._state._replace(
            robber=desert_tile(self._layout.tile_resource)
        )
        B, P = self.batch_size, self.n_players
        self._reward = jnp.zeros((B, P), dtype=jnp.float32)
        self._terminations = jnp.zeros((B, P), dtype=jnp.bool_)
        self._truncations = jnp.zeros((B, P), dtype=jnp.bool_)
        self._result = jnp.full((B,), int(ActionResult.SUCCESS), dtype=jnp.int32)
        self._vps = cast(jax.Array, _total_vp_b(self._state))
        # Flat legality of every action for each lane's acting player, computed
        # once and reused by step (to gate the chosen action), random_actions, and
        # action_mask -- the single legality source. Refreshed after every step.
        self._avail = cast(jax.Array, _flat_available_b(self._layout, self._state))
        self.agents = list(self.possible_agents)

    def step(self, action_type: ActionTypeArray, params: ActionParams) -> None:
        """Apply one action per lane to its acting player (AEC ``step``).

        ``action_type`` is a ``(B,)`` int array of :class:`ActionType` codes and
        ``params`` an :class:`ActionParams` with batched leaves. Terminated lanes
        auto-reset; the resulting reward / termination reflect the transition
        that just happened, while the next observation is the reset game's.

        DISCARD is one card per step: ``idx`` is the resource the acting
        discarder gives up one card of; the action repeats until every owed
        count reaches zero, then the phase advances to MOVE_ROBBER.
        """
        at = jnp.asarray(action_type, dtype=jnp.int32)
        # The whole step -- gate the chosen action with the cached legality, apply,
        # score reward / termination, auto-reset finished lanes, and refresh the vps
        # and legality cache for the next step -- is one fused jit dispatch (small
        # batches are dispatch-bound, so collapsing ~5 kernels into 1 is the win).
        (
            self._layout,
            self._state,
            self._reward,
            self._terminations,
            self._truncations,
            self._result,
            self._vps,
            self._avail,
            self._key,
        ) = _env_step_core(
            self._layout,
            self._state,
            at,
            params,
            self._avail,
            self._vps,
            self._key,
            self.batch_size,
            self.reward_mode,
            self.auto_reset,
            self.number_placement,
            self.n_players,
        )

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
        """``(B, n_players)`` reward from the last :meth:`step`."""
        return self._reward

    @property
    def terminations(self) -> jax.Array:
        """``(B, n_players)`` per-lane game-over flags (broadcast across players)."""
        return self._terminations

    @property
    def truncations(self) -> jax.Array:
        """``(B, n_players)`` truncation flags (always False -- no time limit)."""
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
        }

    def observation_space(self, agent: object = None) -> dict[str, Space]:
        """Descriptor of one lane's observation (see :meth:`observe`)."""
        return {
            "tile_resource": Box((N_TILES,), "uint8", 0, 5),
            "tile_number": Box((N_TILES,), "uint8", 0, 12),
            "port_allocation": Box((N_PORTS,), "uint8", 0, 5),
            "vertex_owner": Box((N_VERTICES,), "uint8", 0, self.n_players),
            "vertex_type": Box((N_VERTICES,), "uint8", 0, 2),
            "edge_road": Box((N_EDGES,), "uint8", 0, self.n_players),
            "robber": Discrete(N_TILES),
            "victory_points": Box(
                (self.n_players,), "uint8", 0, VICTORY_POINTS_TO_WIN
            ),
            "knights_played": Box((self.n_players,), "uint8", 0, 14),
            "hand_size": Box((self.n_players,), "int32", 0, 255),
            "dev_card_count": Box((self.n_players,), "int32", 0, 25),
            "longest_road_owner": Discrete(self.n_players + 1),
            "largest_army_owner": Discrete(self.n_players + 1),
            "longest_road_len": Discrete(N_EDGES + 1),
            "bank": Box((N_RESOURCES,), "uint8", 0, BANK_INITIAL),
            "phase": Discrete(len(GamePhase)),
            "current_player": Discrete(self.n_players),
            "dice_roll": Discrete(13),
            "has_rolled": Discrete(2),
            "self": Discrete(self.n_players),
            "self_resources": Box((N_RESOURCES,), "uint8", 0, BANK_INITIAL),
            "self_dev_hand": Box((N_DEV_CARD_TYPES,), "uint8", 0, 25),
            "self_pending_discard": Discrete(256),
        }

    # -- Observations -----------------------------------------------------

    def observe(self, agent: int | str) -> Observation:
        """Partial observation from ``agent``'s point of view, across all lanes.

        ``agent`` may be a player index (0..n_players-1) or ``"player_i"``. The
        observer sees its own hand / dev cards in full but only public counts for
        opponents.
        """
        me = self._agent_index(agent)
        sel = jnp.full((self.batch_size,), me, dtype=jnp.int32)
        return self._obs_for(sel)

    def action_mask(self) -> jax.Array:
        """``(B, N_ACTION_TYPES)`` -- which action types the acting player can use.

        Reduced from the cached flat-legality sweep (:attr:`_avail`) -- no separate
        per-type avail computation.
        """
        return cast(jax.Array, _type_mask_from_flat(self._avail))

    def available_indices(self, action_type: int | ActionType) -> jax.Array:
        """``(B, D)`` legality over the primary index of an index-parameterized action.

        Supported: SETUP_SETTLEMENT / BUILD_SETTLEMENT / BUILD_CITY (vertices),
        SETUP_ROAD / BUILD_ROAD (edges), DISCARD / PLAY_MONOPOLY (resources),
        and MOVE_ROBBER / PLAY_KNIGHT (tiles, legal if some victim choice works).
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

    def random_actions(
        self, key: jax.Array
    ) -> tuple[ActionTypeArray, ActionParams]:
        """A uniformly-random *legal* action per lane (the random-rollout driver).

        Reads the cached flat-legality sweep (:attr:`_avail`, computed once per
        step) and samples one legal action per lane via a per-lane masked argmax --
        no avail is recomputed here. A lane with no legal action yields an INVALID
        action and simply stalls until its next auto-reset. ``key`` is a JAX PRNG
        key, split per lane for the choice.
        """
        atype, idx, target = _random_actions_b(self._avail, key)
        return atype, ActionParams(idx=idx, target=target)

    # -- Internals --------------------------------------------------------

    def _agent_index(self, agent: int | str) -> int:
        if isinstance(agent, str):
            return self.possible_agents.index(agent)
        if not 0 <= agent < self.n_players:
            raise ValueError(f"agent index {agent} out of range [0, {self.n_players})")
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


def _where_lane(mask: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """Per-lane ``where``: pick ``a`` where ``mask`` (``(B,)``) is set, else ``b``."""
    m = mask.reshape((mask.shape[0],) + (1,) * (a.ndim - 1))
    return jnp.where(m, a, b)


def _select_key(mask: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """``_where_lane`` for typed PRNG key arrays (select on the raw key data)."""
    ad, bd = jax.random.key_data(a), jax.random.key_data(b)
    m = mask.reshape((mask.shape[0],) + (1,) * (ad.ndim - 1))
    return cast(jax.Array, jax.random.wrap_key_data(jnp.where(m, ad, bd)))


@functools.partial(
    jax.jit, static_argnames=("batch_size", "number_placement", "n_players")
)
def _auto_reset_core(
    layout: BoardLayout,
    state: BoardState,
    done_lane: jax.Array,
    key: jax.Array,
    batch_size: int,
    number_placement: Literal["random", "spiral"],
    n_players: int,
) -> tuple[BoardLayout, BoardState]:
    """Replace finished lanes (``done_lane``) with fresh games, fully on device."""

    def do_reset(
        operand: tuple[BoardLayout, BoardState],
    ) -> tuple[BoardLayout, BoardState]:
        layout, state = operand
        k_layout, k_state = jax.random.split(key)
        fresh_layout = make_layout(
            batch_size, key=k_layout, number_placement=number_placement
        )
        fresh_state = make_board_state(batch_size, key=k_state, n_players=n_players)
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

    def no_reset(
        operand: tuple[BoardLayout, BoardState],
    ) -> tuple[BoardLayout, BoardState]:
        return operand

    return cast(
        "tuple[BoardLayout, BoardState]",
        jax.lax.cond(jnp.any(done_lane), do_reset, no_reset, (layout, state)),
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "batch_size",
        "reward_mode",
        "auto_reset",
        "number_placement",
        "n_players",
    ),
)
def _env_step_core(
    layout: BoardLayout,
    state: BoardState,
    action_type: jax.Array,
    params: ActionParams,
    avail_flat: jax.Array,
    vps_before: jax.Array,
    key: jax.Array,
    batch_size: int,
    reward_mode: str,
    auto_reset: bool,
    number_placement: Literal["random", "spiral"],
    n_players: int,
) -> tuple[
    BoardLayout,
    BoardState,
    jax.Array,
    jax.Array,
    jax.Array,
    ResultCode,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """One whole ``BatchedCatanEnv.step`` as a single fused ``jit`` dispatch.

    Gates the chosen action with the cached flat legality, applies the avail-free
    core, scores reward / termination, auto-resets finished lanes, and refreshes the
    per-player VPs and the legality cache for the next step. Returns the new
    ``(layout, state, reward, terminations, truncations, result, vps, avail, key)``.
    Collapsing what were ~5 separate kernels (gate+apply, two VP sweeps, auto-reset,
    legality refresh) into one removes the per-step dispatch overhead that dominates
    small-batch throughput; the cache refresh rides along for free.

    ``reward_mode`` (``"sparse"`` / ``"vp_delta"``), ``auto_reset``, and
    ``number_placement`` / ``n_players`` (forwarded to the auto-reset board
    construction) are static,
    so their Python branches are resolved at trace time. ``key`` is threaded (split
    for auto-reset, returned unchanged when ``auto_reset`` is False).
    """
    # Gate: read the chosen action's legality from the cache (no avail recompute).
    legal = flat_legality(avail_flat, action_type, params.idx, params.target)
    applied, result = jax.vmap(apply_action, in_axes=(0, 0, 0, 0, 0))(
        layout, state, action_type, params, legal
    )

    vps_after = jax.vmap(_total_vp_single)(applied)
    done_lane = jnp.any(vps_after >= VICTORY_POINTS_TO_WIN, axis=1)  # (B,)

    if reward_mode == "vp_delta":
        reward = (vps_after - vps_before).astype(jnp.float32)
    else:  # "sparse": +1 to each winner on the terminal step (only a done lane wins).
        reward = (vps_after >= VICTORY_POINTS_TO_WIN).astype(jnp.float32)
    terminations = jnp.broadcast_to(done_lane[:, None], (batch_size, n_players))
    truncations = jnp.zeros((batch_size, n_players), dtype=jnp.bool_)

    if auto_reset:
        key, subkey = jax.random.split(key)
        new_layout, new_state = _auto_reset_core(
            layout, applied, done_lane, subkey, batch_size, number_placement, n_players
        )
    else:
        new_layout, new_state = layout, applied

    new_vps = jax.vmap(_total_vp_single)(new_state)
    new_avail = jax.vmap(_flat_available_for)(new_layout, new_state)
    return (
        new_layout,
        new_state,
        reward,
        terminations,
        truncations,
        result,
        new_vps,
        new_avail,
        key,
    )
