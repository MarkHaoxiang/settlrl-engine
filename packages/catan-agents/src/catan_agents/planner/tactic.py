"""One-step tactical lookahead for the planner's leaf decisions.

The planner's strategic layer (plans, saving, award races, trade memory)
stays scripted; the *tactical* picks — which setup spot, which robber cell,
which discard, accept or reject — consult a real lookahead: rebuild a
single-game engine board from the observation, apply every flat action with
the env's own legality mask, and score the successors with the shipped
heuristic from the player's seat.

Public fields reconstruct exactly. Hidden ones get a neutral fill — each
opponent's hand spread evenly over the resource types, their dev cards held
as knights (only the count enters the opponent strength term), the dev deck
scaled to the unseen remainder — and ``free_roads`` is inferred (a legal
BUILD_ROAD the hand cannot pay for means at least one is owed). The mask is
passed straight through as the availability ``apply_action`` trusts, so
reconstruction gaps cannot make an illegal action look applied.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from catan_engine.board.dev_cards import DEV_CARD_COUNTS, N_DEV_CARD_TYPES, DevCard
from catan_engine.board.layout import BoardLayout
from catan_engine.board.resources import N_RESOURCES, ROAD_COST
from catan_engine.board.state import BoardState, GamePhase, KeyScalar
from catan_engine.env import N_FLAT, ActionType, flat_to_action
from catan_engine.mechanics.action import ActionParams, apply_action
from catan_engine.mechanics.flat import flat_available_for

from catan_agents.planner.pov import Pov
from catan_agents.shared.policy import FlatMask
from catan_agents.shared.value import heuristic_value

_ROW_TYPE, _ROW_PARAMS = flat_to_action(jnp.arange(N_FLAT))
_DEV_TOTAL = int(sum(DEV_CARD_COUNTS))
_ROAD_COST = np.asarray(ROAD_COST, np.int64)


def _u8(x: object) -> jax.Array:
    return jnp.asarray(x, jnp.uint8)


def _spread(total: int) -> np.ndarray:
    """``total`` cards spread as evenly as possible over the resource types."""
    base, rem = divmod(total, N_RESOURCES)
    return base + (np.arange(N_RESOURCES) < rem).astype(np.int64)


def reconstruct(pov: Pov, key: KeyScalar) -> tuple[BoardLayout, BoardState]:
    """A single-game engine board consistent with the observation."""
    n = pov.n_players
    res = np.zeros((n, N_RESOURCES), np.int64)
    dev = np.zeros((n, N_DEV_CARD_TYPES), np.int64)
    pending = np.zeros((n,), np.int64)
    for p in range(n):
        if p == pov.me:
            res[p] = pov.hand
            dev[p] = pov.dev_hand
            pending[p] = pov.pending_discard
        else:
            res[p] = _spread(int(pov.hand_size[p]))
            dev[p, DevCard.KNIGHT] = int(pov.dev_card_count[p])
    drawn = int(pov.dev_card_count.sum() + pov.knights_played.sum())
    deck = np.asarray(DEV_CARD_COUNTS) * max(_DEV_TOTAL - drawn, 0) // _DEV_TOTAL
    road_legal = pov.legal_rows(ActionType.BUILD_ROAD).size > 0
    free_roads = int(road_legal and bool(np.any(pov.hand < _ROAD_COST)))
    setup_index = int((pov.vertex_owner > 0).sum()) - (
        pov.phase == GamePhase.SETUP_ROAD
    )

    layout = BoardLayout(
        tile_resource=_u8(pov.tile_resource),
        tile_number=_u8(pov.tile_number),
        port_allocation=_u8(pov.port_allocation),
    )
    state = BoardState(
        vertex_owner=_u8(pov.vertex_owner),
        vertex_type=_u8(pov.vertex_type),
        edge_road=_u8(pov.edge_road),
        robber=_u8(pov.robber),
        player_resources=_u8(res),
        victory_points=_u8(pov.victory_points),
        dev_deck=_u8(deck),
        dev_hand=_u8(dev),
        knights_played=_u8(pov.knights_played),
        phase=_u8(pov.phase),
        current_player=_u8(pov.current_player),
        setup_index=_u8(setup_index),
        dice_roll=_u8(pov.dice_roll),
        has_rolled=_u8(pov.has_rolled),
        dev_played=_u8(0),
        dev_bought=_u8(np.zeros(N_DEV_CARD_TYPES, np.int64)),
        free_roads=_u8(free_roads),
        pending_discard=_u8(pending),
        trade_partner=_u8(pov.trade_partner),
        trade_give=_u8(pov.trade_give),
        trade_receive=_u8(pov.trade_receive),
        longest_road_owner=_u8(pov.longest_road_owner),
        largest_army_owner=_u8(pov.largest_army_owner),
        longest_road_len=_u8(pov.longest_road_len),
        key=key,
    )
    return layout, state


@jax.jit
def _successor_values(
    layout: BoardLayout, state: BoardState, player: jax.Array, mask: FlatMask
) -> jax.Array:
    succ, _ = jax.vmap(apply_action, in_axes=(None, None, 0, 0, 0))(
        layout, state, _ROW_TYPE, _ROW_PARAMS, mask
    )
    return jax.vmap(heuristic_value, in_axes=(None, 0, None))(layout, succ, player)


@jax.jit
def _after(
    layout: BoardLayout, state: BoardState, row: jax.Array, player: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Apply one (legal) row, then sweep legality and successor values of the
    follow-up position: the second ply of an own-turn combo."""
    params = ActionParams(idx=_ROW_PARAMS.idx[row], target=_ROW_PARAMS.target[row])
    succ, _ = apply_action(layout, state, _ROW_TYPE[row], params, jnp.bool_(True))
    mask2 = flat_available_for(layout, succ)
    return mask2, _successor_values(layout, succ, player, mask2)


# Rows an imagined opponent may not use in a reply: their reconstructed dev
# hand is fiction, so their dev plays would be too.
_NO_DEV_PLAYS = jnp.asarray(
    ~np.isin(
        np.asarray(_ROW_TYPE),
        [
            int(ActionType.PLAY_KNIGHT),
            int(ActionType.PLAY_ROAD_BUILDING),
            int(ActionType.PLAY_YEAR_OF_PLENTY),
            int(ActionType.PLAY_MONOPOLY),
        ],
    )
)


@jax.jit
def _best_reply(
    layout: BoardLayout, state: BoardState, my_row: jax.Array, opp: jax.Array
) -> jax.Array:
    """The opponent's best answer to ``my_row``, valued from *their* seat:
    apply my action, hand them a fabricated MAIN turn on the result, and take
    the max over their grounded options (builds and bank trades — public
    roads, real hand size, spread composition)."""
    params = ActionParams(
        idx=_ROW_PARAMS.idx[my_row], target=_ROW_PARAMS.target[my_row]
    )
    succ, _ = apply_action(layout, state, _ROW_TYPE[my_row], params, jnp.bool_(True))
    theirs = succ._replace(
        current_player=opp.astype(jnp.uint8),
        phase=jnp.uint8(GamePhase.MAIN),
        has_rolled=jnp.uint8(1),
        dev_played=jnp.uint8(0),
        free_roads=jnp.uint8(0),
    )
    mask = flat_available_for(layout, theirs) & _NO_DEV_PLAYS
    vals = _successor_values(layout, theirs, opp.astype(jnp.int32), mask)
    return jnp.max(jnp.where(mask, vals, -jnp.inf))


class Tactic:
    """Per-decision successor values, computed lazily and cached per ``Pov``."""

    def __init__(self, seed: int) -> None:
        self._key = jax.random.key(seed)
        self._cache: np.ndarray | None = None
        self._board: tuple[BoardLayout, BoardState] | None = None
        self._cache_for: Pov | None = None

    def values(self, pov: Pov) -> np.ndarray:
        """``(N_FLAT,)`` heuristic value of each action's successor (one
        sampled outcome for the stochastic ones); junk on illegal rows."""
        if self._cache_for is not pov:
            self._key, sub = jax.random.split(self._key)
            self._board = reconstruct(pov, sub)
            vals = _successor_values(
                *self._board, jnp.int32(pov.me), jnp.asarray(pov.mask)
            )
            self._cache = np.asarray(jax.device_get(vals))
            self._cache_for = pov
        assert self._cache is not None
        return self._cache

    def best(self, pov: Pov, rows: np.ndarray) -> int:
        """The row among ``rows`` whose successor scores best."""
        vals = self.values(pov)
        return int(rows[int(np.argmax(vals[rows]))])

    def best_paranoid(self, pov: Pov, rows: np.ndarray, margin: float = 1.0) -> int:
        """``best``, with near-ties (within ``margin`` of the top) broken by
        the opponents' best reply: among moves equally good for us, prefer
        the one that leaves them the worst answer."""
        vals = self.values(pov)
        top = float(np.max(vals[rows]))
        short = [int(r) for r in rows if vals[r] >= top - margin]
        if len(short) == 1:
            return short[0]
        assert self._board is not None
        layout, state = self._board
        opponents = [p for p in range(pov.n_players) if p != pov.me]

        def reply(row: int) -> float:
            return max(
                float(_best_reply(layout, state, jnp.int32(row), jnp.int32(p)))
                for p in opponents
            )

        return min(short, key=reply)

    def combo_best(
        self, pov: Pov, enablers: list[int], follow: np.ndarray
    ) -> tuple[int, int, float] | None:
        """The best own-turn pair (enabler row, follow-up row) by the pair's
        final value: the second ply lookahead alone cannot see — a bank trade
        or dev play that *enables* a build this same turn. ``follow`` flags
        the rows allowed as the second action. None if no pair is playable."""
        self.values(pov)  # ensure the board cache
        assert self._board is not None
        layout, state = self._board
        player = jnp.int32(pov.me)
        out: tuple[int, int, float] | None = None
        for e in enablers:
            mask2, vals2 = _after(layout, state, jnp.int32(e), player)
            allowed = np.asarray(mask2) & follow
            if not allowed.any():
                continue
            v2 = np.asarray(vals2)
            row2 = int(np.flatnonzero(allowed)[np.argmax(v2[allowed])])
            if out is None or v2[row2] > out[2]:
                out = (e, row2, float(v2[row2]))
        return out
