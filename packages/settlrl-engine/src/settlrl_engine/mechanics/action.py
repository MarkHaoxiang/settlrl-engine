"""Unified action dispatch: one ``(action_type, params)`` interface.

``apply_action`` / ``action_available`` dispatch over the per-action cores
with ``jax.lax.switch`` so the whole thing stays traceable and vmappable; the
heterogeneous per-action params are packed into one ``ActionParams`` and each
branch unpacks what it needs. The flat enumeration of the same action space
lives in ``flat.py``.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import VICTORY_POINTS_TO_WIN, BoardState
from settlrl_engine.mechanics.awards import (
    resolve_step,
    road_build_needed,
    settlement_break_needed,
)
from settlrl_engine.mechanics.common import (
    ActionResult,
    ActionTypeArray,
    IndexParam,
    Mask,
    ResultCode,
    player_total_vp,
)
from settlrl_engine.mechanics.development import (
    _buy_dev_apply,
    _buy_dev_avail,
    _knight_apply,
    _knight_avail,
    _monopoly_apply,
    _monopoly_avail,
    _road_building_apply,
    _road_building_avail,
    _yop_apply,
    _yop_avail,
)
from settlrl_engine.mechanics.dice import _roll_apply, _roll_avail
from settlrl_engine.mechanics.placement import (
    _build_city_apply,
    _build_city_avail,
    _build_road_apply,
    _build_road_avail,
    _build_settlement_apply,
    _build_settlement_avail,
)
from settlrl_engine.mechanics.robber import (
    _discard_apply,
    _discard_avail,
    _move_robber_apply,
    _move_robber_avail,
)
from settlrl_engine.mechanics.setup import (
    _setup_road_apply,
    _setup_road_avail,
    _setup_settlement_apply,
    _setup_settlement_avail,
)
from settlrl_engine.mechanics.trade import (
    _accept_trade_apply,
    _accept_trade_avail,
    _maritime_apply,
    _maritime_avail,
    _propose_trade_apply,
    _propose_trade_avail,
    _reject_trade_apply,
    _reject_trade_avail,
)
from settlrl_engine.mechanics.turn import _end_turn_apply, _end_turn_avail

__all__ = [
    "N_ACTION_TYPES",
    "ActionParams",
    "ActionResult",
    "ActionType",
    "ActionTypeArray",
    "Mask",
    "ResultCode",
    "action_available",
    "apply_action",
    "player_total_vp",
]


# ===========================================================================
# Unified action dispatch
# ===========================================================================


class ActionType(IntEnum):
    """Index into the unified action set (the order of the dispatch branches)."""

    SETUP_SETTLEMENT = 0
    SETUP_ROAD = 1
    ROLL_DICE = 2
    DISCARD = 3
    MOVE_ROBBER = 4
    BUILD_ROAD = 5
    BUILD_SETTLEMENT = 6
    BUILD_CITY = 7
    BUY_DEVELOPMENT_CARD = 8
    PLAY_KNIGHT = 9
    PLAY_ROAD_BUILDING = 10
    PLAY_YEAR_OF_PLENTY = 11
    PLAY_MONOPOLY = 12
    MARITIME_TRADE = 13
    PROPOSE_TRADE = 14
    ACCEPT_TRADE = 15
    REJECT_TRADE = 16
    END_TURN = 17


N_ACTION_TYPES = len(ActionType)


class ActionParams(NamedTuple):
    """Packed parameters for the unified ``apply_action`` / ``action_available``.

    Single game: all fields are 0-d arrays (batch by vmapping). Each action
    reads only what it needs; unused fields are ignored. By action:

    - vertex/edge/tile/resource/give (single index)  -> ``idx``
      (Discard's ``idx`` is the resource to give up one card of; ProposeTrade's
      packs (give, receive) as ``give * N_RESOURCES + receive``)
    - victim / receive / second resource (Year of Plenty) / trade partner
      -> ``target``
    - RollDice reads ``idx`` as an optional forced outcome (2..12), and
      BuyDevelopmentCard as an optional forced card type (1..5, meaning type
      ``idx - 1``); any other value, including the flat table's 0, samples
      from the state key — the chance-node seams for stochastic search.
    - parameterless actions (PlayRoadBuilding, EndTurn) read nothing.

    ``target`` follows the ``victim == -1`` ("steal from no one") convention for
    the robber actions.
    """

    idx: IndexParam  # primary index
    target: IndexParam  # secondary index (victim / receive / resource_b)


# Branch adapters in ActionType order: each maps the packed ActionParams onto the
# single-game core's native param shape. Each core takes the precomputed
# ``available`` legality (the caller computes it once -- see ``apply_action``); no
# branch recomputes avail.
_APPLY_BRANCHES = (
    lambda lay, st, pp, av: _setup_settlement_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _setup_road_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _roll_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _discard_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _move_robber_apply(lay, st, (pp.idx, pp.target), av),
    lambda lay, st, pp, av: _build_road_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _build_settlement_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _build_city_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _buy_dev_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _knight_apply(lay, st, (pp.idx, pp.target), av),
    lambda lay, st, pp, av: _road_building_apply(lay, st, None, av),
    lambda lay, st, pp, av: _yop_apply(lay, st, (pp.idx, pp.target), av),
    lambda lay, st, pp, av: _monopoly_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _maritime_apply(lay, st, (pp.idx, pp.target), av),
    lambda lay, st, pp, av: _propose_trade_apply(lay, st, (pp.idx, pp.target), av),
    lambda lay, st, pp, av: _accept_trade_apply(lay, st, None, av),
    lambda lay, st, pp, av: _reject_trade_apply(lay, st, None, av),
    lambda lay, st, pp, av: _end_turn_apply(lay, st, None, av),
)

_AVAIL_BRANCHES = (
    lambda lay, st, pp: _setup_settlement_avail(lay, st, pp.idx),
    lambda lay, st, pp: _setup_road_avail(lay, st, pp.idx),
    lambda lay, st, pp: _roll_avail(lay, st, None),
    lambda lay, st, pp: _discard_avail(lay, st, pp.idx),
    lambda lay, st, pp: _move_robber_avail(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _build_road_avail(lay, st, pp.idx),
    lambda lay, st, pp: _build_settlement_avail(lay, st, pp.idx),
    lambda lay, st, pp: _build_city_avail(lay, st, pp.idx),
    lambda lay, st, pp: _buy_dev_avail(lay, st, None),
    lambda lay, st, pp: _knight_avail(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _road_building_avail(lay, st, None),
    lambda lay, st, pp: _yop_avail(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _monopoly_avail(lay, st, pp.idx),
    lambda lay, st, pp: _maritime_avail(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _propose_trade_avail(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _accept_trade_avail(lay, st, None),
    lambda lay, st, pp: _reject_trade_avail(lay, st, None),
    lambda lay, st, pp: _end_turn_avail(lay, st, None),
)


def apply_action(
    layout: BoardLayout,
    state: BoardState,
    action_type: ActionTypeArray,
    params: ActionParams,
    available: Mask,
    victory_points_to_win: int = VICTORY_POINTS_TO_WIN,
) -> tuple[BoardState, ResultCode]:
    """Apply ``action_type`` (single game) and return (new state, ActionResult code).

    ``available`` is the precomputed legality of this ``(action_type, params)``
    move: if unset the state is returned unchanged with ``INVALID``. Obtain it
    from :func:`action_available` (exact for any params) or a cached
    flat-legality sweep (see :func:`flat_legality`). Awards are recomputed and a
    winning move returns ``GAME_COMPLETE``.
    """
    state, result = jax.lax.switch(
        action_type, _APPLY_BRANCHES, layout, state, params, available
    )
    # Only a gated BuildRoad / BuildSettlement can change a road length; every
    # other lane skips the DFS (see awards.py).
    lr_needed = jnp.where(
        action_type == ActionType.BUILD_ROAD,
        road_build_needed(state, result),
        jnp.where(
            action_type == ActionType.BUILD_SETTLEMENT,
            settlement_break_needed(state, params.idx, result),
            False,
        ),
    )
    return resolve_step(state, result, lr_needed, victory_points_to_win)


def action_available(
    layout: BoardLayout,
    state: BoardState,
    action_type: ActionTypeArray,
    params: ActionParams,
) -> Mask:
    """Legality of ``action_type`` with ``params`` (single game) as a scalar bool."""
    return cast(
        Mask,
        jax.lax.switch(action_type, _AVAIL_BRANCHES, layout, state, params),
    )
