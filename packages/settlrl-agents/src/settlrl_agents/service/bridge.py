"""Bridge a reference game into a single-game engine env for the bot service.

The game server speaks ``settlrl_game.reference``; the agents (``settlrl_agents``) reason
on a ``settlrl_engine`` board. This module rebuilds the engine's ``(BoardLayout,
BoardState)`` and ``BeliefState`` from a reference ``Game`` + ``Belief`` (the
inverse of the engine test's ``conversion`` bridge), wraps them in a
``BatchedSettlrlEnv`` so a policy can observe and choose, and maps the policy's
chosen engine action back to the game's flat index.

Index translation is by cube coordinate, the only thing the two libraries share
(see ``settlrl_game.reference.board`` and ``settlrl_engine.board.layout``).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import settlrl_game.reference as ref
from settlrl_engine.belief import BeliefState
from settlrl_engine.board import layout as eng_layout
from settlrl_engine.board import make_board
from settlrl_engine.board.dev_cards import DEV_CARD_COUNTS
from settlrl_engine.board.layout import (
    N_EDGES,
    N_PORTS,
    N_TILES,
    N_VERTICES,
    PORT_V,
    BoardLayout,
)
from settlrl_engine.board.resources import N_RESOURCES
from settlrl_engine.board.state import NO_INDEX, BoardState, GamePhase
from settlrl_engine.env import N_FLAT, BatchedSettlrlEnv, flat_to_action
from settlrl_engine.env.batched import (
    _agent_selection_b,
    _total_vp_b,
)
from settlrl_engine.mechanics.action import ActionType
from settlrl_engine.mechanics.flat import flat_available_b
from settlrl_engine.mechanics.trade import _COUNT_BITS, _COUNT_MASK, _PARTNER_BITS
from settlrl_game.actions import flat_for_action
from settlrl_game.reference import board as rb

_N_DEV = len(DEV_CARD_COUNTS)
_DESERT = N_RESOURCES  # engine Tile value for the desert
_GENERIC_PORT = N_RESOURCES  # engine Port value for the 3:1 port

# -- index bridges (engine index <-> reference index, via cube coords) -------

_ENG2REF_VERTEX = [
    rb.cube_to_vertex(eng_layout.vertex_cube(v)) for v in range(N_VERTICES)
]
_ENG2REF_TILE = [rb.cube_to_tile(eng_layout.tile_cube(t)) for t in range(N_TILES)]


def _eng2ref_edge(e: int) -> int:
    a, b = eng_layout.edge_cubes(e)
    return rb.edge_between(rb.cube_to_vertex(a), rb.cube_to_vertex(b))


_ENG2REF_EDGE = [_eng2ref_edge(e) for e in range(N_EDGES)]
_REF2ENG_TILE = [0] * N_TILES
for _e, _r in enumerate(_ENG2REF_TILE):
    _REF2ENG_TILE[_r] = _e


def _engine_layout_state(game: ref.Game) -> tuple[BoardLayout, BoardState]:
    """The engine ``(BoardLayout, BoardState)`` describing ``game`` (batch 1)."""
    layout0, state0 = make_board(1, seed=0, n_players=game.n_players)
    rl = game.layout

    tile_resource = np.full(N_TILES, _DESERT, dtype=np.int32)
    tile_number = np.zeros(N_TILES, dtype=np.int32)
    for et in range(N_TILES):
        rt = _ENG2REF_TILE[et]
        res = rl.tile_resource[rt]
        tile_resource[et] = _DESERT if res is None else int(res)
        tile_number[et] = rl.tile_number[rt]

    # Port allocation, by engine port position: find the reference port covering
    # those two coastal vertices and take its type.
    port_alloc = np.full(N_PORTS, _GENERIC_PORT, dtype=np.int32)
    for i in range(N_PORTS):
        rverts = {
            _ENG2REF_VERTEX[int(PORT_V[i, 0])],
            _ENG2REF_VERTEX[int(PORT_V[i, 1])],
        }
        for port in rl.ports:
            if set(port.vertices) == rverts:
                port_alloc[i] = (
                    _GENERIC_PORT
                    if port.type is ref.PortType.GENERIC
                    else int(port.type.value)
                )
                break

    layout = layout0._replace(
        tile_resource=jnp.asarray(tile_resource[None], jnp.uint8),
        tile_number=jnp.asarray(tile_number[None], jnp.uint8),
        port_allocation=jnp.asarray(port_alloc[None], jnp.uint8),
    )

    n = game.n_players
    vertex_owner = np.zeros(N_VERTICES, dtype=np.int32)
    vertex_type = np.zeros(N_VERTICES, dtype=np.int32)
    for ev in range(N_VERTICES):
        owner = game.buildings.get(_ENG2REF_VERTEX[ev])
        if owner is not None:
            vertex_owner[ev] = owner[0] + 1
            vertex_type[ev] = int(owner[1])
    edge_road = np.zeros(N_EDGES, dtype=np.int32)
    for ee in range(N_EDGES):
        player = game.roads.get(_ENG2REF_EDGE[ee])
        if player is not None:
            edge_road[ee] = player + 1

    resources = np.array(
        [
            [pl.resources[ref.Resource(r)] for r in range(N_RESOURCES)]
            for pl in game.players
        ]
    )
    dev_hand = np.array(
        [[pl.dev_cards[ref.DevCard(c)] for c in range(_N_DEV)] for pl in game.players]
    )
    dev_deck = np.array([game.dev_deck[ref.DevCard(c)] for c in range(_N_DEV)])
    dev_bought = np.array(
        [
            game.players[game.current_player].dev_bought_this_turn[ref.DevCard(c)]
            for c in range(_N_DEV)
        ]
    )
    give = game.trade_give or (0,) * N_RESOURCES
    receive = game.trade_receive or (0,) * N_RESOURCES

    def _award(owner: int | None) -> int:
        return NO_INDEX if owner is None else owner

    state = state0._replace(
        vertex_owner=jnp.asarray(vertex_owner[None], jnp.uint8),
        vertex_type=jnp.asarray(vertex_type[None], jnp.uint8),
        edge_road=jnp.asarray(edge_road[None], jnp.uint8),
        robber=jnp.asarray([_REF2ENG_TILE[game.robber]], jnp.uint8),
        player_resources=jnp.asarray(resources[None], jnp.uint8),
        victory_points=jnp.asarray(
            [[game.building_vp(p) for p in range(n)]], jnp.uint8
        ),
        dev_deck=jnp.asarray(dev_deck[None], jnp.uint8),
        dev_hand=jnp.asarray(dev_hand[None], jnp.uint8),
        knights_played=jnp.asarray(
            [[pl.knights_played for pl in game.players]], jnp.uint8
        ),
        phase=jnp.asarray([int(GamePhase[game.phase.name])], jnp.uint8),
        current_player=jnp.asarray([game.current_player], jnp.uint8),
        setup_index=jnp.asarray([game.setup_index], jnp.uint8),
        dice_roll=jnp.asarray([game.dice_roll], jnp.uint8),
        has_rolled=jnp.asarray([int(game.has_rolled)], jnp.uint8),
        dev_played=jnp.asarray([int(game.dev_played_this_turn)], jnp.uint8),
        dev_bought=jnp.asarray(dev_bought[None], jnp.uint8),
        free_roads=jnp.asarray([game.free_roads], jnp.uint8),
        pending_discard=jnp.asarray([game.pending_discard], jnp.uint8),
        trade_partner=jnp.asarray(
            [NO_INDEX if game.trade_partner is None else game.trade_partner], jnp.int32
        ),
        trade_give=jnp.asarray([list(give)], jnp.uint8),
        trade_receive=jnp.asarray([list(receive)], jnp.uint8),
        longest_road_owner=jnp.asarray([_award(game.longest_road_owner)], jnp.int32),
        largest_army_owner=jnp.asarray([_award(game.largest_army_owner)], jnp.int32),
        longest_road_len=jnp.asarray([game.longest_road_len], jnp.uint8),
    )
    return layout, state


def _engine_belief(belief: ref.Belief) -> BeliefState:
    """The engine ``BeliefState`` (batch 1) from a reference ``Belief``."""
    res_lo = np.array(belief.res_lo, dtype=np.uint8)  # (O, P, R)
    res_hi = np.array(belief.res_hi, dtype=np.uint8)
    dev_played = np.array(
        [belief.dev_played[ref.DevCard(c)] for c in range(_N_DEV)], dtype=np.uint8
    )
    return BeliefState(
        res_lo=jnp.asarray(res_lo[None]),
        res_hi=jnp.asarray(res_hi[None]),
        dev_played=jnp.asarray(dev_played[None]),
    )


def engine_env(game: ref.Game, belief: ref.Belief) -> BatchedSettlrlEnv:
    """A single-game engine env whose state mirrors ``game`` (+ ``belief``)."""
    env = BatchedSettlrlEnv(batch_size=1, n_players=game.n_players, track_beliefs=True)
    layout, state = _engine_layout_state(game)
    env._layout, env._state = layout, state
    env._belief = _engine_belief(belief)
    # Refresh the caches that step / observe / masks read off the injected state.
    env._vps = _total_vp_b(state)
    env._avail = flat_available_b(layout, state)
    env._agent_sel = _agent_selection_b(state)
    return env


def _unpack_counts(packed: int) -> tuple[int, ...]:
    return tuple(
        (packed >> (_COUNT_BITS * r)) & _COUNT_MASK for r in range(N_RESOURCES)
    )


def _engine_action_to_ref(atype: int, idx: int, target: int) -> ref.Action:
    """A reference action (with reference indices) from an engine action."""
    victim = None if target < 0 else target
    match ActionType(atype):
        case ActionType.SETUP_SETTLEMENT:
            return ref.SetupSettlement(_ENG2REF_VERTEX[idx])
        case ActionType.SETUP_ROAD:
            return ref.SetupRoad(_ENG2REF_EDGE[idx])
        case ActionType.ROLL_DICE:
            return ref.Roll()
        case ActionType.DISCARD:
            return ref.Discard(0, ref.Resource(idx))  # player ignored by the flat key
        case ActionType.MOVE_ROBBER:
            return ref.MoveRobber(_ENG2REF_TILE[idx], victim)
        case ActionType.BUILD_ROAD:
            return ref.BuildRoad(_ENG2REF_EDGE[idx])
        case ActionType.BUILD_SETTLEMENT:
            return ref.BuildSettlement(_ENG2REF_VERTEX[idx])
        case ActionType.BUILD_CITY:
            return ref.BuildCity(_ENG2REF_VERTEX[idx])
        case ActionType.BUY_DEVELOPMENT_CARD:
            return ref.BuyDevelopmentCard()
        case ActionType.PLAY_KNIGHT:
            return ref.PlayKnight(_ENG2REF_TILE[idx], victim)
        case ActionType.PLAY_ROAD_BUILDING:
            return ref.PlayRoadBuilding()
        case ActionType.PLAY_YEAR_OF_PLENTY:
            return ref.PlayYearOfPlenty(ref.Resource(idx), ref.Resource(target))
        case ActionType.PLAY_MONOPOLY:
            return ref.PlayMonopoly(ref.Resource(idx))
        case ActionType.MARITIME_TRADE:
            return ref.MaritimeTrade(ref.Resource(idx), ref.Resource(target))
        case ActionType.PROPOSE_TRADE:
            return ref.ProposeTrade(
                partner=target & ((1 << _PARTNER_BITS) - 1),
                give=_unpack_counts(idx),
                receive=_unpack_counts(target >> _PARTNER_BITS),
            )
        case ActionType.ACCEPT_TRADE:
            return ref.AcceptTrade()
        case ActionType.REJECT_TRADE:
            return ref.RejectTrade()
        case _:
            return ref.EndTurn()


def game_flat(engine_flat: int) -> int:
    """Translate a policy's chosen engine flat index to a game flat index."""
    atype, params = flat_to_action(jnp.asarray([engine_flat]))
    action = _engine_action_to_ref(
        int(atype[0]), int(params.idx[0]), int(params.target[0])
    )
    return flat_for_action(action)


__all__ = ["N_FLAT", "engine_env", "game_flat"]
