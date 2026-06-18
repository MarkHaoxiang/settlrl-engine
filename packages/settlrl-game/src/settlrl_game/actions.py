"""The game's flat action space over reference actions.

The frontend identifies a move by an opaque integer (``flat``). This module
fixes an enumeration of every concrete action (``_ROWS``), maps a flat index to
the ``settlrl_game.reference`` ``Action`` it names (and back), lists the flats legal
in a game, and decodes a flat into the JSON :class:`ActionModel` the frontend
acts on — geometry expressed in the cube/axial coordinates the SVG board uses.
The ordering is the game's own; the frontend never assumes specific values.
"""

from __future__ import annotations

import settlrl_game.reference as ref
from settlrl_game.botproto import MoveModel
from settlrl_game.convert import (
    _RESOURCE_NAMES,
    EDGE_VERTICES,
    TILE_COORDS,
    VERTEX_COORDS,
    Cube,
    _cube,
)
from settlrl_game.models import ActionModel, EdgeModel, HexModel
from settlrl_game.reference import board as rb

__all__ = [
    "N_FLAT",
    "decode_actions",
    "flat_for_action",
    "flat_for_move",
    "legal_flats",
    "legal_moves",
    "move_for_flat",
    "to_action",
]

_NR = len(ref.RESOURCES)
# Victim slots a robber/knight row can name: "no one" plus the four seats (the
# full domain at every player count, so the flat space is constant).
_VICTIMS = (-1, 0, 1, 2, 3)

# A row is (type, a, b, c): the lowercased action type and up to three int
# parameters (-1 = unused), enough to build the reference action and decode it.
Row = tuple[str, int, int, int]
_ROWS: list[Row] = []


def _add(type_: str, a: int = 0, b: int = -1, c: int = -1) -> None:
    _ROWS.append((type_, a, b, c))


for _v in range(rb.N_VERTICES):
    _add("setup_settlement", _v)
for _e in range(rb.N_EDGES):
    _add("setup_road", _e)
_add("roll_dice")
for _r in range(_NR):
    _add("discard", _r)
for _t in range(rb.N_TILES):
    for _victim in _VICTIMS:
        _add("move_robber", _t, _victim)
for _e in range(rb.N_EDGES):
    _add("build_road", _e)
for _v in range(rb.N_VERTICES):
    _add("build_settlement", _v)
for _v in range(rb.N_VERTICES):
    _add("build_city", _v)
_add("buy_development_card")
for _t in range(rb.N_TILES):
    for _victim in _VICTIMS:
        _add("play_knight", _t, _victim)
_add("play_road_building")
for _a in range(_NR):
    for _b in range(_NR):
        _add("play_year_of_plenty", _a, _b)
for _r in range(_NR):
    _add("play_monopoly", _r)
for _g in range(_NR):
    for _rc in range(_NR):
        _add("maritime_trade", _g, _rc)
for _g in range(_NR):
    for _rc in range(_NR):
        for _partner in range(4):  # the full four-seat partner domain
            _add("propose_trade", _g, _rc, _partner)
_add("accept_trade")
_add("reject_trade")
_add("end_turn")

N_FLAT = len(_ROWS)
_KEY_TO_FLAT: dict[Row, int] = {row: i for i, row in enumerate(_ROWS)}


def _one_index(counts: tuple[int, ...]) -> int:
    """The single set position of a 1:1 trade count vector."""
    return next(i for i, c in enumerate(counts) if c)


def _key(action: ref.Action) -> Row:
    """The table key for a reference action (the inverse of :func:`to_action`).

    Only ever called on actions from ``Game.legal_actions()``, which enumerates
    the 1:1 trade subset and the first owing player's discards — matching the
    table's rows.
    """
    match action:
        case ref.SetupSettlement(vertex=v):
            return ("setup_settlement", v, -1, -1)
        case ref.SetupRoad(edge=e):
            return ("setup_road", e, -1, -1)
        case ref.Roll():
            return ("roll_dice", 0, -1, -1)
        case ref.Discard(resource=r):
            return ("discard", int(r), -1, -1)
        case ref.MoveRobber(tile=t, victim=victim):
            return ("move_robber", t, -1 if victim is None else victim, -1)
        case ref.BuildRoad(edge=e):
            return ("build_road", e, -1, -1)
        case ref.BuildSettlement(vertex=v):
            return ("build_settlement", v, -1, -1)
        case ref.BuildCity(vertex=v):
            return ("build_city", v, -1, -1)
        case ref.BuyDevelopmentCard():
            return ("buy_development_card", 0, -1, -1)
        case ref.PlayKnight(tile=t, victim=victim):
            return ("play_knight", t, -1 if victim is None else victim, -1)
        case ref.PlayRoadBuilding():
            return ("play_road_building", 0, -1, -1)
        case ref.PlayYearOfPlenty(first=a, second=b):
            return ("play_year_of_plenty", int(a), int(b), -1)
        case ref.PlayMonopoly(resource=r):
            return ("play_monopoly", int(r), -1, -1)
        case ref.MaritimeTrade(give=g, receive=rc):
            return ("maritime_trade", int(g), int(rc), -1)
        case ref.ProposeTrade(partner=p, give=g, receive=rc):
            return ("propose_trade", _one_index(g), _one_index(rc), p)
        case ref.AcceptTrade():
            return ("accept_trade", 0, -1, -1)
        case ref.RejectTrade():
            return ("reject_trade", 0, -1, -1)
        case ref.EndTurn():
            return ("end_turn", 0, -1, -1)
        case _:  # pragma: no cover - exhaustive over Action
            raise AssertionError(f"unhandled action: {action!r}")


def to_action(flat: int, game: ref.Game) -> ref.Action:
    """The reference action named by flat index ``flat``.

    Stochastic outcome fields are left unset (the caller injects them). A
    discard row names the first owing player, matching the acting seat during
    the discard phase.
    """
    type_, a, b, _c = _ROWS[flat]
    match type_:
        case "setup_settlement":
            return ref.SetupSettlement(a)
        case "setup_road":
            return ref.SetupRoad(a)
        case "roll_dice":
            return ref.Roll()
        case "discard":
            owing = (p for p in range(game.n_players) if game.pending_discard[p] > 0)
            return ref.Discard(next(owing, 0), ref.Resource(a))
        case "move_robber":
            return ref.MoveRobber(a, None if b < 0 else b)
        case "build_road":
            return ref.BuildRoad(a)
        case "build_settlement":
            return ref.BuildSettlement(a)
        case "build_city":
            return ref.BuildCity(a)
        case "buy_development_card":
            return ref.BuyDevelopmentCard()
        case "play_knight":
            return ref.PlayKnight(a, None if b < 0 else b)
        case "play_road_building":
            return ref.PlayRoadBuilding()
        case "play_year_of_plenty":
            return ref.PlayYearOfPlenty(ref.Resource(a), ref.Resource(b))
        case "play_monopoly":
            return ref.PlayMonopoly(ref.Resource(a))
        case "maritime_trade":
            return ref.MaritimeTrade(ref.Resource(a), ref.Resource(b))
        case "propose_trade":
            return ref.ProposeTrade.one_card(_c, ref.Resource(a), ref.Resource(b))
        case "accept_trade":
            return ref.AcceptTrade()
        case "reject_trade":
            return ref.RejectTrade()
        case "end_turn":
            return ref.EndTurn()
        case _:  # pragma: no cover
            raise AssertionError(f"unhandled flat row: {_ROWS[flat]!r}")


def legal_flats(game: ref.Game) -> list[int]:
    """The flat indices of every action legal in ``game`` right now."""
    return sorted(_KEY_TO_FLAT[_key(a)] for a in game.legal_actions())


def flat_for_action(action: ref.Action) -> int:
    """The flat index naming ``action`` (for the bot service's chosen move)."""
    return _KEY_TO_FLAT[_key(action)]


# -- decode to the wire ActionModel -----------------------------------------

_VERTEX_TYPES = {"setup_settlement", "build_settlement", "build_city"}
_ROAD_TYPES = {"setup_road", "build_road"}
_ROBBER_TYPES = {"move_robber", "play_knight"}
_BASE_LABELS = {
    "roll_dice": "Roll dice",
    "end_turn": "End turn",
    "buy_development_card": "Buy dev card",
    "play_road_building": "Road building",
    "accept_trade": "Accept trade",
    "reject_trade": "Reject trade",
}


def _decode(flat: int) -> ActionModel:
    """Turn one flat index into an :class:`ActionModel`."""
    type_, a, b, c = _ROWS[flat]

    if type_ in _VERTEX_TYPES:
        label = "City" if type_ == "build_city" else "Settlement"
        return ActionModel(
            flat=flat, type=type_, label=label, vertex=_cube(VERTEX_COORDS[a])
        )
    if type_ in _ROAD_TYPES:
        v1, v2 = EDGE_VERTICES[a]
        edge = EdgeModel(a=_cube(VERTEX_COORDS[v1]), b=_cube(VERTEX_COORDS[v2]))
        return ActionModel(flat=flat, type=type_, label="Road", edge=edge)
    if type_ in _ROBBER_TYPES:
        q, r = TILE_COORDS[a]
        verb = "Knight" if type_ == "play_knight" else "Move robber"
        steal = f" (steal P{b + 1})" if b >= 0 else ""
        return ActionModel(
            flat=flat,
            type=type_,
            label=f"{verb}{steal}",
            tile=HexModel(q=q, r=r),
            victim=b,
        )
    if type_ == "discard":
        res = _RESOURCE_NAMES[a]
        return ActionModel(flat=flat, type=type_, label=f"Discard {res}", resource=res)
    if type_ == "play_monopoly":
        res = _RESOURCE_NAMES[a]
        return ActionModel(
            flat=flat, type=type_, label=f"Monopoly: {res}", resource=res
        )
    if type_ == "play_year_of_plenty":
        first, second = _RESOURCE_NAMES[a], _RESOURCE_NAMES[b]
        return ActionModel(
            flat=flat,
            type=type_,
            label=f"Plenty: {first} + {second}",
            resources=[first, second],
        )
    if type_ == "maritime_trade":
        give, receive = _RESOURCE_NAMES[a], _RESOURCE_NAMES[b]
        return ActionModel(
            flat=flat,
            type=type_,
            label=f"Trade {give} → {receive}",
            give=give,
            receive=receive,
        )
    if type_ == "propose_trade":
        give, receive = _RESOURCE_NAMES[a], _RESOURCE_NAMES[b]
        return ActionModel(
            flat=flat,
            type=type_,
            label=f"Offer P{c + 1} {give} → {receive}",
            give=give,
            receive=receive,
            partner=c,
        )
    return ActionModel(flat=flat, type=type_, label=_BASE_LABELS.get(type_, type_))


def decode_actions(flat_indices: list[int]) -> list[ActionModel]:
    """Decode a list of legal flat action indices into action descriptors."""
    return [_decode(f) for f in flat_indices]


# -- structured MoveModel <-> flat (the bot-service wire form) ----------------
#
# `MoveModel` names an action in board coordinates (the inverse of the engine's
# opaque flat index), so it survives across engine versions. The tables below
# invert the geometry lookups so a structured move maps back to its flat row.

_CUBE_TO_VERTEX: dict[Cube, int] = {c: v for v, c in enumerate(VERTEX_COORDS)}
_QR_TO_TILE: dict[tuple[int, int], int] = {qr: t for t, qr in enumerate(TILE_COORDS)}
_EDGE_BY_CUBES: dict[frozenset[Cube], int] = {
    frozenset((VERTEX_COORDS[v1], VERTEX_COORDS[v2])): e
    for e, (v1, v2) in enumerate(EDGE_VERTICES)
}
_RESOURCE_INDEX: dict[str, int] = {n: i for i, n in enumerate(_RESOURCE_NAMES)}


def move_for_flat(flat: int) -> MoveModel:
    """The structured move naming flat index ``flat``."""
    type_, a, b, c = _ROWS[flat]
    if type_ in _VERTEX_TYPES:
        return MoveModel(type=type_, vertex=_cube(VERTEX_COORDS[a]))
    if type_ in _ROAD_TYPES:
        v1, v2 = EDGE_VERTICES[a]
        edge = EdgeModel(a=_cube(VERTEX_COORDS[v1]), b=_cube(VERTEX_COORDS[v2]))
        return MoveModel(type=type_, edge=edge)
    if type_ in _ROBBER_TYPES:
        q, r = TILE_COORDS[a]
        return MoveModel(
            type=type_, tile=HexModel(q=q, r=r), victim=None if b < 0 else b
        )
    if type_ in ("discard", "play_monopoly"):
        return MoveModel(type=type_, resource=_RESOURCE_NAMES[a])
    if type_ == "play_year_of_plenty":
        return MoveModel(type=type_, resources=[_RESOURCE_NAMES[a], _RESOURCE_NAMES[b]])
    if type_ == "maritime_trade":
        return MoveModel(
            type=type_, give=_RESOURCE_NAMES[a], receive=_RESOURCE_NAMES[b]
        )
    if type_ == "propose_trade":
        return MoveModel(
            type=type_, give=_RESOURCE_NAMES[a], receive=_RESOURCE_NAMES[b], partner=c
        )
    return MoveModel(type=type_)


def _res(name: str | None) -> int:
    if name is None or name not in _RESOURCE_INDEX:
        raise ValueError(f"unknown resource: {name!r}")
    return _RESOURCE_INDEX[name]


def flat_for_move(move: MoveModel) -> int:
    """The flat index naming a structured ``move`` (raises ``ValueError`` if the
    move names no action: an unknown type or off-board geometry)."""
    t = move.type
    try:
        if t in _VERTEX_TYPES:
            if move.vertex is None:
                raise ValueError("vertex action needs a vertex")
            v = move.vertex
            key: Row = (t, _CUBE_TO_VERTEX[(v.q, v.r, v.s)], -1, -1)
        elif t in _ROAD_TYPES:
            if move.edge is None:
                raise ValueError("road action needs an edge")
            a, b = move.edge.a, move.edge.b
            cubes = frozenset(((a.q, a.r, a.s), (b.q, b.r, b.s)))
            key = (t, _EDGE_BY_CUBES[cubes], -1, -1)
        elif t in _ROBBER_TYPES:
            if move.tile is None:
                raise ValueError("robber action needs a tile")
            victim = -1 if move.victim is None else move.victim
            key = (t, _QR_TO_TILE[(move.tile.q, move.tile.r)], victim, -1)
        elif t in ("discard", "play_monopoly"):
            key = (t, _res(move.resource), -1, -1)
        elif t == "play_year_of_plenty":
            if move.resources is None or len(move.resources) != 2:
                raise ValueError("year of plenty needs two resources")
            key = (t, _res(move.resources[0]), _res(move.resources[1]), -1)
        elif t == "maritime_trade":
            key = (t, _res(move.give), _res(move.receive), -1)
        elif t == "propose_trade":
            if move.partner is None:
                raise ValueError("propose trade needs a partner")
            key = (t, _res(move.give), _res(move.receive), move.partner)
        else:
            key = (t, 0, -1, -1)
        return _KEY_TO_FLAT[key]
    except KeyError as exc:
        raise ValueError(f"no such move: {move!r}") from exc


def legal_moves(game: ref.Game) -> list[MoveModel]:
    """The structured moves legal in ``game`` right now."""
    return [move_for_flat(f) for f in legal_flats(game)]
