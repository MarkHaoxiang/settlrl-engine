"""Hand-engineered features over the board, weight-free.

Two families, split by input. :func:`board_features` reads one player's
position off a concrete ``(layout, state)`` — the terms behind
``value.make_heuristic``, which is nothing but a weighting of them (the
weights live with the agents). :func:`target_build` / :func:`maritime_ratio`
read a single ``Observation`` — the trade-sense features the scripted greedy
weighs. Everything is pure and ``jit`` / ``vmap`` compatible.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float
from settlrl_engine.board.dev_cards import DEV_CARD_COST, DEV_CARD_COUNTS, DevCard
from settlrl_engine.board.layout import (
    EDGE_V,
    N_TILES,
    N_VERTICES,
    PORT_V,
    TILE_V,
    BoardLayout,
    TileNumberVec,
)
from settlrl_engine.board.resources import (
    CITY_COST,
    N_RESOURCES,
    ROAD_COST,
    SETTLEMENT_COST,
)
from settlrl_engine.board.state import (
    CITY,
    MAX_SETTLEMENTS,
    SETTLEMENT,
    BoardState,
    BoolScalar,
    Player,
)
from settlrl_engine.env import Observation

Scalar = Float[Array, ""]

_VP_CARD_SHARE = float(DEV_CARD_COUNTS[DevCard.VICTORY_POINT]) / float(
    sum(DEV_CARD_COUNTS)
)
_ROAD_COST_ARR = jnp.asarray(ROAD_COST, jnp.float32)
_SETTLEMENT_COST_ARR = jnp.asarray(SETTLEMENT_COST, jnp.float32)
_CITY_COST_ARR = jnp.asarray(CITY_COST, jnp.float32)
_DEV_COST_ARR = jnp.asarray(DEV_CARD_COST, jnp.float32)


def tile_pips(tile_number: TileNumberVec) -> Float[Array, f"tiles={N_TILES}"]:
    """Expected-production weight per tile: 6 - |7 - number| (0 for the desert)."""
    n = tile_number.astype(jnp.int32)
    return jnp.where(n == 0, 0, 6 - jnp.abs(7 - n)).astype(jnp.float32)


def vertex_pips(tile_number: TileNumberVec) -> Float[Array, f"vertices={N_VERTICES}"]:
    """Summed pips of each vertex's adjacent tiles."""
    pips = tile_pips(tile_number)
    acc = jnp.zeros((N_VERTICES,), jnp.float32)
    return acc.at[TILE_V.reshape(-1)].add(jnp.repeat(pips, TILE_V.shape[1]))


class BoardFeatures(NamedTuple):
    """One player's hand-engineered strength terms on one board."""

    vp: Scalar
    """Total VP: buildings, held awards, and VP cards (own exactly; an
    opponent's by the deck-share prior — ``exact_dev`` picks)."""
    production: Scalar
    """Pip-weighted production of own buildings (robber-aware, city double)."""
    diversity: Scalar
    """Distinct resource types produced."""
    hand: Scalar
    """Hand quality: sqrt of the count per type."""
    scarce: Scalar
    """The hand term weighted by production scarcity: cards the player barely
    produces are the hard ones to replace, so conversions toward them (ports,
    trades) read as gains."""
    over: Scalar
    """Cards past seven — the next 7's discard exposure."""
    n_dev: Scalar
    """Held development cards."""
    best_spot: Scalar
    """Pips of the best settlement spot buildable right now (empty, distance
    rule, touching an own road) — what makes a road worth its cost."""
    n_roads: Scalar
    """Own roads on the board."""
    progress: Scalar
    """Completeness of the closest usable build (max over the three costs)."""
    knights: Scalar
    """Knights played toward Largest Army (capped at the threshold)."""
    ports: Scalar
    """Owned-port leverage: a 2:1 port is worth the production it converts, a
    3:1 a fraction of all production."""
    wheat_ore: Scalar
    """The wheat + ore share of production (cities and dev cards)."""
    race: Scalar
    """Closing urgency: VPs above six, squared."""
    numbers: Scalar
    """Distinct dice numbers collected on (income smoothness)."""
    n_spots: Scalar
    """sqrt of how many spots are buildable — expansion optionality."""
    fill: Scalar
    """Summed completeness over all three builds (measured *harmful* as a
    value term: it rewards hoarding toward several builds at once)."""
    held_knights: Scalar
    """Unplayed knights in hand (capped) — army-race potential."""
    second_spot: Scalar
    """Pips of the second-best buildable spot — expansion depth beyond the
    single best."""
    reach: Scalar
    """Pips of the best spot exactly one more road away (through own or empty
    vertices) — what the next road could unlock."""
    army_lead: Scalar
    """Knights played minus the best opponent's (clipped ±3): the Largest
    Army race's margin, not just own progress."""
    wood_brick: Scalar
    """The wood + brick share of production — the expansion engine, the
    counterpart of ``wheat_ore``."""
    settlements: Scalar
    """Settlements standing (city upgrades return them to stock)."""
    cities: Scalar
    """Cities built."""
    blocked_pips: Scalar
    """Own production pips the robber is denying right now — what a knight
    or a 7 would recover."""
    biggest_stack: Scalar
    """Largest single-resource count in hand — monopoly and discard
    exposure concentrated in one type."""
    hand_types: Scalar
    """Distinct resource types held — trade flexibility."""
    affordable: Scalar
    """How many of the four buys (road/settlement/city/dev) the hand covers
    right now (0-4) — immediately convertible purchasing power."""
    road_lead: Scalar
    """Own road count minus the best opponent's (clipped ±5) — the cheap
    Longest Road race proxy (the exact trail DFS is too hot for a sweep)."""
    port_count: Scalar
    """Distinct port kinds owned (each 2:1 counts one, the 3:1 counts one)."""
    reach2: Scalar
    """Pips of the best spot exactly two roads away (through own or empty
    vertices), beyond ``reach`` — deeper expansion on the horizon."""


def board_features(
    layout: BoardLayout, state: BoardState, p: Player, exact_dev: BoolScalar
) -> BoardFeatures:
    """Every :class:`BoardFeatures` term for player ``p`` (one fused pass —
    the terms share their intermediates)."""
    vp = state.victory_points[p].astype(jnp.float32)
    vp += 2.0 * (state.longest_road_owner == p)
    vp += 2.0 * (state.largest_army_owner == p)
    raw_pips = tile_pips(layout.tile_number)
    pips = raw_pips * (jnp.arange(N_TILES) != state.robber)
    weight = (
        ((state.vertex_owner[TILE_V] == p + 1) * state.vertex_type[TILE_V])
        .sum(axis=1)
        .astype(jnp.float32)
    )
    per_tile = pips * weight  # (T,)
    per_res = (
        jnp.zeros((5,), jnp.float32)
        .at[layout.tile_resource.astype(jnp.int32) % 5]  # desert pips are 0
        .add(per_tile)
    )
    production = per_res.sum()

    res = state.player_resources[p].astype(jnp.float32)

    n_dev = state.dev_hand[p].astype(jnp.float32).sum()
    own_vp_cards = state.dev_hand[p, DevCard.VICTORY_POINT].astype(jnp.float32)
    dev_vp = jnp.where(exact_dev, own_vp_cards, n_dev * _VP_CARD_SHARE)

    own_road = state.edge_road == p + 1
    occ = state.vertex_owner > 0
    u, v = EDGE_V[:, 0], EDGE_V[:, 1]
    nb_occ = jnp.zeros((N_VERTICES,), bool).at[u].max(occ[v]).at[v].max(occ[u])
    touched = jnp.zeros((N_VERTICES,), bool).at[u].max(own_road).at[v].max(own_road)
    is_settlement = (state.vertex_owner == p + 1) & (state.vertex_type == SETTLEMENT)
    in_stock = is_settlement.sum() < MAX_SETTLEMENTS
    spot = ~occ & ~nb_occ & touched & in_stock

    def completeness(cost: Float[Array, f"resources={N_RESOURCES}"]) -> Scalar:
        return jnp.minimum(res, cost).sum() / cost.sum()

    deck_left = state.dev_deck.astype(jnp.int32).sum() > 0
    each = jnp.stack(
        [
            completeness(_SETTLEMENT_COST_ARR) * jnp.any(spot),
            completeness(_CITY_COST_ARR) * jnp.any(is_settlement),
            completeness(_DEV_COST_ARR) * deck_left,
        ]
    )

    port_alloc = layout.port_allocation.astype(jnp.int32)
    on_port = (state.vertex_owner[PORT_V] == p + 1).any(axis=1)  # (P_ports,)
    has_2to1 = jnp.zeros((5,), bool).at[port_alloc % 5].max(on_port & (port_alloc < 5))
    has_3to1 = jnp.any(on_port & (port_alloc == 5))

    v_pips = vertex_pips(layout.tile_number)
    spot_pips = jnp.where(spot, v_pips, 0.0)
    # One more road away: an empty edge leaving a road-touched vertex the
    # player may build through (own or empty) reaches its far end.
    empty = state.edge_road == 0
    pass_ok = touched & ((state.vertex_owner == 0) | (state.vertex_owner == p + 1))
    reach_v = (
        jnp.zeros((N_VERTICES,), bool)
        .at[u]
        .max(empty & pass_ok[v])
        .at[v]
        .max(empty & pass_ok[u])
    )
    reach_spot = reach_v & ~occ & ~nb_occ & ~spot & in_stock

    knights_mine = state.knights_played[p].astype(jnp.float32)
    others = jnp.where(
        jnp.arange(state.n_players) == p, 0, state.knights_played
    ).astype(jnp.float32)

    # Two roads out: continue from reach vertices that are empty.
    pass2 = reach_v & (state.vertex_owner == 0)
    reach2_v = (
        jnp.zeros((N_VERTICES,), bool)
        .at[u]
        .max(empty & pass2[v])
        .at[v]
        .max(empty & pass2[u])
    )
    reach2_spot = reach2_v & ~occ & ~nb_occ & ~spot & ~reach_spot & in_stock

    road_counts = (
        jnp.zeros((state.n_players + 1,), jnp.float32).at[state.edge_road].add(1.0)
    )[1:]
    road_others = jnp.where(jnp.arange(state.n_players) == p, -1.0, road_counts)

    affordable = jnp.stack(
        [
            jnp.all(res >= _ROAD_COST_ARR),
            jnp.all(res >= _SETTLEMENT_COST_ARR),
            jnp.all(res >= _CITY_COST_ARR),
            jnp.all(res >= _DEV_COST_ARR),
        ]
    )

    return BoardFeatures(
        vp=vp + dev_vp,
        production=production,
        diversity=(per_res > 0).sum().astype(jnp.float32),
        hand=jnp.sqrt(res).sum(),
        scarce=(jnp.sqrt(res) / (1.0 + per_res)).sum(),
        over=jnp.maximum(res.sum() - 7.0, 0.0),
        n_dev=n_dev,
        best_spot=jnp.max(spot_pips),
        n_roads=own_road.sum().astype(jnp.float32),
        progress=jnp.max(each),
        knights=jnp.minimum(state.knights_played[p].astype(jnp.float32), 3.0),
        ports=(has_2to1 * per_res).sum() + 0.3 * has_3to1 * production,
        wheat_ore=per_res[1] + per_res[4],
        race=jnp.maximum(vp + dev_vp - 6.0, 0.0) ** 2,
        numbers=(
            jnp.zeros((13,), jnp.bool_)
            .at[layout.tile_number.astype(jnp.int32)]
            .max(weight > 0)[2:]
            .sum()
            .astype(jnp.float32)
        ),
        n_spots=jnp.sqrt(spot.sum().astype(jnp.float32)),
        fill=each.sum(),
        held_knights=jnp.minimum(
            state.dev_hand[p, DevCard.KNIGHT].astype(jnp.float32), 2.0
        ),
        second_spot=jnp.sort(spot_pips)[-2],
        reach=jnp.max(jnp.where(reach_spot, v_pips, 0.0)),
        army_lead=jnp.clip(knights_mine - jnp.max(others), -3.0, 3.0),
        wood_brick=per_res[2] + per_res[3],
        settlements=is_settlement.sum().astype(jnp.float32),
        cities=((state.vertex_owner == p + 1) & (state.vertex_type == CITY))
        .sum()
        .astype(jnp.float32),
        blocked_pips=raw_pips[state.robber] * weight[state.robber],
        biggest_stack=res.max(),
        hand_types=(res > 0).sum().astype(jnp.float32),
        affordable=affordable.sum().astype(jnp.float32),
        road_lead=jnp.clip(road_counts[p] - jnp.max(road_others), -5.0, 5.0),
        port_count=has_2to1.sum().astype(jnp.float32) + has_3to1,
        reach2=jnp.max(jnp.where(reach2_spot, v_pips, 0.0)),
    )


# -- Observation-side features (the scripted greedy's trade sense) -----------


def target_build(
    obs: Observation,
) -> tuple[
    Float[Array, f"resources={N_RESOURCES}"],
    Float[Array, f"resources={N_RESOURCES}"],
    Float[Array, f"resources={N_RESOURCES}"],
]:
    """The next build worth saving for and its economics: ``(cost, need,
    surplus)``. City with a settlement to upgrade, else a settlement with a
    spot buildable right now (empty, distance rule, touching an own road),
    else a dev card."""
    held = obs["self_resources"].astype(jnp.float32)
    me = obs["self"].astype(jnp.uint8) + 1
    owner = obs["vertex_owner"]
    has_settlement = jnp.any((owner == me) & (obs["vertex_type"] == SETTLEMENT))
    own_road = obs["edge_road"] == me
    occ = owner > 0
    u, v = EDGE_V[:, 0], EDGE_V[:, 1]
    nb_occ = jnp.zeros((N_VERTICES,), bool).at[u].max(occ[v]).at[v].max(occ[u])
    touched = jnp.zeros((N_VERTICES,), bool).at[u].max(own_road).at[v].max(own_road)
    has_spot = jnp.any(~occ & ~nb_occ & touched)
    cost = jnp.where(
        has_settlement,
        _CITY_COST_ARR,
        jnp.where(has_spot, _SETTLEMENT_COST_ARR, _DEV_COST_ARR),
    )
    return cost, jnp.maximum(cost - held, 0.0), jnp.maximum(held - cost, 0.0)


def maritime_ratio(obs: Observation) -> Float[Array, f"resources={N_RESOURCES}"]:
    """The observer's bank ratio per resource: 2 at the matching port, 3 with
    a generic port, else 4."""
    me = obs["self"].astype(jnp.uint8) + 1
    port_alloc = obs["port_allocation"].astype(jnp.int32)
    on_port = (obs["vertex_owner"][PORT_V] == me).any(axis=1)
    has_2to1 = (
        jnp.zeros((N_RESOURCES,), bool)
        .at[port_alloc % N_RESOURCES]
        .max(on_port & (port_alloc < N_RESOURCES))
    )
    has_3to1 = jnp.any(on_port & (port_alloc == N_RESOURCES))
    return jnp.where(has_2to1, 2.0, jnp.where(has_3to1, 3.0, 4.0))
