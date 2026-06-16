"""Goal economics: scoring every candidate build goal the planner could adopt.

Candidates: a city on each own settlement, a settlement on each spot
reachable within ``depth`` new roads (paths from BFS), a Longest Road grab
when within two trail extensions of taking the card, a dev-card buy, and a
single road extending the network. Scores are quality-weighted pips
discounted by the bottleneck rounds our production needs to afford the goal,
a closing-urgency bonus near 10 VP (a goal that wins outright dominates
everything), and the spot-race model — an opponent road already touching a
spot settles it the moment they afford it, so a multi-road path of ours
loses that race.
"""

from __future__ import annotations

import numpy as np
from settlrl_engine.board.resources import N_RESOURCES
from settlrl_engine.env import ActionType

from settlrl_agents.planner.pov import (
    EDGE_ENDPOINTS,
    VERTEX_EDGES,
    VERTEX_NEIGHBORS,
    Pov,
)
from settlrl_agents.planner.tree import Blackboard, Plan, Step

# Resource quality in pip terms [sheep, wheat, wood, brick, ore]: wheat/ore
# feed cities and dev cards (the late game), wood/brick only expansion.
RES_WEIGHT = np.asarray([0.9, 1.3, 1.0, 1.1, 1.3])

_NO_OWNER = 255  # NO_INDEX as stored in the award-owner observation fields


def wprod(prod: np.ndarray) -> float:
    """Production pips weighted by resource quality."""
    return float(prod @ RES_WEIGHT)


def port_bonus(pov: Pov, vertex: int, prod: np.ndarray) -> float:
    """Value of the port under a prospective settlement at ``vertex``."""
    kind = pov.port_kind(vertex)
    if kind is None:
        return 0.0
    if kind >= N_RESOURCES:  # generic 3:1
        return 0.6
    return 0.3 + 0.35 * float(prod[kind])  # 2:1 ports want matching production


def _extension_edge(pov: Pov) -> int | None:
    """An empty edge our road network can grow along (longest-road filler)."""
    best, best_score = None, -np.inf
    reachable = {int(v) for v in pov.my_vertices}
    for e in pov.my_edges:
        for v in EDGE_ENDPOINTS[e]:
            if int(pov.vertex_owner[v]) in (0, pov.me + 1):
                reachable.add(int(v))
    for e in range(pov.edge_road.shape[0]):
        if int(pov.edge_road[e]) != 0:
            continue
        a, b = int(EDGE_ENDPOINTS[e, 0]), int(EDGE_ENDPOINTS[e, 1])
        if a not in reachable and b not in reachable:
            continue
        far = b if a in reachable else a
        score = float(pov.vertex_production(far).sum())
        if score > best_score:
            best, best_score = e, score
    return best


def _trail_extension_steps(pov: Pov, needed: int) -> list[Step] | None:
    """``needed`` empty edges growing our longest trail from one of its free
    ends (so each realized step lengthens the trail by one)."""
    _, ends = pov.my_longest_trail()
    for v in sorted(ends):
        for e in VERTEX_EDGES[v]:
            if int(pov.edge_road[e]) != 0:
                continue
            if needed == 1:
                return [Step(ActionType.BUILD_ROAD, e)]
            a, b = int(EDGE_ENDPOINTS[e, 0]), int(EDGE_ENDPOINTS[e, 1])
            far = b if a == v else a
            if int(pov.vertex_owner[far]) not in (0, pov.me + 1):
                continue  # cannot continue through an opponent's building
            for e2 in VERTEX_EDGES[far]:
                if e2 != e and int(pov.edge_road[e2]) == 0:
                    return [
                        Step(ActionType.BUILD_ROAD, e),
                        Step(ActionType.BUILD_ROAD, e2),
                    ]
    return None


def plan_candidates(pov: Pov, bb: Blackboard, depth: int) -> list[tuple[float, Plan]]:
    """Every candidate goal, scored (see the module docstring)."""
    vp = pov.my_total_vp

    def urgency(vp_gain: int, builds: int) -> float:
        if vp + vp_gain >= 10:
            return 25.0 / builds  # a winning goal, fastest first
        return 1.5 * max(vp - 6, 0) * vp_gain / builds

    income = pov.my_production * pov.n_players / 36.0 + 0.04  # cards per round

    def turns(steps: list[Step]) -> float:
        """Bottleneck rounds until our production affords the whole goal."""
        total = sum((s.cost for s in steps), np.zeros(N_RESOURCES, dtype=np.int64))
        missing = np.clip(total - pov.hand, 0, None).astype(np.float64)
        return float(min(np.max(missing / income), 20.0))

    cands: list[tuple[float, Plan]] = []
    for v in pov.my_settlements:
        steps = [Step(ActionType.BUILD_CITY, int(v))]
        score = 7.5 + wprod(pov.vertex_production(int(v)))
        score += urgency(1, 1) - 0.35 * turns(steps)
        cands.append((score + bb.noise(), Plan(f"city@{int(v)}", steps)))

    def their_road(edge: int) -> bool:
        return int(pov.edge_road[edge]) not in (0, pov.me + 1)

    for v, path in pov.expansion_paths(depth):
        gain = pov.vertex_production(v)
        new_types = int(((gain > 0) & (pov.my_production == 0)).sum())
        steps = [Step(ActionType.BUILD_ROAD, e) for e in path]
        steps.append(Step(ActionType.BUILD_SETTLEMENT, v))
        score = 5.5 + wprod(gain) + 1.8 * new_types + port_bonus(pov, v, gain)
        score += -1.0 * len(path) + urgency(1, 1 + len(path))
        score += -0.35 * turns(steps)
        # The race for the spot: an opponent road already touching it settles
        # the moment they afford it, so a multi-road path of ours loses; one
        # edge away, claiming first is worth a hurry. A path edge adjacent to
        # their network can be cut under us either way.
        if any(their_road(e) for e in VERTEX_EDGES[v]):
            score -= 2.0 * len(path) - 1.0
        elif any(their_road(e2) for n in VERTEX_NEIGHBORS[v] for e2 in VERTEX_EDGES[n]):
            score += 0.5
        cut_risk = sum(
            1
            for e in path
            if any(
                their_road(e2)
                for vv in EDGE_ENDPOINTS[e]
                for e2 in VERTEX_EDGES[int(vv)]
            )
        )
        score -= 0.4 * cut_risk
        cands.append((score + bb.noise(), Plan(f"settle@{v}", steps)))
    if pov.longest_road_owner != pov.me:
        target = 5 if pov.longest_road_owner == _NO_OWNER else pov.longest_road_len + 1
        length, _ = pov.my_longest_trail()
        needed = target - length
        if 1 <= needed <= 2:
            steps_or_none = _trail_extension_steps(pov, needed)
            if steps_or_none is not None:
                score = 10.0 - 2.5 * needed + urgency(2, needed)
                score += -0.35 * turns(steps_or_none)
                cands.append((score + bb.noise(), Plan("longroad", steps_or_none)))
    army_live = pov.largest_army_owner != pov.me and int(pov.knights_played[pov.me])
    dev_score = 3.0 + 0.8 * bool(army_live) + 1.2 * (vp >= 8)
    dev_score -= 0.35 * turns([Step(ActionType.BUY_DEVELOPMENT_CARD)])
    cands.append(
        (
            dev_score + bb.noise(),
            Plan(
                "dev",
                [
                    Step(
                        ActionType.BUY_DEVELOPMENT_CARD,
                        idx=int(pov.dev_card_count[pov.me]),
                    )
                ],
            ),
        )
    )
    ext = _extension_edge(pov)
    if ext is not None:
        cands.append(
            (1.0 + bb.noise(), Plan(f"road@{ext}", [Step(ActionType.BUILD_ROAD, ext)]))
        )
    return cands


def choose_plan(
    pov: Pov, bb: Blackboard, depth: int, exclude: str | None
) -> Plan | None:
    """The best candidate goal. ``exclude`` skips the plan just abandoned for
    staleness so a starved goal cannot be re-adopted forever."""
    cands = [c for c in plan_candidates(pov, bb, depth) if c[1].name != exclude]
    if not cands:
        return None
    return max(cands, key=lambda c: c[0])[1]
