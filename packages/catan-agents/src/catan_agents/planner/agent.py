"""The planner: a stateful decision-tree agent over plain Python logic.

One :class:`~catan_agents.planner.tree.Selector` of leaf handlers covers the
forced phases (setup, discard, robber, trade response, roll); the MAIN-phase
pair carries the strategy: ``ExecutePlan`` keeps a persistent build plan
(chosen by scoring city upgrades, road-path-plus-settlement expansions, dev
buys, and a road filler) and plays its next step the moment it is legal;
``Acquire`` spends the turn closing the plan's resource gap (dev-card plays,
maritime trades at our best port ratio, capped domestic proposals with a
rejected-offer memory). Everything reads legality off the mask, so the tree
never re-implements a rule; an unexpected state just falls through to a
fixed-priority fallback.
"""

from __future__ import annotations

import random

import numpy as np
from catan_engine.board.resources import N_RESOURCES
from catan_engine.board.state import GamePhase
from catan_engine.env import ActionType
from catan_engine.mechanics.trade import pack_trade_single

from catan_agents.planner.pov import (
    EDGE_ENDPOINTS,
    ROW_IDX,
    ROW_TARGET,
    TILE_CORNERS,
    VERTEX_NEIGHBORS,
    Pov,
    flat_row,
)
from catan_agents.planner.tree import Blackboard, Node, Plan, Selector, Step
from catan_agents.shared.policy import (
    GameAgent,
    HostFlatMask,
    HostObservation,
    StatefulPolicy,
)

# Own turns a plan may sit with no step realized before it is abandoned
# (covers goals starved by piece limits or an empty dev deck, which the
# observation cannot rule out up front).
_PLAN_PATIENCE = 12


def _noise(bb: Blackboard) -> float:
    """Sub-unit score jitter: varies tie-breaks across seats and games."""
    return bb.rng.random() * 0.3


class SetupSettlement(Node):
    """Best legal starting spot: production pips plus a new-resource bonus."""

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        rows = pov.legal_rows(ActionType.SETUP_SETTLEMENT)
        if rows.size == 0:
            return None
        best, best_score = int(rows[0]), -np.inf
        for row in rows:
            v = int(ROW_IDX[row])
            prod = pov.vertex_production(v)
            new_types = int(((prod > 0) & (pov.my_production == 0)).sum())
            score = float(prod.sum()) + 1.5 * new_types + _noise(bb)
            if score > best_score:
                best, best_score = int(row), score
        bb.last_setup_vertex = int(ROW_IDX[best])
        return best


class SetupRoad(Node):
    """Point the free road toward the most promising next settlement spot."""

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        rows = pov.legal_rows(ActionType.SETUP_ROAD)
        if rows.size == 0:
            return None
        best, best_score = int(rows[0]), -np.inf
        for row in rows:
            e = int(ROW_IDX[row])
            a, b = int(EDGE_ENDPOINTS[e, 0]), int(EDGE_ENDPOINTS[e, 1])
            far = b if a == bb.last_setup_vertex else a
            potential = max(
                (
                    float(pov.vertex_production(n).sum())
                    for n in VERTEX_NEIGHBORS[far]
                    if pov.settleable(n)
                ),
                default=0.0,
            )
            score = potential + 0.3 * float(pov.vertex_production(far).sum())
            score += _noise(bb)
            if score > best_score:
                best, best_score = int(row), score
        return best


class DiscardSurplus(Node):
    """Give up the card the current plan can spare most."""

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        rows = pov.legal_rows(ActionType.DISCARD)
        if rows.size == 0:
            return None
        reserved = bb.plan.reserved(pov) if bb.plan else np.zeros(5, dtype=np.int64)
        surplus = pov.hand - reserved
        return int(max(rows, key=lambda r: (surplus[ROW_IDX[r]], pov.hand[ROW_IDX[r]])))


def _robber_pick(pov: Pov, bb: Blackboard, rows: np.ndarray) -> int:
    """Best (tile, victim) among legal robber rows: block the most opponent
    production (weighted toward the VP leader), avoid our own tiles, prefer
    rows that steal — and among those, the fattest hand."""
    best, best_score = int(rows[0]), -np.inf
    for row in rows:
        t, victim = int(ROW_IDX[row]), int(ROW_TARGET[row])
        pips = float(pov.tile_pips[t])
        score = 0.0
        for c in TILE_CORNERS[t]:
            owner = int(pov.vertex_owner[c])
            if owner == 0:
                continue
            weight = pips * int(pov.vertex_type[c])
            if owner == pov.me + 1:
                score -= 1.5 * weight
            else:
                score += weight * (1.0 + 0.15 * float(pov.victory_points[owner - 1]))
        if victim >= 0:
            score += 2.0 + 0.2 * float(pov.hand_size[victim])
        score += _noise(bb)
        if score > best_score:
            best, best_score = int(row), score
    return best


class MoveRobber(Node):
    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        rows = pov.legal_rows(ActionType.MOVE_ROBBER)
        return _robber_pick(pov, bb, rows) if rows.size else None


class PlayKnight(Node):
    """Unblock our own production: play a knight whenever the robber sits on
    a tile we build on (legal pre-roll too, rulebook p.7)."""

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        rows = pov.legal_rows(ActionType.PLAY_KNIGHT)
        if rows.size == 0 or pov.tile_pips[pov.robber] == 0:
            return None
        blocked = any(
            int(pov.vertex_owner[c]) == pov.me + 1 for c in TILE_CORNERS[pov.robber]
        )
        return _robber_pick(pov, bb, rows) if blocked else None


class RespondToTrade(Node):
    """Accept exactly the offers that advance the plan: every card we pay is
    plan-surplus and at least one card we get is plan-needed."""

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        accept = pov.legal(ActionType.ACCEPT_TRADE)
        reject = pov.legal(ActionType.REJECT_TRADE)
        if accept is None and reject is None:
            return None
        get, pay = pov.trade_give, pov.trade_receive
        if accept is not None and bb.plan is not None:
            reserved = bb.plan.reserved(pov)
            need = bb.plan.need(pov)
            if bool(np.all(pay <= pov.hand - reserved)) and int(need @ get) >= 1:
                return accept
        return reject if reject is not None else accept


class RollDice(Node):
    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        return pov.legal(ActionType.ROLL_DICE)


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


def choose_plan(
    pov: Pov, bb: Blackboard, depth: int, exclude: str | None
) -> Plan | None:
    """Score every candidate goal and adopt the best.

    Candidates: a city on each own settlement, a settlement on each spot
    reachable within ``depth`` new roads (paths from BFS), a dev-card buy,
    and a single road extending the network. ``exclude`` skips the plan just
    abandoned for staleness so a starved goal cannot be re-adopted forever.
    """
    cands: list[tuple[float, Plan]] = []
    for v in pov.my_settlements:
        prod = float(pov.vertex_production(int(v)).sum())
        cands.append(
            (
                8.0 + prod + _noise(bb),
                Plan(f"city@{int(v)}", [Step(ActionType.BUILD_CITY, int(v))]),
            )
        )
    for v, path in pov.expansion_paths(depth):
        gain = pov.vertex_production(v)
        new_types = int(((gain > 0) & (pov.my_production == 0)).sum())
        score = 6.0 + float(gain.sum()) + 1.5 * new_types - 2.5 * len(path)
        steps = [Step(ActionType.BUILD_ROAD, e) for e in path]
        steps.append(Step(ActionType.BUILD_SETTLEMENT, v))
        cands.append((score + _noise(bb), Plan(f"settle@{v}", steps)))
    cands.append(
        (
            3.0 + _noise(bb),
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
            (1.0 + _noise(bb), Plan(f"road@{ext}", [Step(ActionType.BUILD_ROAD, ext)]))
        )
    cands = [c for c in cands if c[1].name != exclude]
    if not cands:
        return None
    return max(cands, key=lambda c: c[0])[1]


class ExecutePlan(Node):
    """Keep a live plan and play its next step the moment it is legal."""

    def __init__(self, depth: int) -> None:
        self.depth = depth

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        if not pov.my_turn_main:
            return None
        stale = bb.plan is not None and bb.plan_age > _PLAN_PATIENCE
        if (
            bb.plan is None
            or stale
            or bb.plan.invalid(pov)
            or bb.plan.next_step(pov) is None
        ):
            exclude = bb.plan.name if stale and bb.plan is not None else None
            bb.set_plan(choose_plan(pov, bb, self.depth, exclude))
        if bb.plan is None:
            return None
        step = bb.plan.next_step(pov)
        if step is None:
            return None
        if step.kind == ActionType.BUY_DEVELOPMENT_CARD:
            return pov.legal(ActionType.BUY_DEVELOPMENT_CARD)
        return pov.legal(step.kind, step.idx)


class Acquire(Node):
    """Close the plan's resource gap: dev-card plays, then the bank, then a
    domestic proposal (capped per turn, never re-offering a rejected one)."""

    def __init__(self, max_proposals: int) -> None:
        self.max_proposals = max_proposals

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        if not pov.my_turn_main or bb.plan is None:
            return None
        need = bb.plan.need(pov)
        if need.sum() == 0:
            return None
        surplus = pov.hand - bb.plan.reserved(pov)
        by_need = [int(r) for r in np.argsort(-need) if need[r] > 0]

        rows = pov.legal_rows(ActionType.PLAY_YEAR_OF_PLENTY)
        if rows.size:
            best = max(rows, key=lambda r: int(need[ROW_IDX[r]] + need[ROW_TARGET[r]]))
            if need[ROW_IDX[best]] + need[ROW_TARGET[best]] >= 1:
                return int(best)

        rows = pov.legal_rows(ActionType.PLAY_MONOPOLY)
        if rows.size:
            opp = (19 - pov.bank) - pov.hand  # cards of each type in other hands
            best = max(
                rows, key=lambda r: int(opp[ROW_IDX[r]] * (need[ROW_IDX[r]] > 0))
            )
            r = int(ROW_IDX[best])
            if need[r] > 0 and opp[r] >= 2:
                return int(best)

        road_steps = sum(
            1
            for s in bb.plan.steps
            if s.kind == ActionType.BUILD_ROAD and not s.realized(pov)
        )
        if road_steps >= 2:
            row = pov.legal(ActionType.PLAY_ROAD_BUILDING)
            if row is not None:
                return row

        best_row, best_score = None, -np.inf
        for g in range(N_RESOURCES):
            if need[g] > 0 or surplus[g] < pov.port_ratio[g]:
                continue
            for r in by_need:
                row = pov.legal(ActionType.MARITIME_TRADE, g, r)
                if row is None:
                    continue
                score = 10.0 * float(need[r]) + float(surplus[g])
                if score > best_score:
                    best_row, best_score = row, score
        if best_row is not None:
            return best_row

        if (
            pov.n_players > 2
            and bb.proposals_this_turn < self.max_proposals
            and bb.pending_proposal is None
        ):
            partners = sorted(
                (p for p in range(pov.n_players) if p != pov.me),
                key=lambda p: -int(pov.hand_size[p]),
            )
            for r in by_need:
                for g in range(N_RESOURCES):
                    if need[g] > 0 or surplus[g] < 1:
                        continue
                    for partner in partners:
                        if (g, r, partner) in bb.rejected:
                            continue
                        idx, target = pack_trade_single(g, r, partner)
                        row = flat_row(ActionType.PROPOSE_TRADE, int(idx), int(target))
                        if not pov.mask[row]:
                            continue
                        bb.pending_proposal = (g, r, partner, pov.hand.copy())
                        bb.proposals_this_turn += 1
                        return row
        return None


class EndTurn(Node):
    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        return pov.legal(ActionType.END_TURN)


# Fixed action-type priority for states the tree declines on; PROPOSE_TRADE
# is deliberately absent (an unmanaged proposal could re-offer forever).
_FALLBACK_ORDER = (
    ActionType.ROLL_DICE,
    ActionType.SETUP_SETTLEMENT,
    ActionType.SETUP_ROAD,
    ActionType.DISCARD,
    ActionType.MOVE_ROBBER,
    ActionType.REJECT_TRADE,
    ActionType.ACCEPT_TRADE,
    ActionType.BUILD_CITY,
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_ROAD,
    ActionType.BUY_DEVELOPMENT_CARD,
    ActionType.END_TURN,
    ActionType.MARITIME_TRADE,
    ActionType.PLAY_KNIGHT,
    ActionType.PLAY_ROAD_BUILDING,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_MONOPOLY,
)


def _fallback(pov: Pov) -> int:
    for atype in _FALLBACK_ORDER:
        rows = pov.legal_rows(atype)
        if rows.size:
            return int(rows[0])
    return 0


class PlannerAgent:
    """One seat's stateful decision tree (a :class:`GameAgent`)."""

    def __init__(
        self,
        seed: int,
        *,
        expansion_depth: int = 3,
        max_proposals_per_turn: int = 2,
    ) -> None:
        self._bb = Blackboard(rng=random.Random(seed))
        self._was_my_roll = False
        self._root = Selector(
            SetupSettlement(),
            SetupRoad(),
            DiscardSurplus(),
            MoveRobber(),
            RespondToTrade(),
            PlayKnight(),
            RollDice(),
            ExecutePlan(expansion_depth),
            Acquire(max_proposals_per_turn),
            EndTurn(),
        )

    def _sync(self, pov: Pov) -> None:
        """Per-turn memory upkeep: reset on our new turn, and mark the last
        proposal rejected if our hand shows no trace of it."""
        my_roll = pov.phase == GamePhase.ROLL and pov.current_player == pov.me
        if my_roll and not self._was_my_roll:
            self._bb.begin_turn()
        self._was_my_roll = my_roll
        if self._bb.pending_proposal is not None and pov.my_turn_main:
            g, r, partner, before = self._bb.pending_proposal
            if pov.hand[r] <= before[r]:
                self._bb.rejected.add((g, r, partner))
            self._bb.pending_proposal = None

    def act(self, obs: HostObservation, mask: HostFlatMask) -> int:
        pov = Pov(obs, mask)
        if not pov.mask.any():
            return 0
        self._sync(pov)
        row = self._root.tick(pov, self._bb)
        if row is None or not pov.mask[row]:
            row = _fallback(pov)
        return int(row)


def make_planner(
    *, expansion_depth: int = 3, max_proposals_per_turn: int = 2
) -> StatefulPolicy:
    """The planner family: ``expansion_depth`` bounds how many new roads a
    settlement plan may path through; ``max_proposals_per_turn`` caps domestic
    offers. Returns the per-game agent factory."""

    def build(seed: int) -> GameAgent:
        return PlannerAgent(
            seed,
            expansion_depth=expansion_depth,
            max_proposals_per_turn=max_proposals_per_turn,
        )

    return build
