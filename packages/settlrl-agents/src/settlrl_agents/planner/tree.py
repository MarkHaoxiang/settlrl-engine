"""The stateful decision-tree toolkit: nodes, the blackboard, and plans.

A tree ticks once per decision: each :class:`Node` either returns a flat
action row or declines with None, and a :class:`Selector` takes the first
child that acts. State lives on the :class:`Blackboard` the agent threads
through every tick — most importantly the current :class:`Plan`, a build
goal that persists across turns (so the agent saves toward it instead of
re-deciding from scratch every move).

Plan steps are declarative targets, not queued actions: each tick the plan
reports the first step whose effect is not yet on the board, and the agent
re-checks the whole plan against the observation — a step that became
impossible (spot taken, path cut) invalidates the plan and triggers a replan.
That re-validation is what makes carrying state across an auto-resetting,
adversarial game safe.
"""

from __future__ import annotations

import abc
import dataclasses
import random

import numpy as np
from settlrl_engine.board.resources import N_RESOURCES
from settlrl_engine.board.state import CITY
from settlrl_engine.env import ActionType

from settlrl_agents.planner.pov import (
    COST_CITY,
    COST_DEV,
    COST_ROAD,
    COST_SETTLEMENT,
    Pov,
)


@dataclasses.dataclass
class Step:
    """One declarative plan target: an action type plus its primary index."""

    kind: ActionType
    idx: int = 0

    @property
    def cost(self) -> np.ndarray:
        return {
            ActionType.BUILD_ROAD: COST_ROAD,
            ActionType.BUILD_SETTLEMENT: COST_SETTLEMENT,
            ActionType.BUILD_CITY: COST_CITY,
            ActionType.BUY_DEVELOPMENT_CARD: COST_DEV,
        }[self.kind]

    def realized(self, pov: Pov) -> bool:
        """The step's effect is already on the board."""
        if self.kind == ActionType.BUILD_ROAD:
            return int(pov.edge_road[self.idx]) == pov.me + 1
        if self.kind == ActionType.BUILD_SETTLEMENT:
            return int(pov.vertex_owner[self.idx]) == pov.me + 1
        if self.kind == ActionType.BUILD_CITY:
            return (
                int(pov.vertex_owner[self.idx]) == pov.me + 1
                and int(pov.vertex_type[self.idx]) == CITY
            )
        # BUY_DEVELOPMENT_CARD: realized once our public dev count passed the
        # count recorded when the plan was made (idx carries that baseline).
        return int(pov.dev_card_count[pov.me]) > self.idx

    def impossible(self, pov: Pov) -> bool:
        """The step can never be realized from here (triggers a replan)."""
        if self.kind == ActionType.BUILD_ROAD:
            return int(pov.edge_road[self.idx]) not in (0, pov.me + 1)
        if self.kind == ActionType.BUILD_SETTLEMENT:
            owner = int(pov.vertex_owner[self.idx])
            if owner not in (0, pov.me + 1):
                return True
            return owner == 0 and not pov.settleable(self.idx)
        if self.kind == ActionType.BUILD_CITY:
            return int(pov.vertex_owner[self.idx]) not in (0, pov.me + 1)
        return False


@dataclasses.dataclass
class Plan:
    """A persistent build goal: ordered steps toward one scored target."""

    name: str
    steps: list[Step]

    def next_step(self, pov: Pov) -> Step | None:
        """The first step not yet on the board (None when complete)."""
        for step in self.steps:
            if not step.realized(pov):
                return step
        return None

    def invalid(self, pov: Pov) -> bool:
        return any(s.impossible(pov) for s in self.steps if not s.realized(pov))

    def need(self, pov: Pov) -> np.ndarray:
        """Resources still missing for the remaining steps (per type)."""
        remaining = sum(
            (s.cost for s in self.steps if not s.realized(pov)),
            np.zeros(N_RESOURCES, dtype=np.int64),
        )
        return np.asarray(np.clip(remaining - pov.hand, 0, None))

    def reserved(self, pov: Pov) -> np.ndarray:
        """The part of our hand earmarked for the remaining steps."""
        remaining = sum(
            (s.cost for s in self.steps if not s.realized(pov)),
            np.zeros(N_RESOURCES, dtype=np.int64),
        )
        return np.asarray(np.minimum(remaining, pov.hand))


@dataclasses.dataclass
class Blackboard:
    """The agent's memory across ticks (one game, one seat)."""

    rng: random.Random
    plan: Plan | None = None
    plan_age: int = 0
    """Own turns since the plan was adopted (staleness abandon, see agent)."""
    last_setup_vertex: int | None = None
    # Trade memory: the engine forgets rejected offers, so the agent must not
    # re-offer them forever. ``pending_proposal`` holds (give, get, partner,
    # hand-before); on the next tick in MAIN an unchanged hand means the offer
    # was rejected and it joins ``rejected`` for the rest of the turn.
    rejected: set[tuple[int, int, int]] = dataclasses.field(default_factory=set)
    pending_proposal: tuple[int, int, int, np.ndarray] | None = None
    proposals_this_turn: int = 0
    forced_row: int | None = None
    """A committed follow-up (the second half of an own-turn combo): played
    the next tick it is legal, dropped the moment it is not."""

    def noise(self) -> float:
        """Sub-unit score jitter: varies tie-breaks across seats and games."""
        return self.rng.random() * 0.3

    def set_plan(self, plan: Plan | None) -> None:
        self.plan = plan
        self.plan_age = 0

    def begin_turn(self) -> None:
        self.rejected.clear()
        self.pending_proposal = None
        self.proposals_this_turn = 0
        self.forced_row = None
        self.plan_age += 1


class Node(abc.ABC):
    """One decision point: act (a flat row) or decline (None)."""

    @abc.abstractmethod
    def tick(self, pov: Pov, bb: Blackboard) -> int | None: ...


class Selector(Node):
    """The first child that acts wins; declines only if every child does."""

    def __init__(self, *children: Node) -> None:
        self.children = children

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        for child in self.children:
            row = child.tick(pov, bb)
            if row is not None:
                return row
        return None
