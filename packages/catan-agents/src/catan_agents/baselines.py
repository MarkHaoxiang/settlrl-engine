"""Baseline policies."""

from __future__ import annotations

from catan_engine.board.state import KeyScalar
from catan_engine.env import Observation, random_flat

from catan_agents.policy import FlatAction, FlatMask


def random_policy(key: KeyScalar, obs: Observation, mask: FlatMask) -> FlatAction:
    """Random legal play (``obs`` is ignored): the engine's shared type-first
    sampler — a uniform legal action *type*, then a uniform legal move of it.
    """
    return random_flat(key, mask)
