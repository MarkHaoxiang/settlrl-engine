"""Bot seats: the catan-agents registry adapted to the renderer's single game.

``POLICIES`` is catan-agents' registry (name -> ``AgentSpec``); a spec's class
declares which protocol its policy speaks (``ObservationSpec`` /
``BeliefSpec``) and the seat counts it supports. A spec is a policy *family*:
its scalar build parameters (int / float / bool keyword arguments of
``spec.make``) are exposed as per-seat knobs — :func:`bot_catalog` describes
them to the client and :func:`bot_act` builds (and caches) the configured
policy. :func:`bot_act` hides the protocol dispatch: it slices the single
game out of the renderer's batch-of-one env and calls the policy through the
right protocol (belief seats read the env's honest ``belief_view``, so the
env must be built with ``track_beliefs=True`` when any are seated).
"""

import inspect
from collections.abc import Mapping
from typing import cast

import jax
import jax.numpy as jnp
from catan_agents import POLICIES, AgentSpec, BeliefSpec, ObservationSpec
from catan_agents.policy import BeliefPolicy, Policy, StatefulPolicy
from catan_engine.board.state import KeyScalar
from catan_engine.env import BatchedCatanEnv, Observation

__all__ = [
    "POLICIES",
    "AgentSpec",
    "BeliefSpec",
    "bot_act",
    "bot_catalog",
    "coerce_params",
]

Knob = int | float | bool
"""A configurable scalar build parameter of a bot family."""

# Configured policies, jitted once per (kind, params) across sessions.
_ACTS: dict[tuple[str, tuple[tuple[str, Knob], ...]], Policy | BeliefPolicy] = {}


def _knobs(
    spec: AgentSpec[Policy] | AgentSpec[BeliefPolicy] | AgentSpec[StatefulPolicy],
) -> dict[str, Knob]:
    """The family's scalar build parameters and their effective defaults."""
    out: dict[str, Knob] = {}
    for name, param in inspect.signature(spec.make).parameters.items():
        default = spec.defaults.get(name, param.default)
        if isinstance(default, bool | int | float):
            out[name] = default
    return out


def bot_catalog() -> dict[str, dict[str, object]]:
    """Every bot kind with its supported seat counts and configurable knobs.

    Shape: ``{kind: {"counts": [2, ...], "params": {name: {"type": "int" |
    "float" | "bool", "default": value}}}}``.
    """
    catalog: dict[str, dict[str, object]] = {}
    for name, spec in POLICIES.items():
        # Stateful families need one live agent per (session, seat) — a seam
        # the per-move bot_act doesn't have — so they are not offered.
        if not isinstance(spec, ObservationSpec | BeliefSpec):
            continue
        params = {
            k: {
                "type": "bool"
                if isinstance(v, bool)
                else "int"
                if isinstance(v, int)
                else "float",
                "default": v,
            }
            for k, v in _knobs(spec).items()
        }
        catalog[name] = {"counts": sorted(spec.n_players), "params": params}
    return catalog


def coerce_params(kind: str, params: Mapping[str, object]) -> dict[str, Knob]:
    """Validate seat ``params`` against ``kind``'s knobs; returns typed values.

    Raises ``ValueError`` on an unknown knob or a value of the wrong shape.
    """
    knobs = _knobs(POLICIES[kind])
    unknown = sorted(set(params) - set(knobs))
    if unknown:
        raise ValueError(f"unknown {kind} parameter(s): {', '.join(unknown)}")
    out: dict[str, Knob] = {}
    for name, value in params.items():
        default = knobs[name]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{kind}.{name} expects a bool, got {value!r}")
            out[name] = value
        elif not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{kind}.{name} expects a number, got {value!r}")
        elif isinstance(default, int):
            out[name] = int(value)
        else:
            out[name] = float(value)
    return out


def _policy(kind: str, params: Mapping[str, Knob]) -> Policy | BeliefPolicy:
    key = (kind, tuple(sorted(params.items())))
    if key not in _ACTS:
        spec = POLICIES[kind]
        if not isinstance(spec, ObservationSpec | BeliefSpec):
            raise ValueError(f"bot kind {kind!r} is not seatable (stateful family)")
        built = spec.make(**{**spec.defaults, **params}) if params else spec.policy
        _ACTS[key] = jax.jit(built)
    return _ACTS[key]


def bot_act(
    kind: str,
    params: Mapping[str, Knob],
    key: KeyScalar,
    benv: BatchedCatanEnv,
    seat: int,
) -> int:
    """One move for ``seat`` from bot ``kind`` built at ``params`` (validated
    knob overrides; empty for the family's defaults)."""
    act = _policy(kind, params)
    mask = benv.flat_mask()[0]
    if isinstance(POLICIES[kind], ObservationSpec):
        obs = cast(Observation, jax.tree.map(lambda x: x[0], benv.observe(seat)))
        return int(cast(Policy, act)(key, obs, mask))
    layout = jax.tree.map(lambda x: x[0], benv.board[0])
    view = jax.tree.map(lambda x: x[0], benv.belief_view(seat))
    return int(cast(BeliefPolicy, act)(key, layout, view, jnp.int32(seat), mask))
