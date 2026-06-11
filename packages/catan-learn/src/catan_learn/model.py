"""Plain-JAX MLPs and their adapters onto the catan-agents seams.

Parameters are an ordinary pytree (a tuple of ``(weights, bias)`` layers), so
a trained model needs nothing beyond jax to run — catan-agents can consume an
exported artifact without any training dependencies.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from catan_agents.shared.policy import PolicyPrior
from catan_agents.shared.value import Value, ValueFunction
from catan_engine.board.layout import BoardLayout
from catan_engine.board.state import BoardState, IntScalar, KeyScalar
from catan_engine.env import N_FLAT
from jaxtyping import Array, Float

from catan_learn.features import FEATURE_DIM, features

MLPParams = tuple[tuple[Float[Array, "fan_out fan_in"], Float[Array, "fan_out"]], ...]
"""One ``(weights, bias)`` pair per layer, input to output."""


def init_mlp(key: KeyScalar, sizes: Sequence[int], scale: float = 1.0) -> MLPParams:
    """He-initialised layers of ``sizes`` (input dim first, output dim last);
    ``scale`` multiplies the output layer (small values start near-uniform /
    near-zero)."""
    layers = []
    for i, (fan_in, fan_out) in enumerate(itertools.pairwise(sizes)):
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (fan_out, fan_in)) * jnp.sqrt(2.0 / fan_in)
        if i == len(sizes) - 2:
            w = w * scale
        layers.append((w, jnp.zeros((fan_out,))))
    return tuple(layers)


def mlp(params: MLPParams, x: Float[Array, "fan_in"]) -> Float[Array, "fan_out"]:
    """Forward pass: ReLU hidden layers, linear output."""
    for w, b in params[:-1]:
        x = jax.nn.relu(w @ x + b)
    w, b = params[-1]
    return w @ x + b


def make_net_value(params: MLPParams) -> ValueFunction:
    """A :class:`ValueFunction` over :func:`features` (scalar head)."""

    def value(layout: BoardLayout, state: BoardState, player: IntScalar) -> Value:
        return mlp(params, features(layout, state, player))[0]

    return value


def make_net_prior(params: MLPParams) -> PolicyPrior:
    """A :class:`PolicyPrior` over :func:`features` (``N_FLAT`` logits head;
    consumers apply legality masking)."""

    def prior(
        layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> Float[Array, f"flat={N_FLAT}"]:
        return mlp(params, features(layout, state, player))

    return prior


def init_value_params(key: KeyScalar, hidden: Sequence[int] = (64, 64)) -> MLPParams:
    """Stand-in value model: ``features -> hidden -> 1``."""
    return init_mlp(key, (FEATURE_DIM, *hidden, 1), scale=0.01)


def init_prior_params(key: KeyScalar, hidden: Sequence[int] = (64,)) -> MLPParams:
    """Stand-in policy model: ``features -> hidden -> N_FLAT`` logits."""
    return init_mlp(key, (FEATURE_DIM, *hidden, N_FLAT), scale=0.01)


def save_params(path: str | Path, params: MLPParams) -> None:
    """Write ``params`` as an ``.npz`` artifact (load with :func:`load_params`)."""
    arrays = {}
    for i, (w, b) in enumerate(params):
        arrays[f"w{i}"] = np.asarray(w)
        arrays[f"b{i}"] = np.asarray(b)
    np.savez(path, **arrays)  # type: ignore[arg-type]  # typeshed kwargs quirk


def load_params(path: str | Path) -> MLPParams:
    """Read an :func:`save_params` artifact back into device arrays."""
    with np.load(path) as doc:
        n = sum(1 for k in doc.files if k.startswith("w"))
        return tuple(
            (jnp.asarray(doc[f"w{i}"]), jnp.asarray(doc[f"b{i}"])) for i in range(n)
        )
