"""Fast structural contracts for the unified training package -- the backend
seams and the net-agnostic self-play, exercised without the search (a uniform
policy stands in for `weights_fn`, so these stay seconds-fast).

Expect tests: the inline snapshot is the contract; regenerate with
``EXPECTTEST_ACCEPT=1 pytest``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline
from jaxtyping import Array
from settlrl_engine.board import Board, make_board
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.env import N_FLAT
from settlrl_learn.nn.graphnet import PRESETS
from settlrl_learn.training import GNNBackend, MLPBackend, RunState
from settlrl_learn.training.backend import Backend, load_run_state, save_run_state
from settlrl_learn.training.selfplay import self_play


def _shapes(tree: object) -> str:
    """Trailing shapes of a pytree's array leaves, one per line (the leading
    sample/batch axis is run-dependent, so it is dropped)."""
    leaves = jax.tree.leaves(tree)
    return "\n".join(str(tuple(np.asarray(x).shape)) for x in leaves)


def _single(n_players: int = 2, seed: int = 0) -> Board:
    layout, state = make_board(batch_size=1, seed=seed, n_players=n_players)
    return jax.tree.map(lambda x: x[0], layout), jax.tree.map(lambda x: x[0], state)


def test_mlp_backend_item_and_observe_shapes() -> None:
    backend = MLPBackend((16,))
    layout, state = _single()
    obs = backend.observe(layout, state, jnp.int32(0))
    assert_expected_inline(
        f"keys={sorted(obs)}\nempty_item:\n{_shapes(backend.empty_item())}",
        """\
keys=['features']
empty_item:
(118,)
(662,)
()""",
    )


def test_gnn_backend_item_and_observe_shapes() -> None:
    backend = GNNBackend(
        PRESETS["gn_global"]._replace(width=16, layers=2, head_depth=1)
    )
    layout, state = _single()
    obs = backend.observe(layout, state, jnp.int32(0))
    assert_expected_inline(
        f"keys={sorted(obs)}\nempty_item:\n{_shapes(backend.empty_item())}",
        """\
keys=['edges', 'glob', 'nodes']
empty_item:
(54, 17)
(144, 3)
(40,)
(662,)
(662,)
()""",
    )


def _uniform_weights(
    key: Array, layout: BoardLayout, view: Any, player: Array, mask: Array
) -> Array:
    """A stand-in for the search: uniform over the legal set (no net, no tree)."""
    return mask.astype(jnp.float32)


def test_self_play_samples_shape_under_uniform_policy() -> None:
    # Drives the real generic self-play (env stepping, pending flush, outcome
    # credit) with the MLP observation but a trivial policy -- fast, no search.
    backend = MLPBackend((16,))
    samples = self_play(
        _uniform_weights, backend.observe,
        n_samples=8, batch_size=4, seed=0,
    )  # fmt: skip
    n = samples["value"].shape[0]
    assert n >= 8 and all(v.shape[0] == n for v in samples.values())
    trailing = {k: tuple(v.shape[1:]) for k, v in sorted(samples.items())}
    assert_expected_inline(
        str(trailing),
        "{'features': (118,), 'mask': (662,), 'policy': (662,), 'value': ()}",
    )
    # the env mask is binary; the policy target is recorded over the legal set.
    assert set(np.unique(samples["mask"])).issubset({0.0, 1.0})
    assert samples["policy"].shape[1] == N_FLAT


def test_runstate_serialise_roundtrip_is_bit_exact(tmp_path: Path) -> None:
    # The resume invariant at the serialization layer (no training): a fresh
    # RunState round-trips bit-exactly through eqx for both backends.
    import optax

    backends: list[tuple[str, Backend]] = [
        ("mlp", MLPBackend((16,))),
        ("gnn", GNNBackend(PRESETS["gn_global"]._replace(width=16, layers=2))),
    ]
    for name, backend in backends:
        net = backend.init(jax.random.key(0))
        opt = optax.adamw(1e-3)
        state = RunState(
            net, backend.init_opt(opt, net), {}, jnp.int32(3), jnp.float32(0.4)
        )
        path = tmp_path / f"{name}.eqx"
        save_run_state(path, state)
        back = load_run_state(path, state)
        a, b = jax.tree.leaves(state.net), jax.tree.leaves(back.net)
        assert all(
            np.array_equal(np.asarray(x), np.asarray(y))
            for x, y in zip(a, b, strict=True)
        )
        assert int(back.iteration) == 3 and float(back.best) == float(jnp.float32(0.4))
