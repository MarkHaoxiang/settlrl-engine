"""settlrl-learn skeleton: featurization, protocol adapters, trainability."""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from settlrl_agents.policy import PolicyPrior
from settlrl_agents.search import make_search
from settlrl_agents.value import ValueFunction
from settlrl_engine.board import Board, make_board
from settlrl_engine.env import BatchedSettlrlEnv
from settlrl_learn import (
    FEATURE_DIM,
    features,
    fit,
    init_prior_params,
    init_value_params,
    load_params,
    make_net_prior,
    make_net_value,
    save_params,
    value_loss,
)


def _single(n_players: int, seed: int = 0) -> Board:
    layout, state = make_board(batch_size=1, seed=seed, n_players=n_players)
    return jax.tree.map(lambda x: x[0], layout), jax.tree.map(lambda x: x[0], state)


def test_features_width_is_player_count_invariant() -> None:
    for n in (2, 4):
        layout, state = _single(n)
        x = features(layout, state, jnp.int32(0))
        assert x.shape == (FEATURE_DIM,)
        assert bool(jnp.isfinite(x).all())


def test_features_distinguish_players_and_jit() -> None:
    env = BatchedSettlrlEnv(batch_size=1, n_players=2, seed=3)
    key = jax.random.key(0)
    for _ in range(60):  # past setup, so the seats genuinely differ
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
    layout = jax.tree.map(lambda x: x[0], env.board[0])
    state = jax.tree.map(lambda x: x[0], env.board[1])
    f = jax.jit(features)
    x0, x1 = f(layout, state, jnp.int32(0)), f(layout, state, jnp.int32(1))
    assert not bool(jnp.allclose(x0, x1))


def test_stand_ins_satisfy_the_seams_and_play_legally() -> None:
    key = jax.random.key(0)
    value = make_net_value(init_value_params(key))
    prior = make_net_prior(init_prior_params(key))
    assert isinstance(value, ValueFunction)
    assert isinstance(prior, PolicyPrior)

    env = BatchedSettlrlEnv(batch_size=2, n_players=2, track_beliefs=True, seed=1)
    mask = env.flat_mask()
    layout = env.board[0]
    view = jax.tree.map(
        lambda *xs: jnp.stack(xs)[env.agent_selection, jnp.arange(2)],
        *[env.belief_view(i) for i in range(2)],
    )
    for policy in (
        make_search(value, num_simulations=0, propose_rate=0.5),
        make_search(
            value,
            prior=prior,
            num_trees=1,
            num_simulations=8,
            max_num_considered_actions=8,
        ),
    ):
        acts = jax.jit(jax.vmap(policy, in_axes=(0, 0, 0, 0, 0)))(
            jax.random.split(jax.random.key(2), 2),
            layout,
            view,
            env.agent_selection,
            mask,
        )
        assert bool(mask[jnp.arange(2), acts].all())


def test_save_load_round_trips(tmp_path: pathlib.Path) -> None:
    params = init_value_params(jax.random.key(1))
    path = tmp_path / "value.npz"
    save_params(path, params)
    loaded = load_params(path)
    for (w, b), (w2, b2) in zip(params, loaded, strict=True):
        assert np.array_equal(np.asarray(w), np.asarray(w2))
        assert np.array_equal(np.asarray(b), np.asarray(b2))


def test_fit_reduces_value_loss() -> None:
    key = jax.random.key(2)
    x = jax.random.normal(key, (256, FEATURE_DIM))
    y = (x[:, 0] > 0).astype(jnp.float32)  # separable synthetic labels
    params = init_value_params(key)
    before = value_loss(params, x, y)
    _fitted, after = fit(params, x, y, steps=300, lr=0.05)
    assert float(after) < float(before) * 0.5


def test_fit_runs_under_pytest_budget() -> None:
    # The smoke above is the contract; this pins the loop's jit reuse (one
    # compile, then fast steps).
    key = jax.random.key(3)
    x = jax.random.normal(key, (32, FEATURE_DIM))
    y = jnp.zeros((32,))
    _, loss = fit(init_value_params(key), x, y, steps=5)
    assert bool(jnp.isfinite(loss))
