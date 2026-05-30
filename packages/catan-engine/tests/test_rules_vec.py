"""Equivalence tests: the traceable rules_vec helpers must match the trusted
NumPy single-game reference in catan_engine.rules across randomized boards."""

from typing import TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from expecttest import TestCase

from catan_engine import rules_vec
from tests import reference as rules
from catan_engine.layout import BoardLayout, N_EDGES, N_TILES, N_VERTICES, make_layout
from catan_engine.resources import N_PLAYERS, N_RESOURCES
from catan_engine.state import BoardState, make_board_state

_T = TypeVar("_T")


def _single(tree: _T) -> _T:
    return cast(_T, jax.tree_util.tree_map(lambda x: x[0], tree))


# Compile the single-game DFS once; reused across calls (static shapes).
_LRL = jax.jit(rules_vec.longest_road_length)


def _random_occupancy(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random (edge_road, vertex_owner): values 0=empty, 1..4 = player+1."""
    rng = np.random.default_rng(seed)
    # Bias towards empty so road networks stay realistically small.
    edge_road = rng.choice(
        [0, 1, 2, 3, 4], size=N_EDGES, p=[0.55, 0.15, 0.12, 0.1, 0.08]
    ).astype(np.uint8)
    vertex_owner = rng.choice(
        [0, 1, 2, 3, 4], size=N_VERTICES, p=[0.7, 0.1, 0.08, 0.07, 0.05]
    ).astype(np.uint8)
    return edge_road, vertex_owner


def _state_with(edge_road: np.ndarray, vertex_owner: np.ndarray) -> BoardState:
    state = make_board_state(1, key=jax.random.key(0))
    return state._replace(
        edge_road=jnp.asarray(edge_road)[None],
        vertex_owner=jnp.asarray(vertex_owner)[None],
    )


class TestLongestRoad(TestCase):
    def test_empty_board_is_zero(self) -> None:
        edge_road = np.zeros(N_EDGES, np.uint8)
        owner = np.zeros(N_VERTICES, np.uint8)
        for p in range(N_PLAYERS):
            assert int(_LRL(jnp.asarray(edge_road), jnp.asarray(owner), jnp.int32(p))) == 0

    def test_single_road_is_one(self) -> None:
        edge_road = np.zeros(N_EDGES, np.uint8)
        edge_road[7] = 1  # one road for player 0
        owner = np.zeros(N_VERTICES, np.uint8)
        assert int(_LRL(jnp.asarray(edge_road), jnp.asarray(owner), jnp.int32(0))) == 1
        assert int(_LRL(jnp.asarray(edge_road), jnp.asarray(owner), jnp.int32(1))) == 0

    def test_matches_numpy_reference(self) -> None:
        for seed in range(80):
            edge_road, vertex_owner = _random_occupancy(seed)
            state = _state_with(edge_road, vertex_owner)
            for p in range(N_PLAYERS):
                ref = rules.longest_road_length(state, p, 0)
                got = int(
                    _LRL(jnp.asarray(edge_road), jnp.asarray(vertex_owner), jnp.int32(p))
                )
                assert got == ref, (
                    f"seed={seed} player={p}: vec={got} ref={ref}"
                )


class TestProductionAndPorts(TestCase):
    def _random_state(self, seed: int) -> tuple[BoardLayout, BoardState]:
        rng = np.random.default_rng(seed)
        layout = make_layout(1, key=jax.random.key(seed))
        state = make_board_state(1, key=jax.random.key(seed))
        owner = rng.choice(
            [0, 1, 2, 3, 4], size=N_VERTICES, p=[0.6, 0.12, 0.11, 0.1, 0.07]
        ).astype(np.uint8)
        vtype = np.where(
            owner > 0, rng.integers(1, 3, size=N_VERTICES), 0
        ).astype(np.uint8)
        # Keep hands small so the bank never goes negative.
        pr = rng.integers(0, 3, size=(N_PLAYERS, N_RESOURCES)).astype(np.uint8)
        robber = np.uint8(rng.integers(0, N_TILES))
        state = state._replace(
            vertex_owner=jnp.asarray(owner)[None],
            vertex_type=jnp.asarray(vtype)[None],
            player_resources=jnp.asarray(pr)[None],
            robber=jnp.asarray([robber]),
        )
        return layout, state

    def test_distribute_matches_reference(self) -> None:
        for seed in range(60):
            layout, state = self._random_state(seed)
            layout1, state1 = _single(layout), _single(state)
            for roll in range(2, 13):
                ref = rules.distribute_resources(layout, state, roll, 0)
                got = rules_vec.distribute_resources(layout1, state1, jnp.int32(roll))
                assert np.array_equal(
                    np.asarray(got.player_resources),
                    np.asarray(ref.player_resources[0]),
                ), f"seed={seed} roll={roll}"

    def test_port_ratio_matches_reference(self) -> None:
        for seed in range(40):
            layout, state = self._random_state(seed)
            layout1, state1 = _single(layout), _single(state)
            for p in range(N_PLAYERS):
                for give in range(N_RESOURCES):
                    ref = rules.port_ratio(state, layout, p, give, 0)
                    got = int(
                        rules_vec.port_ratio(
                            state1.vertex_owner,
                            layout1.port_allocation,
                            jnp.int32(p),
                            jnp.int32(give),
                        )
                    )
                    assert got == ref, f"seed={seed} p={p} give={give}"


class TestEconomyHelpers(TestCase):
    def test_can_afford(self) -> None:
        assert bool(
            rules_vec.can_afford(
                jnp.array([1, 1, 1, 1, 0], jnp.uint8), rules_vec.SETTLEMENT_COST_ARR
            )
        )
        assert not bool(
            rules_vec.can_afford(
                jnp.array([0, 1, 1, 1, 0], jnp.uint8), rules_vec.SETTLEMENT_COST_ARR
            )
        )

    def test_pay_clips_at_zero(self) -> None:
        pr = jnp.zeros((N_PLAYERS, N_RESOURCES), jnp.uint8)
        out = rules_vec.pay(pr, jnp.int32(0), rules_vec.ROAD_COST_ARR)
        assert np.array_equal(np.asarray(out[0]), np.zeros(N_RESOURCES, np.uint8))
