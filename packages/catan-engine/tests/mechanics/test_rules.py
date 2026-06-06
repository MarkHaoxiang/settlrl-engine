"""Equivalence tests: the traceable rule modules (``longest_road`` / ``dice`` /
``trade`` and the ``common`` economy helpers) must match the trusted single-game
oracle (the ``catan-reference`` package, via ``tests.conversion``) across
randomized boards."""

import jax
import jax.numpy as jnp
import numpy as np
from expecttest import TestCase
from hypothesis import given, settings
from hypothesis import strategies as st

from catan_engine.mechanics import common, dice, longest_road, trade
from tests import conversion as reference
from tests.mechanics._occupancy import random_occupancy, single as _single
from catan_engine.board.layout import (
    BoardLayout,
    EDGE_V,
    N_EDGES,
    N_TILES,
    N_VERTICES,
    make_layout,
)
from catan_engine.board.resources import N_PLAYERS, N_RESOURCES
from catan_engine.board.state import MAX_ROADS, BoardState, make_board_state

# Bias towards empty so road networks stay realistically small.
_EDGE_P = [0.55, 0.15, 0.12, 0.1, 0.08]
_VERTEX_P = [0.7, 0.1, 0.08, 0.07, 0.05]


# Compile the single-game DFS once; reused across calls (static shapes).
_LRL = jax.jit(longest_road.longest_road_length)


def _random_occupancy(seed: int) -> tuple[np.ndarray, np.ndarray]:
    return random_occupancy(seed, edge_p=_EDGE_P, vertex_p=_VERTEX_P)


def _state_with(edge_road: np.ndarray, vertex_owner: np.ndarray) -> BoardState:
    state = make_board_state(1, key=jax.random.key(0))
    return state._replace(
        edge_road=jnp.asarray(edge_road)[None],
        vertex_owner=jnp.asarray(vertex_owner)[None],
    )


# --- Hypothesis strategy for longest-road fuzzing --------------------------
#
# Random per-edge occupancy (the seed-based test below) rarely forms long
# connected trails. To actually stress the longest-*trail* DFS we grow each
# player's roads as a *connected* subgraph by edge accretion on the real board
# adjacency -- naturally producing long trails, Y-branches and cycles -- and
# bias opponent buildings onto road vertices to exercise the "a trail may end on
# but not pass *through* an opponent" rule. Both implementations are exact, so
# the engine DFS must equal the reference for every generated board.

_EV = np.asarray(EDGE_V)  # (N_EDGES, 2) edge -> endpoint vertices
_ADJ: dict[int, list[tuple[int, int]]] = {v: [] for v in range(N_VERTICES)}
for _e in range(N_EDGES):
    _va, _vb = int(_EV[_e, 0]), int(_EV[_e, 1])
    _ADJ[_va].append((_e, _vb))
    _ADJ[_vb].append((_e, _va))


def _grow_connected(draw: st.DrawFn, free: set[int], target: int) -> set[int]:
    """A connected set of up to ``target`` edges drawn from ``free``."""
    if target == 0 or not free:
        return set()
    start = draw(st.sampled_from(sorted(free)))
    chosen = {start}
    verts = {int(_EV[start, 0]), int(_EV[start, 1])}
    while len(chosen) < target:
        frontier = sorted(
            {e for v in verts for (e, _w) in _ADJ[v] if e in free and e not in chosen}
        )
        if not frontier:
            break
        e = draw(st.sampled_from(frontier))
        chosen.add(e)
        verts.add(int(_EV[e, 0]))
        verts.add(int(_EV[e, 1]))
    return chosen


@st.composite
def _boards(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    """A random ``(edge_road, vertex_owner)`` in the real game domain
    (<= MAX_ROADS roads/player), mixing connected and uniform-sparse networks."""
    edge_road = np.zeros(N_EDGES, np.uint8)
    if draw(st.sampled_from(["connected", "uniform"])) == "connected":
        free = set(range(N_EDGES))
        for p in range(draw(st.integers(1, N_PLAYERS))):
            size = draw(st.integers(0, MAX_ROADS))
            for e in _grow_connected(draw, free, size):
                edge_road[e] = p + 1
                free.discard(e)
    else:
        for e in range(N_EDGES):
            edge_road[e] = draw(st.integers(0, N_PLAYERS))
        for p in range(1, N_PLAYERS + 1):  # keep within the <= MAX_ROADS domain
            owned = np.flatnonzero(edge_road == p)
            edge_road[owned[MAX_ROADS:]] = 0

    # Sparse buildings, half of them placed on road vertices to split trails.
    vertex_owner = np.zeros(N_VERTICES, np.uint8)
    road_vertices = sorted({int(v) for e in np.flatnonzero(edge_road) for v in _EV[e]})
    for _ in range(draw(st.integers(0, 12))):
        on_road = bool(road_vertices) and draw(st.booleans())
        if on_road:
            v = draw(st.sampled_from(road_vertices))
        else:
            v = draw(st.integers(0, N_VERTICES - 1))
        vertex_owner[v] = draw(st.integers(1, N_PLAYERS))
    return edge_road, vertex_owner


class TestLongestRoad(TestCase):
    def test_empty_board_is_zero(self) -> None:
        edge_road = np.zeros(N_EDGES, np.uint8)
        owner = np.zeros(N_VERTICES, np.uint8)
        for p in range(N_PLAYERS):
            assert (
                int(_LRL(jnp.asarray(edge_road), jnp.asarray(owner), jnp.int32(p))) == 0
            )

    def test_single_road_is_one(self) -> None:
        edge_road = np.zeros(N_EDGES, np.uint8)
        edge_road[7] = 1  # one road for player 0
        owner = np.zeros(N_VERTICES, np.uint8)
        assert int(_LRL(jnp.asarray(edge_road), jnp.asarray(owner), jnp.int32(0))) == 1
        assert int(_LRL(jnp.asarray(edge_road), jnp.asarray(owner), jnp.int32(1))) == 0

    def test_matches_numpy_reference(self) -> None:
        for seed in range(25):
            edge_road, vertex_owner = _random_occupancy(seed)
            state = _state_with(edge_road, vertex_owner)
            for p in range(N_PLAYERS):
                ref = reference.longest_road_length(state, p, 0)
                got = int(
                    _LRL(
                        jnp.asarray(edge_road), jnp.asarray(vertex_owner), jnp.int32(p)
                    )
                )
                assert got == ref, f"seed={seed} player={p}: vec={got} ref={ref}"

    @given(_boards())
    @settings(max_examples=400, deadline=None)
    def test_matches_reference_property(
        self, board: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The engine DFS must equal the reference for every fuzzed board
        (connected trails, branches, cycles, opponent-split roads)."""
        edge_road, vertex_owner = board
        state = _state_with(edge_road, vertex_owner)
        for p in range(N_PLAYERS):
            ref = reference.longest_road_length(state, p, 0)
            got = int(
                _LRL(jnp.asarray(edge_road), jnp.asarray(vertex_owner), jnp.int32(p))
            )
            assert got == ref, (
                f"player={p}: engine={got} ref={ref}\n"
                f"roads={ {int(e): int(edge_road[e]) for e in np.flatnonzero(edge_road)} }\n"
                f"buildings={ {int(v): int(vertex_owner[v]) for v in np.flatnonzero(vertex_owner)} }"
            )


class TestProductionAndPorts(TestCase):
    def _random_state(self, seed: int) -> tuple[BoardLayout, BoardState]:
        rng = np.random.default_rng(seed)
        layout = make_layout(1, key=jax.random.key(seed))
        state = make_board_state(1, key=jax.random.key(seed))
        owner = rng.choice(
            [0, 1, 2, 3, 4], size=N_VERTICES, p=[0.6, 0.12, 0.11, 0.1, 0.07]
        ).astype(np.uint8)
        vtype = np.where(owner > 0, rng.integers(1, 3, size=N_VERTICES), 0).astype(
            np.uint8
        )
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
        for seed in range(20):
            layout, state = self._random_state(seed)
            layout1, state1 = _single(layout), _single(state)
            for roll in range(2, 13):
                ref = reference.distribute_resources(layout, state, roll, 0)
                got = dice.distribute_resources(layout1, state1, jnp.int32(roll))
                assert np.array_equal(
                    np.asarray(got.player_resources),
                    np.asarray(ref.player_resources[0]),
                ), f"seed={seed} roll={roll}"

    def test_port_ratio_matches_reference(self) -> None:
        for seed in range(15):
            layout, state = self._random_state(seed)
            layout1, state1 = _single(layout), _single(state)
            for p in range(N_PLAYERS):
                for give in range(N_RESOURCES):
                    ref = reference.port_ratio(state, layout, p, give, 0)
                    got = int(
                        trade.port_ratio(
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
            common.can_afford(
                jnp.array([1, 1, 1, 1, 0], jnp.uint8), common.SETTLEMENT_COST_ARR
            )
        )
        assert not bool(
            common.can_afford(
                jnp.array([0, 1, 1, 1, 0], jnp.uint8), common.SETTLEMENT_COST_ARR
            )
        )

    def test_pay_clips_at_zero(self) -> None:
        pr = jnp.zeros((N_PLAYERS, N_RESOURCES), jnp.uint8)
        out = common.pay(pr, jnp.int32(0), common.ROAD_COST_ARR)
        assert np.array_equal(np.asarray(out[0]), np.zeros(N_RESOURCES, np.uint8))
