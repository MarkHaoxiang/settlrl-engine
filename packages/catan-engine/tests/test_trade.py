"""Tests for trade.py: the best maritime ratio for a resource (4:1 default,
3:1 with a general port, 2:1 with the matching specific port)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from catan_engine import trade
from catan_engine.layout import N_VERTICES, PORT_V, make_layout
from catan_engine.port import Port
from catan_engine.resources import N_RESOURCES

_LAYOUT = make_layout(1, key=jax.random.key(0))
_ALLOC = jnp.asarray(np.asarray(_LAYOUT.port_allocation[0]))
_PORT_V = np.asarray(PORT_V)
_alloc_np = np.asarray(_LAYOUT.port_allocation[0])


def _owner_at(*vertices: int) -> jax.Array:
    """Player-0-owned occupancy (player + 1 == 1) at the given vertices."""
    vo = np.zeros(N_VERTICES, np.uint8)
    for v in vertices:
        vo[v] = 1
    return jnp.asarray(vo)


def _ratio(owner: jax.Array, give: int) -> int:
    return int(trade.port_ratio(owner, _ALLOC, jnp.int32(0), jnp.int32(give)))


def test_no_port_is_four_to_one() -> None:
    empty = _owner_at()
    for give in range(N_RESOURCES):
        assert _ratio(empty, give) == 4


def test_general_port_is_three_to_one() -> None:
    port = int(np.where(_alloc_np == Port.GENERAL)[0][0])
    owner = _owner_at(int(_PORT_V[port, 0]))
    for give in range(N_RESOURCES):
        assert _ratio(owner, give) == 3


def test_specific_port_is_two_to_one_for_its_resource() -> None:
    port = int(np.where(_alloc_np < Port.GENERAL)[0][0])
    resource = int(_alloc_np[port])
    owner = _owner_at(int(_PORT_V[port, 0]))
    assert _ratio(owner, resource) == 2
    other = (resource + 1) % N_RESOURCES
    assert _ratio(owner, other) == 4
