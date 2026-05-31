"""Maritime trade rules: the best bank/port exchange ratio for a resource.

Lives here rather than in ``port.py`` because the port *rule* needs the board
geometry (``layout.PORT_V``) and ``layout`` already imports ``port`` for the
``Port`` enum -- colocating would create an import cycle. ``port.py`` stays the
pure ``Port`` enum; ``trade.py`` is the rule over it.

Port membership is a per-vertex node feature derived by scattering over the dense
``PORT_V`` map (each port owns two vertices), so no padded vertex->port reverse
map / sentinel is needed.
"""

from __future__ import annotations

import jax.numpy as jnp

from catan_engine.layout import N_VERTICES, PORT_V, PortAllocVec
from catan_engine.port import Port
from catan_engine.state import IntScalar, VertexOwnerVec

# Static port geometry, flattened to one slot per (port, vertex) pair.
_PORT_VERTS = PORT_V.reshape(-1)  # (2 * N_PORTS,) port-vertex ids
_PORT_SLOT = jnp.repeat(jnp.arange(PORT_V.shape[0]), 2)  # owning port per slot
_IS_PORT_VERTEX = jnp.zeros((N_VERTICES,), jnp.bool_).at[_PORT_VERTS].set(True)


def port_ratio(
    vertex_owner: VertexOwnerVec,
    port_allocation: PortAllocVec,
    player: IntScalar,
    give: IntScalar,
) -> IntScalar:
    """Best maritime ratio for giving ``give``: 4, or 3 (general), or 2 (match)."""
    # Scatter each port's (per-game) type onto its two vertices; the boolean
    # ``_IS_PORT_VERTEX`` mask, not a fill sentinel, marks non-port vertices.
    ptypes = port_allocation[_PORT_SLOT]  # (2 * N_PORTS,)
    vert_port = jnp.zeros((N_VERTICES,), port_allocation.dtype).at[_PORT_VERTS].set(
        ptypes
    )
    my_port = (vertex_owner == player + 1) & _IS_PORT_VERTEX
    general = jnp.any(my_port & (vert_port == Port.GENERAL))
    match = jnp.any(my_port & (vert_port == give))
    return jnp.where(match, 2, jnp.where(general, 3, 4)).astype(jnp.int32)
