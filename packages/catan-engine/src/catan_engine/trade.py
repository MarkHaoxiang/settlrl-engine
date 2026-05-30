"""Maritime trade rules: the best bank/port exchange ratio for a resource.

Lives here rather than in ``port.py`` because the port *rule* needs the board
geometry (``geometry.V_PORT``, derived from ``layout``) and ``layout`` already
imports ``port`` for the ``Port`` enum -- colocating would create an import cycle.
``port.py`` stays the pure ``Port`` enum; ``trade.py`` is the rule over it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from catan_engine.geometry import NO_IDX, V_PORT
from catan_engine.port import Port


def port_ratio(
    vertex_owner: jax.Array,
    port_allocation: jax.Array,
    player: jax.Array,
    give: jax.Array,
) -> jax.Array:
    """Best maritime ratio for giving ``give``: 4, or 3 (general), or 2 (match)."""
    owns = vertex_owner == player + 1
    is_port = V_PORT != NO_IDX
    ptype = port_allocation[jnp.where(is_port, V_PORT, 0)]
    my_port = owns & is_port
    general = jnp.any(my_port & (ptype == Port.GENERAL))
    match = jnp.any(my_port & (ptype == give))
    return jnp.where(match, 2, jnp.where(general, 3, 4)).astype(jnp.int32)
