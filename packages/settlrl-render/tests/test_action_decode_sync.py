"""Action-decode sync: ``actions.decode_actions`` must interpret the engine's
flat action table (``flat_to_action``) exactly as the engine does. For every
concrete move in the table we check the renderer agrees on the action type and
resolves the parameter ``idx`` to the same board coordinate / resource the
engine would. This catches the table being reordered, an action's ``idx``
changing meaning, or a new action type the renderer can't decode.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from settlrl_engine.board.layout import (
    edge_cubes,
    tile_cube,
    vertex_cube,
)
from settlrl_engine.env import N_FLAT, ActionType, flat_to_action
from settlrl_render.api.actions import _RESOURCE_NAMES, _decode, decode_actions

_row_type, _row_params = flat_to_action(jnp.arange(N_FLAT))
_ATYPE = np.asarray(_row_type)
_IDX = np.asarray(_row_params.idx)
_TARGET = np.asarray(_row_params.target)

ALL_FLAT = list(range(N_FLAT))

_VERTEX_TYPES = {"setup_settlement", "build_settlement", "build_city"}
_ROAD_TYPES = {"setup_road", "build_road"}
_ROBBER_TYPES = {"move_robber", "play_knight"}


def test_every_flat_index_decodes() -> None:
    # The whole table must round-trip without raising, and preserve its flat id.
    models = decode_actions(ALL_FLAT)
    assert len(models) == N_FLAT
    for flat, m in enumerate(models):
        assert m.flat == flat
        assert m.label  # every action carries a human label


def test_decoded_type_matches_engine() -> None:
    for flat in ALL_FLAT:
        expected = ActionType(int(_ATYPE[flat])).name.lower()
        assert _decode(flat).type == expected, f"flat {flat}"


@pytest.mark.parametrize("flat", ALL_FLAT)
def test_decoded_target_matches_engine_lookup(flat: int) -> None:
    """The decoded board target / resource must agree with the engine's own
    index lookups for this action's ``idx`` (and ``target``)."""
    m = _decode(flat)
    idx = int(_IDX[flat])
    target = int(_TARGET[flat])

    if m.type in _VERTEX_TYPES:
        q, r, s = vertex_cube(idx)
        assert m.vertex is not None
        assert (m.vertex.q, m.vertex.r, m.vertex.s) == (q, r, s)

    elif m.type in _ROAD_TYPES:
        assert m.edge is not None
        rendered = {
            (m.edge.a.q, m.edge.a.r, m.edge.a.s),
            (m.edge.b.q, m.edge.b.r, m.edge.b.s),
        }
        assert rendered == set(edge_cubes(idx))

    elif m.type in _ROBBER_TYPES:
        assert m.tile is not None
        assert (m.tile.q, m.tile.r) == tile_cube(idx)[:2]
        assert m.victim == target

    elif m.type in ("discard", "play_monopoly"):
        assert m.resource == _RESOURCE_NAMES[idx]

    elif m.type == "play_year_of_plenty":
        assert m.resources == [_RESOURCE_NAMES[idx], _RESOURCE_NAMES[target]]

    elif m.type == "maritime_trade":
        assert m.give == _RESOURCE_NAMES[idx]
        assert m.receive == _RESOURCE_NAMES[target]


def test_decoded_types_are_known_action_types() -> None:
    # Every type the renderer emits is a real engine ActionType name — no
    # fall-through to a raw/unknown string.
    valid = {a.name.lower() for a in ActionType}
    seen = {m.type for m in decode_actions(ALL_FLAT)}
    assert seen <= valid
    # And the table actually exercises a broad span of action types.
    assert len(seen) >= 10
