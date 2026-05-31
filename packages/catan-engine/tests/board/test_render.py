import jax
import jax.numpy as jnp
import numpy as np
from expecttest import TestCase

from catan_engine.board.dev_cards import DevCard
from catan_engine.board.layout import N_VERTICES, BoardLayout, make_layout
from catan_engine.board.state import BoardState, GamePhase, make_board_state
from tests.render import _EDGE_VERTICES, BoardRenderer


def _sample_board() -> tuple[BoardLayout, BoardState]:
    """A deterministic mid-game board with hand-placed state, for snapshots."""
    layout = make_layout(batch_size=1, key=jax.random.key(0))
    state = make_board_state(batch_size=1)

    vertex_owner = np.asarray(state.vertex_owner).copy()
    vertex_type = np.asarray(state.vertex_type).copy()
    edge_road = np.asarray(state.edge_road).copy()
    resources = np.asarray(state.player_resources).copy()
    vp = np.asarray(state.victory_points).copy()
    dev_hand = np.asarray(state.dev_hand).copy()
    dev_deck = np.asarray(state.dev_deck).copy()
    knights = np.asarray(state.knights_played).copy()

    # player 1: settlement @5, city @44; player 2: city @12; player 3: settlement @30
    for vertex, owner, kind in [(5, 1, 1), (44, 1, 2), (12, 2, 2), (30, 3, 1)]:
        vertex_owner[0, vertex] = owner
        vertex_type[0, vertex] = kind
    for edge, owner in [(0, 1), (3, 1), (10, 2), (20, 3), (40, 1)]:
        edge_road[0, edge] = owner
    resources[0, 0] = [2, 1, 0, 3, 1]
    resources[0, 1] = [0, 0, 2, 1, 0]
    resources[0, 2] = [1, 1, 1, 0, 2]
    vp[0] = [4, 2, 1, 0]

    # dev cards: player 1 holds a VP card + Monopoly, player 2 holds a Knight.
    dev_hand[0, 0, int(DevCard.VICTORY_POINT)] = 1
    dev_hand[0, 0, int(DevCard.MONOPOLY)] = 1
    dev_hand[0, 1, int(DevCard.KNIGHT)] = 1
    for card in (DevCard.VICTORY_POINT, DevCard.MONOPOLY, DevCard.KNIGHT):
        dev_deck[0, int(card)] -= 1
    knights[0] = [3, 0, 1, 0]

    def u8(value: int) -> jax.Array:
        return jnp.asarray(np.array([value], dtype=np.uint8))

    state = state._replace(
        vertex_owner=jnp.asarray(vertex_owner),
        vertex_type=jnp.asarray(vertex_type),
        edge_road=jnp.asarray(edge_road),
        robber=u8(9),
        player_resources=jnp.asarray(resources),
        victory_points=jnp.asarray(vp),
        dev_hand=jnp.asarray(dev_hand),
        dev_deck=jnp.asarray(dev_deck),
        knights_played=jnp.asarray(knights),
        current_player=u8(0),
        phase=u8(int(GamePhase.MAIN)),
        dice_roll=u8(8),
        longest_road_owner=u8(0),
        longest_road_len=u8(5),
        largest_army_owner=u8(0),
    )
    return layout, state


class TestBoardRenderer(TestCase):
    def test_edge_index_in_range(self) -> None:
        for a, b in _EDGE_VERTICES:
            assert 0 <= a < N_VERTICES and 0 <= b < N_VERTICES

    def test_empty_board_snapshot(self) -> None:
        layout = make_layout(batch_size=1, key=jax.random.key(0))
        state = make_board_state(batch_size=1)
        self.assertExpectedInline(
            str(BoardRenderer(layout, state)),
            """\
Catan Board
============================================================




          ORE             3:1
               /o\\     /o\\     /o\\
              /   \\   /   \\   /   \\
            o/     \\o/     \\o/     \\o
            |  SHP  |  ORE  |  BRK  |
            |   5   |   6   |  10   |
            |       |       |  <R>  |
           /o\\     /o\\     /o\\     /o\\   3:1
          /   \\   /   \\   /   \\   /   \\
        o/     \\o/     \\o/     \\o/     \\o
  WOD   |  WHT  |  WOD  |  WOD  |  SHP  |
        |   9   |   2   |  10   |  11   |
        |       |       |       |       |
       /o\\     /o\\     /o\\     /o\\     /o\\
      /   \\   /   \\   /   \\   /   \\   /   \\
    o/     \\o/     \\o/     \\o/     \\o/     \\o
    |  ORE  |  SHP  |  WOD  |  DST  |  WHT  |
    |   8   |   4   |   3   |       |  12   |   3
    |       |       |       |       |       |
    o\\     /o\\     /o\\     /o\\     /o\\     /o
      \\   /   \\   /   \\   /   \\   /   \\   /
       \\o/     \\o/     \\o/     \\o/     \\o/
        |  SHP  |  ORE  |  BRK  |  BRK  |
        |   8   |   3   |  11   |   6   |
  3:1   |       |       |       |       |
        o\\     /o\\     /o\\     /o\\     /o
          \\   /   \\   /   \\   /   \\   /
           \\o/     \\o/     \\o/     \\o/   BRK
            |  WHT  |  WHT  |  WOD  |
            |   4   |   9   |   5   |
            |       |       |       |
            o\\     /o\\     /o\\     /o
              \\   /   \\   /   \\   /
               \\o/     \\o/     \\o/
          SHP             WHT




Legend: o=vertex  1-4=settlement(player)  A-D=city(player)  digit on edge=road(player)  <R>=robber

Phase SETUP_SETTLEMENT  |  Current player 1  |  Dice -  |  Setup 0/8

Players
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   P |   Shp |   Wht |   Wod |   Brk |   Ore |   Hand |   Dev |   Knt |   Set |   Cit |   Rd |   VP | Awd   |
+====+=====+=======+=======+=======+=======+=======+========+=======+=======+=======+=======+======+======+=======+
| >  |   1 |     0 |     0 |     0 |     0 |     0 |      0 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   2 |     0 |     0 |     0 |     0 |     0 |      0 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   3 |     0 |     0 |     0 |     0 |     0 |      0 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   4 |     0 |     0 |     0 |     0 |     0 |      0 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+

Longest Road: unclaimed    Largest Army: unclaimed

Dev deck  KNT:14  RDB:2  YOP:2  MNP:2  VPT:5

Bank
+---------+---------+--------+---------+-------+
|   Sheep |   Wheat |   Wood |   Brick |   Ore |
+=========+=========+========+=========+=======+
|      19 |      19 |     19 |      19 |    19 |
+---------+---------+--------+---------+-------+

Robber: tile 0 (BRK, 10)
""",
        )

    def test_full_board_snapshot(self) -> None:
        layout, state = _sample_board()
        self.assertExpectedInline(
            str(BoardRenderer(layout, state)),
            """\
Catan Board
============================================================




          ORE             3:1
               /o\\     /o\\     1o\\
              /   \\   /   \\   1   \\
            B/     \\o/     \\o1     \\o
            |  SHP  |  ORE  |  BRK  |
            |   5   |   6   |  10   |
            |       |       |       |
           3o\\     /o2     /1\\     1o\\   3:1
          3   \\   /   2   /   \\   1   \\
        o3     \\o/     2o/     \\o1     \\o
  WOD   |  WHT  |  WOD  |  WOD  |  SHP  |
        |   9   |   2   |  10   |  11   |
        |       |       |       |       |
       /o\\     /o\\     /o\\     /o\\     /o\\
      /   \\   /   \\   /   \\   /   \\   /   \\
    o/     \\o/     \\o/     \\o/     \\o/     \\o
    |  ORE  |  SHP  |  WOD  |  DST  |  WHT  1
    |   8   |   4   |   3   |       |  12   1   3
    |       |       |  <R>  |       |       1
    o\\     /o\\     /o\\     /3\\     /o\\     /o
      \\   /   \\   /   \\   /   \\   /   \\   /
       \\o/     \\o/     \\o/     \\o/     \\o/
        |  SHP  |  ORE  |  BRK  |  BRK  |
        |   8   |   3   |  11   |   6   |
  3:1   |       |       |       |       |
        o\\     /A\\     /o\\     /o\\     /o
          \\   /   \\   /   \\   /   \\   /
           \\o/     \\o/     \\o/     \\o/   BRK
            |  WHT  |  WHT  |  WOD  |
            |   4   |   9   |   5   |
            |       |       |       |
            o\\     /o\\     /o\\     /o
              \\   /   \\   /   \\   /
               \\o/     \\o/     \\o/
          SHP             WHT




Legend: o=vertex  1-4=settlement(player)  A-D=city(player)  digit on edge=road(player)  <R>=robber

Phase MAIN  |  Current player 1  |  Dice 8

Players
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   P |   Shp |   Wht |   Wod |   Brk |   Ore |   Hand |   Dev |   Knt |   Set |   Cit |   Rd |   VP | Awd   |
+====+=====+=======+=======+=======+=======+=======+========+=======+=======+=======+=======+======+======+=======+
| >  |   1 |     2 |     1 |     0 |     3 |     1 |      7 |     2 |     3 |     1 |     1 |    3 |    9 | LR LA |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   2 |     0 |     0 |     2 |     1 |     0 |      3 |     1 |     0 |     0 |     1 |    1 |    2 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   3 |     1 |     1 |     1 |     0 |     2 |      5 |     0 |     1 |     1 |     0 |    1 |    1 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   4 |     0 |     0 |     0 |     0 |     0 |      0 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+

Longest Road: player 1 (length 5)    Largest Army: player 1 (3 knights)

Dev deck  KNT:13  RDB:2  YOP:2  MNP:1  VPT:4

Bank
+---------+---------+--------+---------+-------+
|   Sheep |   Wheat |   Wood |   Brick |   Ore |
+=========+=========+========+=========+=======+
|      16 |      17 |     16 |      15 |    16 |
+---------+---------+--------+---------+-------+

Robber: tile 9 (WOD, 3)
""",
        )
