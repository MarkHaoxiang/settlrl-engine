import jax.numpy as jnp
import numpy as np
from expecttest import TestCase

from catan_engine.board.state import make_board_state
from catan_engine.board.resources import (
    BANK_INITIAL,
    N_PLAYERS,
    N_RESOURCES,
    compute_bank_resources,
)


class TestBankResources(TestCase):
    def test_full_bank_when_no_holdings(self) -> None:
        player_resources = jnp.zeros((1, N_PLAYERS, N_RESOURCES), dtype=jnp.uint8)
        bank = np.asarray(compute_bank_resources(player_resources)[0])
        assert bank.tolist() == [BANK_INITIAL] * N_RESOURCES

    def test_bank_decreases_by_single_player_holdings(self) -> None:
        player_resources = jnp.zeros((1, N_PLAYERS, N_RESOURCES), dtype=jnp.uint8)
        # Player 0 holds 5 sheep (resource index 0) and 3 ore (resource index 4)
        player_resources = player_resources.at[0, 0, 0].set(5)
        player_resources = player_resources.at[0, 0, 4].set(3)
        bank = np.asarray(compute_bank_resources(player_resources)[0])
        assert bank[0] == BANK_INITIAL - 5  # sheep
        assert bank[4] == BANK_INITIAL - 3  # ore
        assert bank[1] == BANK_INITIAL  # wheat unchanged
        assert bank[2] == BANK_INITIAL  # wood unchanged
        assert bank[3] == BANK_INITIAL  # brick unchanged

    def test_bank_decreases_across_all_players(self) -> None:
        # Each player holds 1 of every resource
        player_resources = jnp.ones((1, N_PLAYERS, N_RESOURCES), dtype=jnp.uint8)
        bank = np.asarray(compute_bank_resources(player_resources)[0])
        assert bank.tolist() == [BANK_INITIAL - N_PLAYERS] * N_RESOURCES

    def test_bank_batched(self) -> None:
        B = 3
        player_resources = jnp.zeros((B, N_PLAYERS, N_RESOURCES), dtype=jnp.uint8)
        # Batch item 1: player 0 has 2 wheat
        player_resources = player_resources.at[1, 0, 1].set(2)
        bank = np.asarray(compute_bank_resources(player_resources))
        assert bank.shape == (B, N_RESOURCES)
        assert bank[0, 1] == BANK_INITIAL  # batch 0 unchanged
        assert bank[1, 1] == BANK_INITIAL - 2  # batch 1 reduced
        assert bank[2, 1] == BANK_INITIAL  # batch 2 unchanged


class TestBoardStateResources(TestCase):
    def test_make_board_state_resource_shape(self) -> None:
        state = make_board_state(batch_size=2)
        assert state.player_resources.shape == (2, N_PLAYERS, N_RESOURCES)

    def test_make_board_state_resources_zero(self) -> None:
        state = make_board_state(batch_size=1)
        assert np.asarray(state.player_resources).sum() == 0

    def test_make_board_state_victory_points_shape(self) -> None:
        state = make_board_state(batch_size=2)
        assert state.victory_points.shape == (2, N_PLAYERS)

    def test_make_board_state_victory_points_zero(self) -> None:
        state = make_board_state(batch_size=1)
        assert np.asarray(state.victory_points).sum() == 0
