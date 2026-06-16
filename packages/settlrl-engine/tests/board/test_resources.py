import jax.numpy as jnp
import numpy as np
from expecttest import TestCase
from settlrl_engine.board.resources import (
    BANK_INITIAL,
    N_PLAYERS,
    N_RESOURCES,
    compute_bank_resources,
)


class TestBankResources(TestCase):
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
