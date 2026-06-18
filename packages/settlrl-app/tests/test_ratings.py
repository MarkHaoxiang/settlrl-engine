"""The openskill-backed rating helpers (winner-takes-all, multiplayer)."""

from settlrl_app.ratings import (
    INITIAL_MU,
    INITIAL_SIGMA,
    display_rating,
    update_winner_takes_all,
)

_START = (INITIAL_MU, INITIAL_SIGMA)


def test_a_fresh_rating_displays_at_the_baseline() -> None:
    assert round(display_rating(*_START)) == 1000


def test_winner_gains_losers_lose_and_uncertainty_shrinks() -> None:
    after = update_winner_takes_all([_START, _START, _START, _START], winner=1)
    assert after[1][0] > INITIAL_MU  # the winner's skill estimate rises
    assert all(after[i][0] < INITIAL_MU for i in (0, 2, 3))  # everyone else falls
    assert all(sigma < INITIAL_SIGMA for _, sigma in after)  # a game cuts uncertainty


def test_winner_outranks_the_losers_on_the_displayed_number() -> None:
    after = update_winner_takes_all([_START, _START], winner=0)
    assert display_rating(*after[0]) > display_rating(*after[1])


def test_beating_a_stronger_field_gains_more() -> None:
    weak = update_winner_takes_all(
        [_START, (15.0, INITIAL_SIGMA), (15.0, INITIAL_SIGMA)], winner=0
    )[0][0]
    strong = update_winner_takes_all(
        [_START, (35.0, INITIAL_SIGMA), (35.0, INITIAL_SIGMA)], winner=0
    )[0][0]
    assert strong - INITIAL_MU > weak - INITIAL_MU
