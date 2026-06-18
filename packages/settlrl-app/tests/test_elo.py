"""The pure Elo math (winner-takes-all, pairwise)."""

from settlrl_app.elo import INITIAL_RATING, expected_score, winner_takes_all


def test_expected_score_is_symmetric_and_even_for_equal_ratings() -> None:
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1700, 1500) + expected_score(1500, 1700) == 1.0
    assert expected_score(1700, 1500) > 0.5  # the favourite is expected to score


def test_two_equal_players_swing_by_half_k() -> None:
    # K = 32, equal ratings: winner +16, loser -16.
    new = winner_takes_all([1500.0, 1500.0], winner=0)
    assert new[0] == 1516.0 and new[1] == 1484.0


def test_winner_gains_loser_loses_and_the_game_is_zero_sum() -> None:
    before = [1500.0, 1400.0, 1600.0, 1550.0]
    after = winner_takes_all(before, winner=1)
    assert after[1] > before[1]  # the winner gains
    assert all(after[i] < before[i] for i in (0, 2, 3))  # everyone else loses
    assert abs(sum(after) - sum(before)) < 1e-9  # ratings are conserved


def test_beating_a_stronger_field_gains_more() -> None:
    weak_field = winner_takes_all([1500.0, 1400.0, 1400.0], winner=0)[0] - 1500.0
    strong_field = winner_takes_all([1500.0, 1700.0, 1700.0], winner=0)[0] - 1500.0
    assert strong_field > weak_field


def test_losers_only_draw_with_each_other() -> None:
    # Equal-rated losers neither gain nor lose against each other, so each loses
    # exactly what it concedes to the (equal-rated) winner: K * (0 - 0.5) = -16.
    after = winner_takes_all([1500.0, 1500.0, 1500.0], winner=0)
    assert after[1] == 1484.0 and after[2] == 1484.0
    assert after[0] == 1532.0  # +16 from each of the two losers


def test_initial_rating_is_the_baseline() -> None:
    assert INITIAL_RATING == 1500.0
