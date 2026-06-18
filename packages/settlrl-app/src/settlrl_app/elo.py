"""Standard Elo, applied pairwise with a winner-takes-all result.

The classic Elo expected-score formula (Wikipedia, "Elo rating system"): a
player rated ``R`` scores an expected ``E = 1 / (1 + 10**((opp - R) / 400))``
against an opponent rated ``opp`` and is corrected toward the realised score by
``K * (actual - E)``. A multiplayer game is scored as the round-robin of its
pairs: the winner beats every other player (actual 1 / 0) and the non-winners
draw with one another (actual 0.5). Each participant's delta is summed over its
opponents and applied against the pre-game ratings, so a game is zero-sum over
its participants — and ratings in different ``n_players`` buckets never interact.
"""

from collections.abc import Sequence

INITIAL_RATING = 1500.0
_K = 32.0


def expected_score(rating: float, opponent: float) -> float:
    return float(1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0)))


def winner_takes_all(ratings: Sequence[float], winner: int) -> list[float]:
    """New ratings for one game's participants (at least two), ``winner``
    indexing the one who won; every other participant draws with the rest."""
    new = list(ratings)
    for i, rating in enumerate(ratings):
        delta = 0.0
        for j, opponent in enumerate(ratings):
            if i == j:
                continue
            actual = 1.0 if i == winner else 0.0 if j == winner else 0.5
            delta += _K * (actual - expected_score(rating, opponent))
        new[i] = rating + delta
    return new
