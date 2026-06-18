"""Multiplayer rating via openskill — the patent-free Weng-Lin / Plackett-Luce
model (the open cousin of TrueSkill).

A finished game is one rank vector over its seats; winner-takes-all means the
winner ranks first and everyone else ties behind. A subject's skill is a
``(mu, sigma)`` pair (mean and uncertainty); the leaderboard orders by the
conservative ordinal ``mu - 3*sigma``, scaled to a familiar rating range for
display only — the scaling is cosmetic and never changes the ranking.
"""

from collections.abc import Sequence

from openskill.models import PlackettLuce

_MODEL = PlackettLuce()
_DEFAULT = _MODEL.rating()
INITIAL_MU: float = _DEFAULT.mu
INITIAL_SIGMA: float = _DEFAULT.sigma

# Maps a fresh rating (ordinal 0) to 1000 and spreads skill into Elo-like
# numbers. Display only.
_DISPLAY_ALPHA = 20.0
_DISPLAY_TARGET = 1000.0

Skill = tuple[float, float]  # (mu, sigma)


def display_rating(mu: float, sigma: float) -> float:
    """The leaderboard number: the conservative ordinal, scaled for readability."""
    return float(
        _MODEL.rating(mu=mu, sigma=sigma).ordinal(
            alpha=_DISPLAY_ALPHA, target=_DISPLAY_TARGET
        )
    )


def update_winner_takes_all(skills: Sequence[Skill], winner: int) -> list[Skill]:
    """Updated ``(mu, sigma)`` for each seat: ``winner`` ranks first, the rest
    tie behind it."""
    teams = [[_MODEL.rating(mu=mu, sigma=sigma)] for mu, sigma in skills]
    ranks = [0.0 if i == winner else 1.0 for i in range(len(skills))]
    rated = _MODEL.rate(teams, ranks=ranks)
    return [(float(team[0].mu), float(team[0].sigma)) for team in rated]
