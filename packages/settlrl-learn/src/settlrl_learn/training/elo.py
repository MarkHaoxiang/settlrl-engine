"""Anchored Elo: a single comparable strength number per checkpoint.

The arena scores the net against a *fixed* set of anchors whose Elo never moves
(``random``, ``lookahead(heuristic)`` pinned at 0, optionally frozen self-play
checkpoints). :func:`anchored_elo` then places the net on that fixed scale by
maximum likelihood, so the per-iteration number is comparable across the whole
run and across runs -- the AlphaZero/MuZero anchored-baseline scheme, not a
within-pool round-robin (which drifts when the pool changes).

A training-side module: not imported by the package root.
"""

from __future__ import annotations

from collections.abc import Iterable


def expected_score(rating: float, opponent: float) -> float:
    """Logistic win probability of a player at ``rating`` vs ``opponent`` (the
    standard Elo curve, 400-point scale)."""
    return float(1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0)))


def anchored_elo(
    anchors: Iterable[tuple[float, float, int]],
    *,
    lo: float = -4000.0,
    hi: float = 4000.0,
    iters: int = 64,
) -> float:
    """Maximum-likelihood Elo of a player from results vs fixed-Elo anchors.

    ``anchors`` is ``(anchor_elo, wins, games)`` per anchor. The expected total
    score ``sum_a games_a * expected_score(R, elo_a)`` is monotone in ``R``, so
    the MLE solves ``= sum_a wins_a`` by bisection. Wins are continuity-corrected
    to ``[0.5, games-0.5]`` so a saturated anchor (0% / 100%) can't drive ``R`` to
    ``+-inf``. Returns ``nan`` if no anchor has games."""
    data = [(elo, w, g) for elo, w, g in anchors if g > 0]
    if not data:
        return float("nan")
    target = sum(min(max(w, 0.5), g - 0.5) for _, w, g in data)

    def predicted(r: float) -> float:
        return sum(g * expected_score(r, elo) for elo, _, g in data)

    a, b = lo, hi
    for _ in range(iters):
        m = 0.5 * (a + b)
        if predicted(m) < target:
            a = m
        else:
            b = m
    return 0.5 * (a + b)
