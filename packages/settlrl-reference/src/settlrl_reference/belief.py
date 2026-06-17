"""Card counting: per-observer bounds on every player's hidden hand.

A plain, independent statement of what each player can *prove* about the others'
hands from public information alone -- the differential oracle for the engine's
``belief.py`` and the source for the renderer's card-counting panel.

The only hidden thing in this game is the *type* of a card taken by a robber
steal, seen only by the thief and the victim; the held identities of
development cards are also private, tracked separately as the public
played-card tally. Everything else is public: production, build costs,
discards, maritime and domestic trades, Monopoly, and every hand size and the
bank. So a third party's uncertainty about a hand is exactly the steals it did
not witness -- and with two players (you are party to every steal) the bounds
are always exact.
"""

from __future__ import annotations

from dataclasses import dataclass

from settlrl_reference.game import Action, Game, MoveRobber, PlayKnight, PlayMonopoly
from settlrl_reference.types import DEV_CARDS, RESOURCES, DevCard

_NR = len(RESOURCES)


def _hand(game: Game, p: int) -> list[int]:
    """Player ``p``'s resource counts in canonical order."""
    return [game.players[p].resources[r] for r in RESOURCES]


@dataclass
class Belief:
    """Per-observer proven bounds on every player's per-resource holdings.

    ``res_lo[o][p][r] <= true <= res_hi[o][p][r]`` for everything observer ``o``
    can prove; the observer's own row (``o == p``) is exact. ``dev_played`` is
    the public count of played development cards by type (the same for all
    observers).
    """

    res_lo: list[list[list[int]]]
    res_hi: list[list[list[int]]]
    dev_played: dict[DevCard, int]
    n_players: int

    @staticmethod
    def new(n_players: int) -> Belief:
        """The belief for a fresh game: empty hands, nothing played."""

        def zeros() -> list[list[list[int]]]:
            return [[[0] * _NR for _ in range(n_players)] for _ in range(n_players)]

        return Belief(zeros(), zeros(), dict.fromkeys(DEV_CARDS, 0), n_players)

    def update(self, before: Game, after: Game, action: Action) -> None:
        """Advance the belief across one applied action (``before`` -> ``after``)."""
        n = self.n_players
        before_res = [_hand(before, p) for p in range(n)]
        after_res = [_hand(after, p) for p in range(n)]
        delta = [
            [after_res[p][r] - before_res[p][r] for r in range(_NR)] for p in range(n)
        ]
        hand_size = [sum(after_res[p]) for p in range(n)]
        total = [sum(after_res[p][r] for p in range(n)) for r in range(_NR)]

        # A held dev card going down is a public play (a buy goes up, so hidden
        # draws never register here).
        for c in DEV_CARDS:
            self.dev_played[c] += sum(
                max(0, before.players[p].dev_cards[c] - after.players[p].dev_cards[c])
                for p in range(n)
            )

        # The robber steal is the one hidden resource flow.
        thief = before.current_player
        victim = action.victim if isinstance(action, MoveRobber | PlayKnight) else None
        stole = victim is not None and hand_size[victim] < sum(before_res[victim])
        mono = int(action.resource) if isinstance(action, PlayMonopoly) else None

        for o in range(n):
            lo, hi = self.res_lo[o], self.res_hi[o]
            if stole and o != thief and o != victim:
                assert victim is not None  # implied by ``stole``
                # A third party sees one card pass victim -> thief, not its type:
                # the victim keeps at least one fewer of each type it could hold,
                # and the thief may now hold one more of any of them.
                for r in range(_NR):
                    hi[thief][r] += int(hi[victim][r] > 0)
                    lo[victim][r] = max(0, lo[victim][r] - 1)
            else:
                # Every other flow (and any steal this observer was party to) is
                # public, so the typed change tracks both bounds exactly.
                for p in range(n):
                    for r in range(_NR):
                        lo[p][r] = max(0, lo[p][r] + delta[p][r])
                        hi[p][r] = max(0, hi[p][r] + delta[p][r])
            if mono is not None:  # Monopoly announces every exact count of the type
                for p in range(n):
                    lo[p][mono] = hi[p][mono] = after_res[p][mono]
            for r in range(_NR):  # the observer always knows its own hand exactly
                lo[o][r] = hi[o][r] = after_res[o][r]
            _tighten(lo, hi, hand_size, total, n)


def _tighten(
    lo: list[list[int]],
    hi: list[list[int]],
    hand_size: list[int],
    total: list[int],
    n: int,
) -> None:
    """Sharpen the bounds in place with the public hand sizes and per-resource
    totals (each rule is a sound deduction, applied once per step)."""
    for p in range(n):
        for r in range(_NR):
            others_lo = sum(lo[q][r] for q in range(n)) - lo[p][r]
            # No more of a type than your hand holds, nor than the public total
            # leaves once everyone else's proven minimum is set aside.
            hi[p][r] = max(0, min(hi[p][r], hand_size[p], total[r] - others_lo))
    for p in range(n):
        for r in range(_NR):
            other_types = sum(hi[p]) - hi[p][r]
            others_hi = sum(hi[q][r] for q in range(n)) - hi[p][r]
            # At least your hand minus the most the other types could be, and at
            # least the public total minus the most everyone else could hold.
            lo[p][r] = max(
                lo[p][r], hand_size[p] - other_types, total[r] - others_hi, 0
            )
