"""Per-action tests for the vectorized engine.

Convention: one file per action, each with a success case as an
``assertExpectedInline`` snapshot plus invalidation cases (wrong phase, bad
params, unaffordable, ...) that assert ``INVALID`` and that state is unchanged.
Build positions with the shared fixtures (which use the board.py helpers).

All 15 actions are vectorized in action_vec.py and covered here (one file per
action). DomesticTrade is intentionally deferred (never implemented). The NumPy
single-game path has been removed; its reference implementation now lives in
``tests/reference.py`` (the differential oracle for ``tests/test_rules_vec.py``).
``ActionResult`` lives in ``action_vec.py``.

TODO: add a unified ``step`` for ``env.py`` over this action set.
"""
