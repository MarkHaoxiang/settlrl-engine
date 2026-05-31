"""RL environment surface.

Re-exports the batched env's public API (``step`` / ``available`` /
``BatchedCatanEnv`` and the space descriptors) so ``catan_engine.env`` behaves as
before the split into submodules; the single-game PettingZoo-AEC wrapper lives in
``catan_engine.env.aec``.
"""

from catan_engine.env.batched import *  # noqa: F401,F403
