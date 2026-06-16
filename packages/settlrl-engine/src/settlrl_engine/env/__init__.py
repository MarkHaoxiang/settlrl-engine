"""RL environment surface.

Re-exports the batched env's public API (``step`` / ``available`` /
``BatchedSettlrlEnv`` and the space descriptors) so ``settlrl_engine.env`` behaves as
before the split into submodules; the single-game PettingZoo-AEC wrapper lives in
``settlrl_engine.env.aec``.
"""

from settlrl_engine.env.batched import *  # noqa: F403
