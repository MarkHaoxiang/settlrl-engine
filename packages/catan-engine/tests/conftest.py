"""Turn the rule modules' jaxtyping annotations into enforced runtime checks.

The single-game rule modules annotate their array params with the un-batched
aliases defined in state / resources / layout / dev_cards (``EdgeRoadVec``,
``IntScalar``, ...). Installing jaxtyping's import hook (backed by beartype) here
makes those annotations *checked* during the test run -- every call, and every
``jax.jit`` trace, verifies shapes and dtypes -- at zero cost to the shipped
package (the hook only exists in the test session).

The hook must be installed before the target modules are first imported, so this
lives in the top-level ``tests`` conftest, which pytest loads before any test
module (and before ``tests/actions/conftest.py``).

``action`` / ``env`` are intentionally excluded: their cores are annotated with
*batched* aliases but execute unbatched under ``vmap``, so enforcing them needs a
separate batched -> single-game re-annotation pass first.
"""

from jaxtyping import install_import_hook

install_import_hook(
    [
        "catan_engine.mechanics.placement",
        "catan_engine.mechanics.awards",
        "catan_engine.mechanics.dice",
        "catan_engine.mechanics.robber",
        "catan_engine.mechanics.setup",
        "catan_engine.mechanics.trade",
        "catan_engine.mechanics.development",
    ],
    "beartype.beartype",
)
