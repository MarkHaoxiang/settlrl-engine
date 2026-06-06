"""Turn the policy modules' jaxtyping annotations into enforced runtime checks
(same pattern as catan-engine's top-level conftest: the hook must be installed
before the target modules are first imported)."""

from jaxtyping import install_import_hook

install_import_hook(
    [
        "catan_agents.policy",
        "catan_agents.baselines",
        "catan_agents.greedy",
        "catan_agents.evaluate",
    ],
    "beartype.beartype",
)
