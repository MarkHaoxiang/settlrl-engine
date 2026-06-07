"""Turn the policy modules' jaxtyping annotations into enforced runtime checks
(same pattern as catan-engine's top-level conftest: the hook must be installed
before the target modules are first imported)."""

from jaxtyping import install_import_hook

install_import_hook(
    [
        "catan_agents.shared.policy",
        "catan_agents.shared.value",
        "catan_agents.shared.baselines",
        "catan_agents.shared.greedy",
        "catan_agents.shared.sample",
        "catan_agents.shared.evaluate",
        "catan_agents.search.greedy",
        "catan_agents.search.mcts",
    ],
    "beartype.beartype",
)
