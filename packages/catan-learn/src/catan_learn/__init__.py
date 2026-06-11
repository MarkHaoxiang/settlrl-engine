"""Learned value and policy functions for catan-agents.

Networks plug into the agents through their existing seams — a value model is
a :class:`~catan_agents.shared.policy.ValueFunction` and a policy model a
:class:`~catan_agents.shared.policy.PolicyPrior` — so every search agent
consumes them unchanged.
"""

from catan_learn.features import FEATURE_DIM, features
from catan_learn.model import (
    MLPParams,
    init_mlp,
    init_prior_params,
    init_value_params,
    load_params,
    make_net_prior,
    make_net_value,
    mlp,
    save_params,
)
from catan_learn.train import fit, value_loss

__all__ = [
    "FEATURE_DIM",
    "MLPParams",
    "features",
    "fit",
    "init_mlp",
    "init_prior_params",
    "init_value_params",
    "load_params",
    "make_net_prior",
    "make_net_value",
    "mlp",
    "save_params",
    "value_loss",
]
