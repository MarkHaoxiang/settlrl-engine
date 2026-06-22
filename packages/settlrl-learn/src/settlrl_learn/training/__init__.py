"""The training loop: a net-agnostic self-play -> replay -> train -> arena loop
(:func:`learn`, :func:`arena`) over a :class:`Backend`. Two backends share it --
the flat engineered :class:`MLPBackend` and the board-graph :class:`GNNBackend`.

Training-side (equinox/optax/flashbax): not imported by the package root, so the
shipped plain-JAX play path stays dependency-light.
"""

from settlrl_learn.training.arena import arena
from settlrl_learn.training.backend import (
    Backend,
    RunState,
    load_run_state,
    save_run_state,
)
from settlrl_learn.training.gnn_backend import (
    GNNBackend,
    gnn_loss,
    make_net_agent,
    setup_policy,
)
from settlrl_learn.training.loop import learn
from settlrl_learn.training.mlp_backend import MLPBackend, mlp_loss

__all__ = [
    "Backend",
    "GNNBackend",
    "MLPBackend",
    "RunState",
    "arena",
    "gnn_loss",
    "learn",
    "load_run_state",
    "make_net_agent",
    "mlp_loss",
    "save_run_state",
    "setup_policy",
]
