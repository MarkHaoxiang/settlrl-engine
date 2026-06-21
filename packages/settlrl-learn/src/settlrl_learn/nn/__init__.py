"""Network definitions: the shipped plain-JAX value+policy MLP (:mod:`.mlp`) and
the training-side equinox graph nets (:mod:`.graph` / :mod:`.graphnet` /
:mod:`.board_gnn` / :mod:`.architectures`).

Kept import-light on purpose: this ``__init__`` pulls **no** equinox/jraph, so
``import settlrl_learn`` (which reaches only the plain-JAX :mod:`.mlp`) stays free
of training dependencies. A guard test enforces it; import the equinox modules by
their submodule path.
"""
