# settlrl-learn

Learned value and policy functions for [settlrl-agents](../settlrl-agents).

Models plug into the agents through their existing seams: a value model is a
`ValueFunction` and a policy model a `PolicyPrior`, so the search agents consume
them unchanged — `make_search(make_net_value(params),
prior=make_net_prior(params))`.

- `features(layout, state, player)` — the position as a flat vector from one
  seat: the player's own block, the max and mean over opponents' blocks, and
  a global block. The width (`FEATURE_DIM`) is the same at every player
  count, and features are only ever computed on concrete (sampled) worlds.
- `init_value_params` / `init_prior_params` build untrained MLP stand-ins;
  `make_net_value` / `make_net_prior` adapt any params onto the seams.
- `fit(params, x, y)` is a minimal full-batch SGD loop with a logistic value
  loss (the scalar head is a win-probability logit); `save_params` /
  `load_params` move params through `.npz` artifacts, so a trained model
  needs nothing beyond jax to run.
