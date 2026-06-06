# catan-agents

Catan-playing agents for [catan-engine](../catan-engine).

An agent is a **policy**: a pure JAX function `(key, observation, flat_mask) -> flat action index`, `jit` / `vmap` compatible so it can drive whole batches of games on device. Policies consume the engine's flat action space and the acting player's partial observation; decode a chosen index with `catan_engine.env.flat_to_action`.

## Policies

- `random_policy` — uniform over the legal actions.
- `greedy_policy` — scripted priorities (city > settlement > dev card > road), pip-weighted placement and robber moves.

## Evaluation

`evaluate(policies, n_steps=...)` seats one policy per player over a batch of games and counts wins per seat:

```python
from catan_agents import evaluate, greedy_policy, random_policy

result = evaluate([greedy_policy, random_policy], n_steps=2_000)
print(result.wins, result.episodes)
```
