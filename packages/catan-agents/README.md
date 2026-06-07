# catan-agents

Catan-playing agents for [catan-engine](../catan-engine).

Agents are pure JAX functions (`jit` / `vmap` compatible, so they can drive whole batches of games on device) that consume the engine's flat action space; decode a chosen index with `catan_engine.env.flat_to_action`. They come in two kinds, split by what a seat consumes — neither sees anything the player wouldn't:

- **Observation agents** (`(key, observation, flat_mask) -> flat action`) read the acting player's partial view.
- **Belief agents** (`(key, layout, censored_state, belief, player, flat_mask) -> flat action`) read the engine's honest world model: a board with every hidden field removed, plus provable bounds on what opponents hold (the engine's belief tracking — `BatchedCatanEnv(track_beliefs=True)`). They rebuild a concrete world by sampling (`sample_world`) and search in the sample.

Both kinds work at any player count. `POLICIES` maps every shipped agent by name to an `AgentSpec` (its function, which kind it is, and the player counts it supports).

## Agents

- `random` — uniform over the legal actions.
- `greedy` — scripted priorities (city > settlement > dev card > road), pip-weighted placement and robber moves.
- `lookahead` — one-step lookahead: applies every legal action to a sampled world and picks the successor the value function scores best.
- `mcts` — Gumbel-MuZero tree search ([mctx](https://github.com/google-deepmind/mctx)) using the engine as its simulator and the value function at the leaves.

The search agents act on a sampled world consistent with everything the seat knows: stochastic outcomes are their own samples (not the environment's), opponents' hidden cards are dealt from the player's honest belief — they never act on information the seat could not know. With two players the belief pins the opponent's resources exactly, so only dev-card identities are ever sampled.

## Value functions

A `ValueFunction` scores a board for one player (higher is better). `heuristic_value` is the shipped hand-written one: victory points, pip-weighted production, hand diversity, and held dev cards, relative to the strongest opponent. `lookahead` and `mcts` are parameterised by it — plug your own into `make_greedy(value)` / `make_mcts(value)`.

## Evaluation

`evaluate(agents, n_steps=...)` seats one agent per player over a batch of games and counts wins per seat:

```python
from catan_agents import POLICIES, evaluate

result = evaluate([POLICIES["mcts"], POLICIES["greedy"]], n_steps=2_000)
print(result.wins, result.episodes)
```

The budget is either `n_steps` env steps or `n_episodes` finished games.

## CLI

`catan-agents compare <a> <b>` plays two agents head-to-head (100 games by default, seats swapped halfway to cancel the first-mover advantage):

```
$ catan-agents compare mcts greedy
mcts vs greedy: 53-47 over 100 games (mcts 53.0%)
  seated first: mcts 28/50, greedy 25/50
```

`--games`, `--batch-size`, and `--seed` adjust the run. Future tools (tournaments, ...) will hang off the same entry point as subcommands.

## Layout

- `catan_agents.shared` — the seat protocols, value functions, world sampling, observation baselines, and `evaluate`.
- `catan_agents.search` — the model-based agents (`lookahead`, `mcts`).
