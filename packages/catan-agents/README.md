# catan-agents

Catan-playing agents for [catan-engine](../catan-engine).

Agents are pure JAX functions (`jit` / `vmap` compatible, so they can drive whole batches of games on device) that consume the engine's flat action space; decode a chosen index with `catan_engine.env.flat_to_action`. They come in two kinds, split by what a seat may legitimately see:

- **Observation agents** (`(key, observation, flat_mask) -> flat action`) read the acting player's partial view and work at any player count.
- **State agents** (`(key, layout, state, player, flat_mask) -> flat action`) read the full board. With two players every resource flow is publicly inferable, so this is bookkeeping rather than cheating — these agents are offered in two-player games only.

`POLICIES` maps every shipped agent by name to an `AgentSpec` (its function, which kind it is, and the player counts it supports).

## Agents

- `random` — uniform over the legal actions (any count).
- `greedy` — scripted priorities (city > settlement > dev card > road), pip-weighted placement and robber moves (any count).
- `lookahead` — one-step lookahead: applies every legal action and picks the successor the value function scores best (two-player).
- `mcts` — Gumbel-MuZero tree search ([mctx](https://github.com/google-deepmind/mctx)) using the engine as its simulator and the value function at the leaves (two-player).

Both search agents determinize the state before searching: stochastic outcomes are their own samples (not the environment's), and the opponent's hidden dev cards are re-dealt from the deck distribution — they never act on information the seat could not know.

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

- `catan_agents.shared` — the seat protocols, value functions, observation baselines, and `evaluate`.
- `catan_agents.two_player` — the state agents (`lookahead`, `mcts`).
- `catan_agents.four_player` — home for partial-information agents (belief-state / determinization); currently empty.
