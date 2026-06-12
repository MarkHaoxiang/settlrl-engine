# catan-agents

Catan-playing agents for [catan-engine](../catan-engine).

Agents consume the engine's flat action space (decode a chosen index with `catan_engine.env.flat_to_action`) and come in three kinds, split by what a seat consumes and how it runs — none sees anything the player wouldn't:

- **Observation agents** (`(key, observation, flat_mask) -> flat action`) are pure JAX functions (`jit` / `vmap` compatible, so they drive whole batches of games on device) reading the acting player's partial view.
- **Belief agents** (`(key, layout, view, player, flat_mask) -> flat action`) are pure JAX functions reading the engine's honest world model: a `BeliefView` holding the public board fields plus provable bounds on what opponents hold (the engine's belief tracking — `BatchedCatanEnv(track_beliefs=True)`). The view has no field for anything hidden, so an agent can't even represent what the seat couldn't see; it rebuilds a concrete world by sampling (`sample_world`) and searches in the sample.
- **Stateful agents** (`agent.act(observation, flat_mask) -> flat action`) are plain-Python objects, one per seat per game, that keep state across their own moves — plans held over many turns, offers remembered as rejected. No value function or search: the decision logic is written directly in code. `evaluate` seats them through a per-step Python driver instead of the fused scan.

All kinds work at any player count. `POLICIES` maps every shipped agent by name to an `AgentSpec` (its family, which kind it is, and the player counts it supports).

## Agents

- `random` — uniform over the legal actions.
- `greedy` — scripted priorities (city > settlement > dev card > road), pip-weighted placement and robber moves.
- `planner` — a stateful decision tree: adopts a build goal (city upgrade, road-path-plus-settlement expansion, dev buy), saves toward it across turns, and spends each turn closing its resource gap (dev-card plays, port trades, domestic offers it won't re-propose once rejected). Replans when the board invalidates the goal.
- `lookahead` — one-step lookahead: applies every legal action to a sampled world and picks the successor the value function scores best.
- `mcts` — Gumbel-MuZero tree search ([mctx](https://github.com/google-deepmind/mctx)) using the engine as its simulator, the value function at the leaves, and the one-step value sweep as its root prior; searches an ensemble of sampled worlds and averages their action weights.

The search agents act on a sampled world consistent with everything the seat knows: stochastic outcomes are their own samples (not the environment's), opponents' hidden cards are dealt from the player's honest belief — they never act on information the seat could not know. With two players the belief pins the opponent's resources exactly, so only dev-card identities are ever sampled.

Two-player strength (200+ game seat-swapped matches): `mcts` > `lookahead` > `greedy` > `random` — mcts beats lookahead 57%, lookahead beats greedy 90%, greedy beats random 85%. The ladder holds at every count (seat-rotated): one mcts wins 62% against two lookaheads at three players (chance 33%) and 36% against three at four players (chance 25%); one lookahead wins 77% against three greedies. `planner` slots between greedy and the search agents: 78% against greedy, 10% against lookahead (~100 / 60 games).

## Value functions

A `ValueFunction` scores a board for one player (higher is better). `heuristic_value` is the shipped hand-written one: victory points (weighted up superlinearly near the win), pip-weighted production (wheat and ore at a premium) and its diversity across resources, expansion (roads and reachable settlement spots), progress toward the next build, hand quality with a discard-risk penalty, dev cards, and Largest Army progress — all relative to the strongest opponent. `make_heuristic(**weights)` builds a variant with your own weights; `lookahead` and `mcts` accept any value function via `make_greedy(value)` / `make_mcts(value)`.

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

`--games`, `--batch-size`, and `--seed` adjust the run.

`catan-agents bench <a> <b>` is the experiment harness: each side is a registry name or a JSON spec configuring a family — builder knobs under `"params"`, heuristic weight overrides under `"value"`. Two-player runs swap seats; `--players 3|4` rotates agent `a` through every seat against a table of `b`:

```
$ catan-agents bench '{"kind": "mcts", "params": {"num_simulations": 64}}' lookahead --games 200
a=113 b=87 n=200 rate=56.5% se=3.5% (chance 50.0%)
  a seated at seat 0: 59/100, seat 1: 54/100
```

Future tools (tournaments, ...) will hang off the same entry point as subcommands.

## Layout

- `catan_agents.shared` — the seat protocols, value functions, world sampling, observation baselines, and `evaluate`.
- `catan_agents.search` — the model-based agents (`lookahead`, `mcts`).
- `catan_agents.planner` — the stateful decision-tree toolkit (nodes, plans, the numpy point of view) and the `planner` agent built on it.
