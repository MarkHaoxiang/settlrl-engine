# settlrl-agents

Agents that play the game — heuristics and search — plus a CLI for matches and
benchmarks, and the bot service that serves their moves over HTTP.

## Policy

::: settlrl_agents.policy

## Value

::: settlrl_agents.value

## Greedy

::: settlrl_agents.greedy

## Baselines

::: settlrl_agents.baselines

## Search

::: settlrl_agents.search

## Planner

::: settlrl_agents.planner

## Evaluation

::: settlrl_agents.evaluate

## Bot service

The FastAPI app serving a single bot over the shared wire protocol; implement a
`Bot` (`settlrl_agents.service.sdk`) to stand up a new service.

::: settlrl_agents.service
