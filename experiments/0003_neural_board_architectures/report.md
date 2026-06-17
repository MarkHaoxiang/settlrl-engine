# 0003 — neural board architectures

Status: open (framework live; first sweep concluded)

## Hypothesis

On supervised board-prediction tasks, a structure-aware net over the *raw* board
graph (GNN) is competitive with an MLP over the hand-tuned feature vector, and
clearly beats a structure-blind net over the same raw inputs. If raw-board
learning reaches hand-engineered level, the graph representation is the seam to
push for a learned value (settlrl-learn Stage 1).

## Setup

`uv run python experiments/0003_neural_board_architectures/run.py [variant] [k=v]`
— config schema and variants at the top of `run.py`.

Data: greedy 2p self-play, seat-0 positions on the *true* board, cached under
`runs/_cache`. Two labels per position: `heur` (the hand-tuned
`heuristic_value`) and `win` (seat 0's game outcome). Held-out split grouped by
episode. Four architectures, all reading the same position, same width 64:

- `mlp_engineered` — MLP over the 118-d hand-tuned feature vector (baseline);
- `mlp_flat` — MLP over the flattened graph (node features in fixed vertex
  order + globals), structure-blind;
- `deepset` — permutation-invariant mean-pool over per-vertex features +
  globals (set; no edges);
- `gnn` — jraph `GraphNetwork` message passing over the board graph (54
  vertices, 72 edges, both directions) + global readout.

Stack: equinox (models), optax adamw (training), jraph (message passing),
wandb (logging, `mode` configurable), equinox checkpointing of the best-val
model. Node/edge/global features standardized on the train split.

## Results

12,000 positions, 40 epochs, width 64, seed 0 (RTX 5090).

Heuristic regression — held-out R² (`runs/.../2026-06-17T003236Z`):

| arch | R² |
| --- | --- |
| mlp_engineered | **0.996** |
| gnn | 0.978 |
| deepset | 0.951 |
| mlp_flat | 0.538 |

Win prediction — held-out AUC (`runs/.../2026-06-17T003343Z`):

| arch | AUC |
| --- | --- |
| mlp_engineered | **0.834** |
| gnn | 0.825 |
| deepset | 0.789 |
| mlp_flat | 0.519 |

The two tasks agree:

1. **Structure is what makes raw board features usable.** The flat MLP — same
   raw inputs, fixed vertex order — is near-useless (R² 0.54; AUC 0.52 ≈
   chance). Permutation-invariant pooling (DeepSet) and message passing (GNN)
   recover most of the signal from identical inputs.
2. **Edges/topology add a little over a pure set.** GNN > DeepSet on both tasks
   (R² 0.978 vs 0.951; AUC 0.825 vs 0.789).
3. **The GNN nearly matches hand engineering from the raw board** — within
   0.018 R² and 0.009 AUC of the engineered-feature MLP, with no hand-crafted
   features beyond per-vertex production/ports/ownership. The engineered MLP's
   near-perfect heuristic R² is expected (it is fed exactly the heuristic's own
   inputs) and is the ceiling for that task, not a fair strength signal.

## Decision

Framework adopted; architecture not yet promoted to a shipped value. The
hypothesis holds — a GNN over the raw board is competitive with hand
engineering and dominates structure-blind baselines — so the graph
representation is the right seam for settlrl-learn Stage 1. Before promoting:
close the last gap to (and ideally past) the engineered baseline on `win` with
more data / capacity / tuning, then gate a `lookahead(gnn_value)` through
`settlrl-agents bench` per the staged plan. The leaf-is-the-ceiling finding in
`settlrl-agents`'s search notes is the reason this matters: search only starts
paying once the leaf improves, and this shows a learnable leaf is within reach.

Open levers: deeper/wider GNN, more samples, lookahead-agent data (greedy
positions may be off the strong-play manifold), and node-level targets (per-spot
value) the global readout currently discards.
