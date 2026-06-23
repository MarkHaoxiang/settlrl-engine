# Training config refactor: grouped configs + decomposed loop + hydra

Date: 2026-06-23
Status: design, pending implementation

## Problem

`learn()` (`settlrl_learn/training/loop.py`) is a 30-keyword flat surface that
conflates self-play, search, optimiser, replay, arena, teacher, and value-blend
concerns. `experiments/0004_alphazero/run.py` mirrors that flatness with a
`VARIANTS` dict-of-dicts and a flat `AlphaZeroConfig`, then re-threads every knob
through two long `learn(...)` call sites. The flat surface is hard to test in
isolation (e.g. the value-blend formula is only reachable by running a full
`learn`), hard to compose (every variant restates the whole config), and easy to
mis-wire.

Goal, in order: (1) group the config into typed, independently-validatable units;
(2) decompose the loop body into separately-testable steps; (3) compose configs
with hydra's config groups. Pydantic stays the source of truth and validation
boundary throughout.

## Decisions (settled in brainstorming)

- **Pydantic-first, hydra layered on.** Grouped pydantic models are the contract
  and the validation boundary. Hydra only composes/overrides, feeding a
  `DictConfig` into pydantic validation (the pattern `Config.resolve` already
  uses with OmegaConf).
- **Config + loop decomposition** (not config-only).
- **Use the hydra library** via its **Compose API**, not `@hydra.main`. The repo
  rejected `@hydra.main` because its cwd takeover and output-dir creation fight
  `start_run`'s run-dir/manifest ownership (`experiment/config.py`,
  `experiments/CLAUDE.md`). The Compose API (`initialize_config_dir` + `compose`)
  has neither side effect, so the objection is fully sidestepped while we still
  get config groups, defaults lists, and group-level overrides. `hydra-core` is
  already a declared dev dependency (`pyproject.toml:27`), currently unused.
- **Restructure `AlphaZeroConfig` into the same nested groups** (not a flat
  config + adapter).

## A. Config architecture

New `settlrl_learn/training/config.py` — nested pydantic models owned by the
loop (its contract). Each is independently constructible and validatable; the
base carries `extra="forbid"` so a typo'd knob fails loudly.

- **`SearchSettings`** — reuse settlrl-search's existing pydantic `SearchConfig`
  (already pure scalars: `num_simulations`, `max_depth`, `max_considered`,
  `value_scale`, `expected_rolls`, `chance_nodes`, `dev_chance`, `ordered`). No
  duplication; one import. (Training currently hardcodes `value_scale=2.0` and
  does not expose `max_depth` — both become explicit config with those defaults.)
- **`SelfPlayConfig`** — `samples`, `batch`, `temperature`, `max_steps`,
  `max_game_len`.
- **`OptimConfig`** — `lr`, `weight_decay`, `batch_size`, `train_steps`, `reuse`.
- **`ReplayConfig`** — `buffer_max`, `buffer_min`.
- **`TeacherConfig`** — `enabled`, `iters`, `sims`. The value function itself
  (`heuristic_value`) stays code-supplied to `learn()`, not config (not
  serializable).
- **`ValueBlendConfig`** — `max`, `ramp`.
- **`EvalConfig`** — `eval_frac`.
- **`ArenaConfig`** — `games`, `every`, `batch`, `sims`, `considered`,
  `opponents: list[str]`. The chance/ordering *semantics* are inherited from the
  self-play `SearchSettings` (the net is trained under one regime, so the arena
  must match); the arena only overrides the sim/considered *budget*.
- **`LearnConfig`** — `n_iterations`, `seed`, `checkpoint_every`,
  `resume_from`, plus the sub-configs above.

`learn(backend, cfg: LearnConfig, *, on_iter=None, progress=False)` — **hard
cut** from the kwarg signature (no shim; few call sites).

## B. Loop decomposition

`learn()` keeps its current prologue (build the jitted/vmapped callables once;
`eqx.partition`/`combine` threading; resume/RNG scaffolding) and its
orchestration shape. The per-iteration body splits into pure, testable units in
a new `training/steps.py`:

- **`generate(...) -> Samples`** — today's `self_play` (already a standalone
  function; keep).
- **`prepare_targets(fresh, *, eval_frac, blend, iteration, seed) ->
  (train_samples, eval_slice, metrics)`** — the holdout split + value-blend
  `(1-α)z + α·q`, currently inline (`loop.py:213-227`). Pure function.
- **`train_epochs(net, opt_state, buffer, buf_state, step_fn, steps, key) ->
  (net, opt_state, metrics)`** — the inner update loop (`loop.py:249-257`).
- **`evaluate(backend, net, eval_slice) -> metrics`** — wraps `eval_metrics`.
- **`run_arena(backend, net, arena_cfg, seed) -> metrics`** — the arena calls
  (`loop.py:266-281`).

`learn()` becomes orchestration: `generate → prepare_targets → buffer.add →
train_epochs → evaluate → run_arena → checkpoint → on_iter`.

**Invariant the decomposition must not break:** the per-iteration RNG is a pure
function of `seed` and `i` (the offsets `seed+1+i`, `seed+10_000+i`,
`seed+20_000+i`, `seed+30_000+i`, `seed+50_000+i`). Bit-exact resume
(`test_learn_resume_bit_exact_{mlp,gnn}`) depends on this; the extracted steps
take the same derived keys, so resume stays bit-identical.

## C. Hydra composition

`settlrl_learn.experiment.Config` gains a sibling to `resolve`:

```python
@classmethod
def compose(cls, config_dir: str | Path, config_name: str,
            overrides: Sequence[str] = ()) -> Self:
    """Hydra-compose config groups under `config_dir`, validate into `cls`."""
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra_compose(config_name=config_name, overrides=list(overrides))
    return cls.model_validate(OmegaConf.to_container(cfg, resolve=True))
```

No cwd change, no output dir — `start_run` keeps owning the run dir + manifest.

`experiments/0004_alphazero/conf/` holds the groups:

- `config.yaml` — defaults list + top-level fields (`seed`, `n_iterations`,
  `checkpoint_every`, `resume_from`, gate, wandb).
- `net/{mlp,gnn}.yaml`, `search/{default,deep,chance}.yaml`,
  `selfplay/{default,wide}.yaml`, `optim/default.yaml`, `replay/default.yaml`,
  `teacher/{off,heuristic}.yaml`, `value_blend/{off,canopy}.yaml`,
  `eval/default.yaml`, `arena/{default,cheap}.yaml`.
- `experiment/*.yaml` — the hydra "experiment" pattern (`# @package _global_`,
  one file overriding several groups), reproducing today's `VARIANTS`
  (`gnn_run`, `gnn_warm`, `gnn_warm_qblend`, `gnn_warm_qblend_chance`,
  `gnn_overnight`, `gnn_smoke`, `smoke`). Selected with `+experiment=gnn_warm`.

CLI: `uv run python experiments/0004_alphazero/run.py +experiment=gnn_warm
search.num_simulations=128`. Multirun (`-m`) is a `@hydra.main` feature and is
out of scope; if wanted later it is a small loop over `compose`.

The shared harness (`Config.resolve`, `start_run`) is unchanged; experiments
0002/0003 keep using `resolve` with their `VARIANTS` dicts. Only 0004 moves to
the hydra `conf/` groups. `experiments/README.md` + `experiments/CLAUDE.md` note
the per-framework choice.

## D. AlphaZeroConfig restructure

`AlphaZeroConfig` becomes nested, mirroring the loop groups plus experiment-only
sections:

- loop groups: `search` (`SearchSettings`), `selfplay`, `optim`, `replay`,
  `teacher`, `value_blend`, `eval`, `arena` — reused directly.
- experiment-only: `net` (arch: `kind`/`width`/`depth`/`layers`/`preset`,
  `setup_depth`/`setup_temperature`/`setup_beam`), `wandb`
  (`mode`/`project`), `gate` (`winrate`), and top-level `seed`/`n_iterations`/
  `checkpoint_every`/`resume_from`.
- `AlphaZeroConfig.to_learn_config() -> LearnConfig` extracts the loop groups, so
  the two `learn(...)` call sites collapse to
  `learn(backend, cfg.to_learn_config(), on_iter=..., progress=True)`.

## E. Migration scope

1. `learn()` signature → `LearnConfig`.
2. `experiments/0004/run.py`: nested `AlphaZeroConfig` + `to_learn_config()`;
   `main()` reads via `Config.compose("conf", "config", overrides=sys.argv[1:])`;
   `VARIANTS` dict deleted (moves to `conf/experiment/*.yaml`).
3. `tests/test_training.py`: 6 `learn(...)`/resume call sites build a
   `LearnConfig` (a small `_learn_cfg()` helper replaces `_learn_kwargs()`).
4. New unit tests: `prepare_targets` (value-blend formula + holdout split,
   replacing `test_value_blend_formula_matches_loop`'s hand-reproduction of the
   loop math) and `train_epochs` (deterministic given seed).
5. Smoke: `experiments/tests/test_smoke.py` `test_0004_*` switches to composing
   the `smoke`/`gnn_smoke` experiment configs via hydra.

## Out of scope

- The evaluation/Elo work (point 2) — a separate design, after this lands; it
  will build on the new `ArenaConfig`/`run_arena` seam.
- Hydra multirun sweeps.
- Touching experiments 0002/0003.

## Risks

- **Resume bit-exactness** — the headline guarantee; covered by the two existing
  resume tests, which must stay green through the decomposition.
- **Hydra Compose API ergonomics** — `initialize_config_dir` wants an absolute
  path and is a context manager; the helper handles both. No global hydra state
  leaks because we never call `@hydra.main` or `GlobalHydra` outside the `with`.
- **Two config idioms in `experiments/`** (resolve vs compose) — accepted and
  documented; 0004 is the hydra pilot.
