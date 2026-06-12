"""The linear value-fitting framework (see experiments/CLAUDE.md).

One experiment = one ``CONFIG`` driving :func:`run_experiment`: pick the
feature subset (``BoardFeatures`` names) and the optimisation target —
``"predict"`` fits weights to predict game outcomes against a known opponent
(logistic and sign-constrained NNLS, on all positions and the early halves,
ranked by match probes); ``"maximise"`` searches weight space directly for
match win rate with a cross-entropy loop. Either way the winner deploys via
``value.make_linear`` into one-step lookahead, gets benched against the
opponent next to the hand-tuned baseline, and is gated head-to-head against
``lookahead(heuristic)``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from _lib import Run
from catan_agents import POLICIES
from catan_agents.evaluate import _picker, evaluate
from catan_agents.internal.feature_engineering import BoardFeatures, board_features
from catan_agents.policy import AgentSpec, BeliefSpec
from catan_agents.search.lookahead import make_greedy
from catan_agents.value import make_linear
from catan_engine.env import BatchedCatanEnv, flat_to_action
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

# The hand-tuned heuristic's nonzero-weight terms, with their weights — the
# default feature subset and the maximise target's starting point.
HAND_WEIGHTS = {
    "vp": 10.0, "production": 1.0, "diversity": 0.6, "hand": 0.3,
    "scarce": 1.0, "over": -0.4, "n_dev": 1.5, "best_spot": 0.5,
    "n_roads": 0.15, "progress": 2.0, "knights": 0.5, "wheat_ore": 0.25,
    "race": 0.8, "numbers": 0.3, "held_knights": 0.8,
}  # fmt: skip


def _spec(weights: dict[str, float]) -> BeliefSpec:
    return BeliefSpec(
        make_greedy, frozenset((2,)), defaults={"value": make_linear(weights)}
    )


def seat_swapped(
    spec_a: AgentSpec, spec_b: AgentSpec, n_games: int, seed: int
) -> tuple[int, int]:
    """(a's wins, episodes) over a seat-swapped 2p match."""
    r1 = evaluate([spec_a, spec_b], n_episodes=n_games // 2, batch_size=32, seed=seed)
    r2 = evaluate(
        [spec_b, spec_a], n_episodes=n_games // 2, batch_size=32, seed=seed + 1
    )
    return int(r1.wins[0]) + int(r2.wins[1]), r1.episodes + r2.episodes


def collect(
    run: Run, cfg: dict, spec_a: AgentSpec, spec_b: AgentSpec
) -> tuple[np.ndarray, ...]:
    """Positions from ``spec_a`` [seat 0] vs ``spec_b`` [seat 1].

    Rows are seat0 features minus seat1 features, labels seat0-won; episode
    ids group correlated rows, fractions locate each row within its game."""
    c = cfg["collect"]
    env = BatchedCatanEnv(
        batch_size=c["batch_size"], seed=cfg["seed"], reward="sparse",
        n_players=2, track_beliefs=True,
    )  # fmt: skip
    pickers = [
        jax.jit(_picker(spec_a, 2, 0)),
        jax.jit(_picker(spec_b, 2, 1)),
    ]
    feats = jax.jit(
        jax.vmap(
            lambda lo, st: jnp.stack(
                [
                    jnp.stack(board_features(lo, st, jnp.int32(p), jnp.bool_(True)))
                    for p in (0, 1)
                ]
            )
        )
    )
    key = jax.random.key(cfg["seed"])
    buffers: list[list[np.ndarray]] = [[] for _ in range(c["batch_size"])]
    xs: list[np.ndarray] = []
    ys: list[int] = []
    episodes: list[int] = []
    fracs: list[float] = []
    n_episodes = 0
    for step in range(c["steps"]):
        layout, state = env.board
        if step % c["snapshot_every"] == 0:
            f = np.asarray(feats(layout, state))  # (B, 2, n_features)
            for lane in range(c["batch_size"]):
                buffers[lane].append(f[lane, 0] - f[lane, 1])
        key, k = jax.random.split(key)
        seat_keys = jax.random.split(k, 2)
        sel = np.asarray(env.agent_selection)
        mask = env.flat_mask()
        flat = np.zeros((c["batch_size"],), np.int32)
        for i in (0, 1):
            picks = pickers[i](seat_keys[i], layout, state, env.beliefs, mask)
            flat[sel == i] = np.asarray(picks)[sel == i]
        env.step(*flat_to_action(jnp.asarray(flat)))
        rewards = np.asarray(env.rewards)
        for lane in np.flatnonzero(np.asarray(env.terminations).any(axis=1)):
            won = int(rewards[lane, 0] > 0)
            k_rows = len(buffers[lane])
            xs.extend(buffers[lane])
            ys.extend([won] * k_rows)
            episodes.extend([n_episodes] * k_rows)
            fracs.extend((i + 1) / k_rows for i in range(k_rows))
            n_episodes += 1
            buffers[lane] = []
        if step % 2000 == 0:
            run.log(step=step, positions=len(ys))
    return (
        np.asarray(xs, np.float64),
        np.asarray(ys, np.int64),
        np.asarray(episodes, np.int64),
        np.asarray(fracs, np.float64),
    )


def fit_predict(
    run: Run, cfg: dict, data: tuple[np.ndarray, ...]
) -> dict[str, dict[str, Any]]:
    """The predict target: fit the configured features to game outcomes.

    Candidates: {logistic, sign-constrained NNLS} x {all positions, early
    halves} (prediction redistributes correlated credit, constraints keep
    the decision gradient sane; early positions force economy to carry the
    signal). Reported AUC is held out by episode — rows within a game are
    correlated — but candidates must be *ranked by match probes*: AUC is
    flat across candidates whose probes differ by 25 points (exp 0002)."""
    x, y, episodes, fracs = data
    names = list(BoardFeatures._fields)
    terms = cfg["features"]
    cols = [names.index(t) for t in terms]
    flip = np.ones(len(cols))
    if "over" in terms:
        flip[terms.index("over")] = -1.0  # so the >=0 constraint means "penalty"
    train = episodes % 5 != 0  # ~80/20, grouped by episode
    out: dict[str, dict[str, Any]] = {}
    for stage, rows in (("all", np.ones_like(train)), ("early", fracs <= 0.5)):
        tr, te = train & rows, ~train & rows
        for method in ("logistic", "nnls"):
            xt = x[:, cols] * flip
            if method == "logistic":
                m = LogisticRegression(fit_intercept=False, max_iter=2000)
                m.fit(xt[tr], y[tr])
                coef, score = m.coef_[0], m.decision_function(xt[te])
            else:
                m = LinearRegression(fit_intercept=False, positive=True)
                m.fit(xt[tr], y[tr] * 2.0 - 1.0)
                coef, score = m.coef_, m.predict(xt[te])
            auc = float(roc_auc_score(y[te], score))
            weights = dict(zip(terms, (coef * flip).tolist(), strict=True))
            out[f"{stage}/{method}"] = {"auc": auc, "weights": weights}
            run.log(candidate=f"{stage}/{method}", auc=auc)
    return out


def probe_best(
    run: Run, cfg: dict, candidates: dict[str, dict[str, Any]], opponent: AgentSpec
) -> dict[str, float]:
    """Rank candidate weights by cheap seat-swapped matches vs the opponent."""
    probes: dict[str, float] = {}
    for label, cand in candidates.items():
        w, n = seat_swapped(_spec(cand["weights"]), opponent, cfg["probe_games"], 10)
        probes[label] = w / n
        run.log(candidate=label, probe_vs_opponent=w / n)
    return probes


def maximise(
    run: Run,
    cfg: dict,
    opponent: AgentSpec,
    init: dict[str, float],
    seed_offset: int = 0,
) -> dict[str, float]:
    """The maximise target: cross-entropy search over weight vectors, with the
    measured seat-swapped win rate vs ``opponent`` as the objective.

    Starts from ``init``; each generation shares an evaluation seed (common
    random numbers) and the seed changes across generations so the search
    cannot overfit one batch of boards."""
    m = cfg["maximise"]
    terms = cfg["features"]
    rng = np.random.default_rng(cfg["seed"] + seed_offset)
    mean = np.asarray([init.get(t, 0.0) for t in terms])
    sigma = np.maximum(np.abs(mean) * m["sigma"], 0.3)
    best: tuple[float, np.ndarray] = (-1.0, mean)
    for gen in range(m["iterations"]):
        pop = rng.normal(mean, sigma, size=(m["population"], len(terms)))
        rates = []
        for i, w_vec in enumerate(pop):
            weights = dict(zip(terms, w_vec.tolist(), strict=True))
            w, n = seat_swapped(
                _spec(weights), opponent, m["eval_games"], 100 + 7 * (gen + seed_offset)
            )
            rates.append(w / n)
            run.log(gen=gen, member=i, rate=w / n)
        order = np.argsort(rates)[::-1]
        elites = pop[order[: m["elites"]]]
        if rates[order[0]] > best[0]:
            best = (rates[order[0]], pop[order[0]])
        mean = elites.mean(axis=0)
        sigma = np.maximum(elites.std(axis=0) * 1.2, 0.05)
        run.log(
            gen=gen, best_rate=float(rates[order[0]]), mean_rate=float(np.mean(rates))
        )
    return dict(zip(terms, best[1].tolist(), strict=True))


def run_experiment(run: Run, cfg: dict) -> None:
    """Optimize for ``rounds`` iterations, then bench the champion and gate it
    against the hand-tuned lookahead (pass iff the lower 2-sigma bound clears 50%).

    ``opponent: "self"`` is the self-play ladder: every round optimizes
    against the *current champion* (round 0's champion is the hand weights),
    and a challenger replaces it only by winning the acceptance match. A
    ``POLICIES`` name keeps the fixed known opponent (warm-started across
    rounds)."""
    rounds = cfg.get("rounds", 1)
    self_play = cfg["opponent"] == "self"
    champion = {t: HAND_WEIGHTS.get(t, 0.0) for t in cfg["features"]}
    best_label = cfg["target"]
    for rnd in range(rounds):
        champ_spec = _spec(champion)
        opp_spec = champ_spec if self_play else POLICIES[cfg["opponent"]]
        if cfg["target"] == "predict":
            data = collect(run, cfg, champ_spec, opp_spec)
            run.log(
                round=rnd,
                positions=len(data[1]),
                win_rate_seat0=float(np.mean(data[1])),
            )
            candidates = fit_predict(run, cfg, data)
            run.save_json(f"fits_round{rnd}.json", candidates)
            probes = probe_best(run, cfg, candidates, opp_spec)
            best_label = max(probes, key=lambda k: probes[k])
            weights = candidates[best_label]["weights"]
        else:
            weights = maximise(run, cfg, opp_spec, init=champion, seed_offset=37 * rnd)
        w, n = seat_swapped(_spec(weights), champ_spec, cfg["probe_games"], 500 + rnd)
        accepted = w / n > 0.5
        run.log(round=rnd, challenger_vs_champion=w / n, accepted=accepted)
        if accepted:
            champion = weights
    weights = champion
    run.save_json("weights.json", weights)

    learned, hand = _spec(weights), POLICIES["lookahead"]
    opponent = POLICIES[cfg.get("bench_opponent", "greedy")]
    n = cfg["bench_games"]
    w_vs_opp, n_vs_opp = seat_swapped(learned, opponent, n, 30)
    w_base, n_base = seat_swapped(hand, opponent, n, 30)
    run.log(learned_vs_opponent=w_vs_opp / n_vs_opp, hand_vs_opponent=w_base / n_base)

    g = cfg["gate_games"]
    w_gate, n_gate = seat_swapped(learned, hand, g, 20)
    rate = w_gate / n_gate
    se = (rate * (1 - rate) / n_gate) ** 0.5
    run.finish(
        "pass" if rate - 2 * se > 0.5 else "fail",
        best_candidate=best_label,
        learned_vs_hand=rate,
        lower_2se=rate - 2 * se,
        learned_vs_opponent=w_vs_opp / n_vs_opp,
        hand_vs_opponent=w_base / n_base,
    )
