"""sklearn value weights vs greedy.

Hypothesis: logistic-regression weights fit on game outcomes against a known
opponent (greedy) recover or beat the hand-tuned heuristic weights when
deployed in one-step lookahead — and the pipeline (collect positions, fit a
chosen feature subset, deploy via make_linear, bench) is reusable for any
future feature set.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from catan_agents import POLICIES
from catan_agents.evaluate import _picker, evaluate
from catan_agents.internal.feature_engineering import BoardFeatures, board_features
from catan_agents.policy import BeliefSpec
from catan_agents.search.lookahead import make_greedy
from catan_agents.value import make_linear
from catan_engine.env import BatchedCatanEnv, flat_to_action
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import start_run

# The hand-tuned heuristic's nonzero-weight terms (its own feature subset).
HAND_TERMS = [
    "vp",
    "production",
    "diversity",
    "hand",
    "scarce",
    "over",
    "n_dev",
    "best_spot",
    "n_roads",
    "progress",
    "knights",
    "wheat_ore",
    "race",
    "numbers",
    "held_knights",
]

CONFIG = {
    "seed": 0,
    "opponent": "greedy",  # the known opponent (data and target)
    "batch_size": 64,
    "collect_steps": 12_000,
    "snapshot_every": 4,
    "subsets": {
        "all": list(BoardFeatures._fields),
        "hand_terms": HAND_TERMS,
        "compact": ["vp", "production", "progress", "race", "best_spot", "hand"],
    },
    "probe_games": 120,  # quick per-candidate match vs the opponent
    "bench_games": 200,
    "gate_games": 300,  # learned vs hand-tuned lookahead, the verdict match
}


def collect(run) -> tuple[np.ndarray, np.ndarray]:
    """Positions from lookahead(heuristic) [seat 0] vs the opponent [seat 1].

    Rows are seat0 features minus seat1 features, labels seat0-won; episode
    ids group correlated rows, fractions locate each row within its game."""
    env = BatchedCatanEnv(
        batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"],
        reward="sparse",
        n_players=2,
        track_beliefs=True,
    )
    pickers = [
        jax.jit(_picker(POLICIES["lookahead"], 2, 0)),
        jax.jit(_picker(POLICIES[CONFIG["opponent"]], 2, 1)),
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
    key = jax.random.key(CONFIG["seed"])
    buffers: list[list[np.ndarray]] = [[] for _ in range(CONFIG["batch_size"])]
    xs: list[np.ndarray] = []
    ys: list[int] = []
    episodes: list[int] = []
    fracs: list[float] = []
    n_episodes = 0
    for step in range(CONFIG["collect_steps"]):
        layout, state = env.board
        if step % CONFIG["snapshot_every"] == 0:
            f = np.asarray(feats(layout, state))  # (B, 2, n_features)
            for lane in range(CONFIG["batch_size"]):
                buffers[lane].append(f[lane, 0] - f[lane, 1])
        key, k = jax.random.split(key)
        seat_keys = jax.random.split(k, 2)
        sel = np.asarray(env.agent_selection)
        mask = env.flat_mask()
        flat = np.zeros((CONFIG["batch_size"],), np.int32)
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


def seat_swapped(spec_a, spec_b, n_games: int, seed: int) -> tuple[int, int]:
    """(a's wins, episodes) over a seat-swapped 2p match."""
    r1 = evaluate([spec_a, spec_b], n_episodes=n_games // 2, batch_size=32, seed=seed)
    r2 = evaluate(
        [spec_b, spec_a], n_episodes=n_games // 2, batch_size=32, seed=seed + 1
    )
    return int(r1.wins[0]) + int(r2.wins[1]), r1.episodes + r2.episodes


def fit_candidates(
    x: np.ndarray, y: np.ndarray, episodes: np.ndarray, fracs: np.ndarray
) -> dict[str, dict]:
    """The candidate matrix: per subset, a logistic fit and a sign-constrained
    NNLS fit (all coefficients >= 0, ``over`` negated first — prediction
    redistributes correlated credit, constraints keep the *decision* gradient
    sane), each on all positions and on the early half of each game (where
    the outcome signal must come from economy, not from VP already banked).
    Reported AUC is held out by episode (rows within a game are correlated).
    """
    names = list(BoardFeatures._fields)
    flip = np.ones(len(names))
    flip[names.index("over")] = -1.0  # so the >=0 constraint means "penalty"
    train = episodes % 5 != 0  # ~80/20, grouped by episode
    out: dict[str, dict] = {}
    for label, terms in CONFIG["subsets"].items():
        cols = [names.index(t) for t in terms]
        for stage, rows in (("all", np.ones_like(train)), ("early", fracs <= 0.5)):
            tr, te = train & rows, ~train & rows
            for method in ("logistic", "nnls"):
                xt = x[:, cols] * flip[cols]
                if method == "logistic":
                    m = LogisticRegression(fit_intercept=False, max_iter=2000)
                    m.fit(xt[tr], y[tr])
                    coef, score = m.coef_[0], m.decision_function(xt[te])
                else:
                    m = LinearRegression(fit_intercept=False, positive=True)
                    m.fit(xt[tr], y[tr] * 2.0 - 1.0)
                    coef, score = m.coef_, m.predict(xt[te])
                auc = float(roc_auc_score(y[te], score))
                weights = dict(zip(terms, (coef * flip[cols]).tolist(), strict=True))
                out[f"{label}/{stage}/{method}"] = {"auc": auc, "weights": weights}
    return out


def main() -> None:
    run = start_run(Path(__file__).parent, CONFIG)
    x, y, episodes, fracs = collect(run)
    run.log(
        positions=len(y),
        episodes=int(episodes.max()) + 1,
        win_rate_seat0=float(np.mean(y)),
    )

    fits = fit_candidates(x, y, episodes, fracs)
    run.save_json("fits.json", fits)
    for label, fit in fits.items():
        run.log(candidate=label, auc=fit["auc"])

    # Selection by cheap *matches* against the opponent, not by fit metrics:
    # an AUC ranks prediction, the probe ranks decisions.
    opponent = POLICIES[CONFIG["opponent"]]
    hand = POLICIES["lookahead"]
    probe_n = CONFIG["probe_games"]
    probes: dict[str, float] = {}
    for label, fit in sorted(fits.items(), key=lambda kv: -kv[1]["auc"])[:4]:
        spec = BeliefSpec(
            make_greedy,
            frozenset((2,)),
            defaults={"value": make_linear(fit["weights"])},
        )
        w, n = seat_swapped(spec, opponent, probe_n, 10)
        probes[label] = w / n
        run.log(candidate=label, probe_vs_opponent=w / n)
    run.save_json("probes.json", probes)

    best = max(probes, key=lambda k: probes[k])
    learned = BeliefSpec(
        make_greedy,
        frozenset((2,)),
        defaults={"value": make_linear(fits[best]["weights"])},
    )
    n = CONFIG["bench_games"]
    w_vs_opp, n_vs_opp = seat_swapped(learned, opponent, n, 30)
    w_base, n_base = seat_swapped(hand, opponent, n, 30)
    run.log(learned_vs_opponent=w_vs_opp / n_vs_opp, hand_vs_opponent=w_base / n_base)

    g = CONFIG["gate_games"]
    w_gate, n_gate = seat_swapped(learned, hand, g, 20)
    rate = w_gate / n_gate
    se = (rate * (1 - rate) / n_gate) ** 0.5
    run.log(learned_vs_hand=rate, se=se)

    verdict = "pass" if rate - 2 * se > 0.5 else "fail"
    run.finish(
        verdict,
        best_candidate=best,
        learned_vs_hand=rate,
        lower_2se=rate - 2 * se,
        learned_vs_opponent=w_vs_opp / n_vs_opp,
        hand_vs_opponent=w_base / n_base,
    )


if __name__ == "__main__":
    main()
