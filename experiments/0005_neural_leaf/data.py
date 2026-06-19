"""Strong-teacher self-play dataset: board graphs -> seat-0 win/loss.

Roll out 2p games with a fixed teacher (a ``POLICIES`` belief agent) and
snapshot seat-0 board graphs (`board_sample`), each labelled with that game's
eventual seat-0 outcome. The label is therefore the value *of the teacher's
policy* at the position -- so a leaf fit to it and dropped into one-step
lookahead is one policy-improvement step over the teacher (the experiment's
point). Cached by config under ``runs/_cache``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from settlrl_agents import POLICIES
from settlrl_agents.evaluate import _picker
from settlrl_agents.policy import StatefulSpec
from settlrl_engine.env import BatchedSettlrlEnv, flat_to_action
from settlrl_learn.graph import Sample, board_sample

_CACHE = Path(__file__).resolve().parents[2] / "runs" / "_cache" / "0005"


class Dataset(NamedTuple):
    samples: Sample  # batched over a leading sample axis
    win: np.ndarray  # (n,) 0/1, seat-0 outcome
    episode: np.ndarray  # (n,) game index, for a leak-free by-episode split


def _key(cfg: dict) -> str:
    keys = ("agent", "n_samples", "snapshot_every", "batch_size", "seed")
    blob = json.dumps({k: cfg[k] for k in keys}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _collect(cfg: dict) -> Dataset:
    agent = POLICIES[cfg["agent"]]
    assert not isinstance(agent, StatefulSpec), "collector drives pure agents"
    bs = cfg["batch_size"]
    env = BatchedSettlrlEnv(
        batch_size=bs, seed=cfg["seed"], reward="sparse", n_players=2,
        track_beliefs=True,
    )  # fmt: skip
    pickers = [jax.jit(_picker(agent, 2, i)) for i in range(2)]
    feat = jax.jit(jax.vmap(lambda lo, st: board_sample(lo, st, jnp.int32(0))))

    buffers: list[list[Sample]] = [[] for _ in range(bs)]
    rows: list[Sample] = []
    wins: list[int] = []
    episodes: list[int] = []
    key = jax.random.key(cfg["seed"])
    n_ep = 0
    step = 0
    while len(rows) < cfg["n_samples"]:
        layout, state = env.board
        if step % cfg["snapshot_every"] == 0:
            s = jax.device_get(feat(layout, state))
            for lane in range(bs):
                buffers[lane].append(jax.tree.map(lambda x, lane=lane: x[lane], s))
        key, k = jax.random.split(key)
        seat_keys = jax.random.split(k, 2)
        sel = np.asarray(env.agent_selection)
        mask = env.flat_mask()
        flat = np.zeros((bs,), np.int32)
        for i in range(2):
            picks = np.asarray(
                pickers[i](seat_keys[i], layout, state, env.beliefs, mask)
            )
            flat[sel == i] = picks[sel == i]
        env.step(*flat_to_action(jnp.asarray(flat)))
        rewards = np.asarray(env.rewards)
        for lane in np.flatnonzero(np.asarray(env.terminations).any(axis=1)).tolist():
            won = int(rewards[lane, 0] > 0)
            for sample in buffers[lane]:
                rows.append(sample)
                wins.append(won)
                episodes.append(n_ep)
            n_ep += 1
            buffers[lane] = []
        step += 1

    samples = jax.tree.map(lambda *xs: np.stack(xs), *rows)
    return Dataset(
        samples=cast(Sample, samples),
        win=np.asarray(wins, np.float32),
        episode=np.asarray(episodes, np.int64),
    )


def generate(cfg: dict) -> Dataset:
    """Collect (or load from ``runs/_cache``) the dataset for ``cfg``."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / f"{_key(cfg)}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=False)
        samples = Sample(d["nodes"], d["edges"], d["glob"], d["engineered"])
        return Dataset(samples, d["win"], d["episode"])
    ds = _collect(cfg)
    np.savez(
        path,
        nodes=ds.samples.nodes, edges=ds.samples.edges, glob=ds.samples.glob,
        engineered=ds.samples.engineered, win=ds.win, episode=ds.episode,
    )  # fmt: skip
    return ds


def split(ds: Dataset, val_frac: float, seed: int) -> tuple[Dataset, Dataset]:
    """Leak-free split: whole games (episodes) go to train or val, never split."""
    eps = np.unique(ds.episode)
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    n_val = max(1, int(len(eps) * val_frac))
    val_eps = set(eps[:n_val].tolist())
    is_val = np.array([e in val_eps for e in ds.episode])

    def take(mask: np.ndarray) -> Dataset:
        return Dataset(
            jax.tree.map(lambda x: x[mask], ds.samples),
            ds.win[mask],
            ds.episode[mask],
        )

    return take(~is_val), take(is_val)
