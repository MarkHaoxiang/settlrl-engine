"""The mctx embedding codec must round-trip a state exactly: the search is
bit-identical to storing the raw pytree only because nothing is lost."""

import jax
from catan_agents.search.mcts import _codec
from catan_engine.board.state import BoardState
from catan_engine.env import BatchedCatanEnv


def test_packed_state_roundtrips() -> None:
    env = BatchedCatanEnv(batch_size=2, seed=0, n_players=3)
    env.rollout(jax.random.key(0), 150)
    state = env.board[1]
    pack, unpack = _codec(jax.tree.map(lambda x: x[0], state))
    rt = unpack(pack(state))
    for f in BoardState._fields:
        a, b = getattr(rt, f), getattr(state, f)
        if f == "key":
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        assert bool((a == b).all()), f
