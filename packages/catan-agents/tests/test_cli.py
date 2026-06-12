"""Tests for the ``catan-agents`` CLI and the episode budget behind it."""

import pytest
from catan_agents import POLICIES, evaluate
from catan_agents.cli import compare, main


def test_evaluate_requires_exactly_one_budget() -> None:
    agents = [POLICIES["random"], POLICIES["random"]]
    with pytest.raises(ValueError, match="exactly one"):
        evaluate(agents)
    with pytest.raises(ValueError, match="exactly one"):
        evaluate(agents, n_steps=10, n_episodes=10)


def test_evaluate_episode_budget_completes_games() -> None:
    result = evaluate(
        [POLICIES["random"], POLICIES["random"]], n_episodes=3, batch_size=8
    )
    assert result.episodes >= 3
    assert result.episodes == int(result.wins.sum())


def test_compare_counts_and_attributes_wins() -> None:
    result = compare("random", "greedy", n_games=4, batch_size=8, seed=0)
    assert result.names == ("random", "greedy")
    assert result.episodes >= 4
    assert sum(result.wins) == result.episodes
    assert sum(result.first_seat_episodes) == result.episodes
    # Per-seat wins are a subset of each agent's total.
    assert result.first_seat_wins[0] <= result.wins[0]
    assert result.first_seat_wins[1] <= result.wins[1]


def test_compare_rejects_unknown_agent() -> None:
    with pytest.raises(ValueError, match="unknown agent"):
        compare("random", "clever")


def test_main_prints_a_result(capsys: pytest.CaptureFixture[str]) -> None:
    main(["compare", "random", "greedy", "--games", "2", "--batch-size", "8"])
    out = capsys.readouterr().out
    assert "random vs greedy" in out
    assert "seated first" in out


def test_build_spec_accepts_names_and_json() -> None:
    from catan_agents.cli import build_spec

    assert build_spec("random") is POLICIES["random"]
    spec = build_spec('{"kind": "mcts", "params": {"num_simulations": 8}}')
    assert spec.defaults["num_simulations"] == 8
    weighted = build_spec('{"kind": "lookahead", "value": {"w_vp": 5.0}}')
    assert weighted.defaults["value"] is not POLICIES["lookahead"].defaults["value"]
    with pytest.raises(ValueError, match="unknown agent"):
        build_spec("clever")
    with pytest.raises(ValueError, match="unknown agent kind"):
        build_spec('{"kind": "clever"}')


def test_bench_two_player_seat_swaps() -> None:
    from catan_agents.cli import bench

    result = bench("random", "greedy", n_games=4, players=2, batch_size=8, seed=0)
    assert result.episodes >= 4
    assert result.wins_a + result.wins_b == result.episodes
    assert len(result.by_position) == 2
    assert sum(n for _, n in result.by_position) == result.episodes


def test_bench_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    import json

    from catan_agents.cli import main

    main(["bench", "random", "random", "--games", "2", "--batch-size", "8", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["wins_a"] + doc["wins_b"] == doc["episodes"]
    assert doc["players"] == 2


def test_bench_rotates_three_player_seats(capsys: pytest.CaptureFixture[str]) -> None:
    from catan_agents.cli import main

    main(
        [
            "bench",
            "random",
            "random",
            "--games",
            "3",
            "--players",
            "3",
            "--batch-size",
            "8",
        ]
    )
    out = capsys.readouterr().out
    assert "chance 33.3%" in out
    assert "seat 2" in out
