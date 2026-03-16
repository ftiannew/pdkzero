from __future__ import annotations

import argparse
import json
from typing import Sequence

from pdkzero.agents.heuristic_agent import HeuristicAgent
from pdkzero.agents.random_agent import RandomAgent
from pdkzero.simulation import average_scores, play_many_games


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a simple heuristic-vs-random Paodekuai table")
    parser.add_argument("--games", type=int, default=8, help="Number of games to evaluate")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Print each move during evaluation")
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    agent_seeds = (
        (None, None, None)
        if args.seed is None
        else (args.seed + 1, args.seed + 2, args.seed + 3)
    )
    agents = (
        HeuristicAgent(),
        RandomAgent(seed=agent_seeds[0]),
        RandomAgent(seed=agent_seeds[1]),
        RandomAgent(seed=agent_seeds[2]),
    )
    score_rows = play_many_games(agents=agents, games=args.games, seed=args.seed, verbose=args.verbose)
    result = {
        "games": args.games,
        "average_scores": average_scores(score_rows),
        "scores": score_rows,
    }
    if argv is None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
