from __future__ import annotations

import argparse
import json
from typing import Sequence

from pdkzero.simulation import default_heuristic_table, play_many_games


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play local Paodekuai self-play games")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    score_rows = play_many_games(default_heuristic_table(), games=args.games, seed=args.seed)
    result = {"games": args.games, "scores": score_rows}
    if argv is None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
