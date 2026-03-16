from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pdkzero.agents.deep_agent import DeepAgent
from pdkzero.agents.heuristic_agent import HeuristicAgent
from pdkzero.agents.random_agent import RandomAgent
from pdkzero.dmc.models import CandidateScoringModel
from pdkzero.env.observation import ACTION_VECTOR_DIM, STATE_VECTOR_DIM
from pdkzero.simulation import average_scores, play_many_games


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate agents in Paodekuai")
    parser.add_argument("--games", type=int, default=8, help="Number of games to evaluate")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Print each move during evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (auto-detect latest if not specified)")
    parser.add_argument("--device", default="cpu", help="Device to run model on")
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(Path("checkpoints"))
        if checkpoint_path:
            print(f"[eval] Using latest checkpoint: {checkpoint_path}")

    if checkpoint_path:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        model = CandidateScoringModel(
            state_dim=checkpoint.get("state_dim", STATE_VECTOR_DIM),
            action_dim=checkpoint.get("action_dim", ACTION_VECTOR_DIM),
            hidden_dim=checkpoint.get("hidden_dim", 128),
            history_hidden_dim=checkpoint.get("history_hidden_dim", 64),
        )
        model.load_state_dict(checkpoint["state_dict"])
        p0_agent = DeepAgent(model, device=args.device)
    else:
        print("[eval] No checkpoint found, using HeuristicAgent for player 0")
        p0_agent = HeuristicAgent()

    agents = (
        p0_agent,
        HeuristicAgent(),
        HeuristicAgent(),
        HeuristicAgent(),
    )
    score_rows = play_many_games(agents=agents, games=args.games, seed=args.seed, verbose=args.verbose)
    result = {
        "games": args.games,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "average_scores": average_scores(score_rows),
        "scores": score_rows,
    }
    if argv is None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    import torch
    main()
