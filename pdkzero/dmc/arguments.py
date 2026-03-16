from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Paodekuai candidate-scoring model")
    parser.add_argument("--device", default="cpu", help="Training device: cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument("--max-episodes", type=int, default=32, help="Number of self-play episodes")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel self-play worker count")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print training progress every N completed episodes",
    )
    parser.add_argument(
        "--save-interval-updates",
        type=int,
        default=5000,
        help="Write a step checkpoint every N optimizer updates",
    )
    parser.add_argument(
        "--actor-sync-interval",
        type=int,
        default=100,
        help="Publish latest learner weights to actors every N optimizer updates",
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=200_000,
        help="Maximum number of recent samples kept in replay",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for the MLP")
    parser.add_argument(
        "--history-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for the history LSTM",
    )
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration probability")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume-from", default=None, help="Optional checkpoint path to continue training from")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help="Run evaluation every N completed episodes; 0 disables periodic evaluation",
    )
    parser.add_argument("--num-eval-games", type=int, default=8, help="Evaluation games after training")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()
