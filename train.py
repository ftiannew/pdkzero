from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from random import Random
import time
from typing import Sequence

import torch

from pdkzero.agents.deep_agent import DeepAgent
from pdkzero.agents.heuristic_agent import HeuristicAgent
from pdkzero.dmc.arguments import build_parser
from pdkzero.dmc.models import CandidateScoringModel
from pdkzero.dmc.selfplay import (
    create_episode_executor,
    generate_episode,
    generate_episode_batch,
    publish_actor_snapshot,
    snapshot_actor_model,
    start_actor_workers,
    stop_actor_workers,
)
from pdkzero.dmc.trainer import build_batch, train_step
from pdkzero.env.observation import ACTION_VECTOR_DIM, STATE_VECTOR_DIM
from pdkzero.simulation import average_scores, play_many_games


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"CUDA requested via --device={device_name}, but torch.cuda.is_available() is False"
        )
    return device


def make_cpu_actor_model(model: CandidateScoringModel) -> CandidateScoringModel:
    actor_model = CandidateScoringModel(
        state_dim=model.state_dim,
        action_dim=model.action_dim,
        hidden_dim=model.hidden_dim,
        history_hidden_dim=model.history_hidden_dim,
    )
    sync_cpu_actor_model(model, actor_model)
    actor_model.eval()
    return actor_model


def sync_cpu_actor_model(
    learner_model: CandidateScoringModel, actor_model: CandidateScoringModel
) -> None:
    cpu_state = {key: value.detach().cpu() for key, value in learner_model.state_dict().items()}
    actor_model.load_state_dict(cpu_state)


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def load_checkpoint(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    return torch.load(Path(path), map_location="cpu")


def resolve_model_dims(args, checkpoint: dict | None) -> tuple[int, int, int, int]:
    if checkpoint is None:
        return STATE_VECTOR_DIM, ACTION_VECTOR_DIM, args.hidden_dim, args.history_hidden_dim
    return (
        int(checkpoint.get("state_dim", STATE_VECTOR_DIM)),
        int(checkpoint.get("action_dim", ACTION_VECTOR_DIM)),
        int(checkpoint.get("hidden_dim", args.hidden_dim)),
        int(checkpoint.get("history_hidden_dim", args.history_hidden_dim)),
    )


def save_checkpoint(
    path: Path,
    model: CandidateScoringModel,
    optimizer: torch.optim.Optimizer,
    update_count: int,
    episodes_completed: int,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
            "hidden_dim": model.hidden_dim,
            "history_hidden_dim": model.history_hidden_dim,
            "update_count": update_count,
            "episodes_completed": episodes_completed,
        },
        path,
    )


def compute_eval_win_rate(score_rows: list[dict[int, int]], seat: int = 0) -> float:
    if not score_rows:
        return 0.0
    wins = sum(1 for row in score_rows if row.get(seat, 0) > 0)
    return wins / len(score_rows)


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.actor_sync_interval < 1:
        raise ValueError("--actor-sync-interval must be >= 1")
    if args.replay_capacity < 1:
        raise ValueError("--replay-capacity must be >= 1")
    rng = Random(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.resume_from)
    state_dim, action_dim, hidden_dim, history_hidden_dim = resolve_model_dims(args, checkpoint)

    model = CandidateScoringModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        history_hidden_dim=history_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    update_count = 0
    episodes_completed = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            move_optimizer_to_device(optimizer, device)
        update_count = int(checkpoint.get("update_count", 0))
        episodes_completed = int(checkpoint.get("episodes_completed", 0))

    actor_model = make_cpu_actor_model(model)
    replay: list = []
    last_loss = 0.0
    loss_window: deque[float] = deque(maxlen=100)
    reward_window: deque[float] = deque(maxlen=100)
    steps_window: deque[int] = deque(maxlen=100)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    episode_executor = create_episode_executor(args.num_workers)
    actor_runtime = None
    actor_backend = "single"
    training_started_at = time.monotonic()
    try:
        if args.num_workers > 1:
            actor_runtime = start_actor_workers(
                snapshot_actor_model(actor_model),
                num_workers=args.num_workers,
                epsilon=args.epsilon,
                base_seed=args.seed,
            )
            actor_backend = actor_runtime.backend
        print(f"[train] actor_backend={actor_backend}")
        local_episode_count = 0
        while local_episode_count < args.max_episodes:
            if actor_runtime is not None:
                episodes = [actor_runtime.result_queue.get()]
            else:
                round_size = min(args.num_workers, args.max_episodes - local_episode_count)
                episode_seeds = [
                    None if args.seed is None else args.seed + episodes_completed + offset
                    for offset in range(round_size)
                ]
                if args.num_workers <= 1:
                    episodes = [
                        generate_episode(
                            model=actor_model,
                            seed=episode_seed,
                            epsilon=args.epsilon,
                            device="cpu",
                        )
                        for episode_seed in episode_seeds
                    ]
                else:
                    episodes = generate_episode_batch(
                        model=actor_model,
                        seeds=episode_seeds,
                        epsilon=args.epsilon,
                        num_workers=args.num_workers,
                        executor=episode_executor,
                    )

            for episode in episodes:
                replay.extend(episode.samples)
                if len(replay) > args.replay_capacity:
                    replay = replay[-args.replay_capacity :]
                reward_window.append(float(episode.scores.get(0, 0.0)))
                steps_window.append(len(episode.samples))
                if len(replay) >= args.batch_size:
                    batch_samples = rng.sample(replay, k=min(args.batch_size, len(replay)))
                    batch = build_batch(batch_samples)
                    step_metrics = train_step(model=model, optimizer=optimizer, batch=batch, device=device)
                    last_loss = (
                        step_metrics["loss"]
                        if isinstance(step_metrics, dict)
                        else float(step_metrics)
                    )
                    loss_window.append(last_loss)
                    update_count += 1
                    sync_cpu_actor_model(model, actor_model)
                    if actor_runtime is not None and update_count % args.actor_sync_interval == 0:
                        publish_actor_snapshot(actor_runtime, snapshot_actor_model(actor_model))
                    if args.save_interval_updates > 0 and update_count % args.save_interval_updates == 0:
                        save_checkpoint(
                            checkpoint_dir / f"step-{update_count}.pt",
                            model=model,
                            optimizer=optimizer,
                            update_count=update_count,
                            episodes_completed=episodes_completed + 1,
                        )

                local_episode_count += 1
                episodes_completed += 1
                if args.log_interval > 0 and local_episode_count % args.log_interval == 0:
                    elapsed_seconds = max(time.monotonic() - training_started_at, 1e-9)
                    episodes_per_sec = local_episode_count / elapsed_seconds
                    avg_loss_100 = (
                        sum(loss_window) / len(loss_window) if loss_window else 0.0
                    )
                    avg_reward_100 = (
                        sum(reward_window) / len(reward_window) if reward_window else 0.0
                    )
                    avg_steps_100 = (
                        sum(steps_window) / len(steps_window) if steps_window else 0.0
                    )
                    print(
                        f"[train] episode {local_episode_count}/{args.max_episodes} "
                        f"total={episodes_completed} samples={len(replay)} "
                        f"last_q_loss={last_loss:.4f} avg_q_loss_100={avg_loss_100:.4f} "
                        f"avg_reward_100={avg_reward_100:.4f} avg_steps_100={avg_steps_100:.4f} "
                        f"episodes_per_sec={episodes_per_sec:.2f}"
                    )
                if (
                    args.eval_interval > 0
                    and local_episode_count % args.eval_interval == 0
                    and args.num_eval_games > 0
                ):
                    evaluation_seed = (
                        None
                        if args.seed is None
                        else args.seed + episodes_completed + 10_000
                    )
                    evaluation_rows = play_many_games(
                        agents=(
                            DeepAgent(actor_model, device="cpu"),
                            HeuristicAgent(),
                            HeuristicAgent(),
                            HeuristicAgent(),
                        ),
                        games=args.num_eval_games,
                        seed=evaluation_seed,
                    )
                    eval_scores = average_scores(evaluation_rows)
                    eval_score = float(eval_scores.get(0, 0.0))
                    eval_win_rate = compute_eval_win_rate(evaluation_rows, seat=0)
                    print(
                        f"[eval] episode {local_episode_count}/{args.max_episodes} "
                        f"eval_score={eval_score:.4f} eval_win_rate={eval_win_rate:.4f}"
                    )
    finally:
        if actor_runtime is not None:
            stop_actor_workers(actor_runtime)
        if episode_executor is not None:
            episode_executor.shutdown(wait=True)

    checkpoint_path = checkpoint_dir / "model.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        update_count=update_count,
        episodes_completed=episodes_completed,
    )

    evaluation_seed = None if args.seed is None else args.seed + episodes_completed + 10_000
    evaluation_rows = play_many_games(
        agents=(DeepAgent(actor_model, device="cpu"), HeuristicAgent(), HeuristicAgent(), HeuristicAgent()),
        games=args.num_eval_games,
        seed=evaluation_seed,
    )
    result = {
        "episodes": episodes_completed,
        "updates": update_count,
        "samples": len(replay),
        "last_loss": last_loss,
        "actor_backend": actor_backend,
        "checkpoint": str(checkpoint_path),
        "evaluation": average_scores(evaluation_rows),
    }
    if argv is None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
