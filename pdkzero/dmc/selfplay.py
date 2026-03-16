from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import Event, Process, Queue, get_context
from random import Random
import queue as queue_module
import threading
from typing import Any

import numpy as np
import torch

from pdkzero.dmc.models import CandidateScoringModel
from pdkzero.env.observation import ACTION_VECTOR_DIM
from pdkzero.env.game_env import PaodekuaiEnv


@dataclass
class EpisodeSample:
    player: int
    state: np.ndarray | None = None
    history: np.ndarray | None = None
    candidates: np.ndarray | None = None
    chosen_index: int = 0
    mc_return: float = 0.0
    win_target: float = 0.0
    x: np.ndarray | None = None
    z: np.ndarray | None = None
    target: float = 0.0

    def __post_init__(self) -> None:
        if self.state is None and self.x is not None:
            self.state = np.array(self.x[:-ACTION_VECTOR_DIM], copy=True)
        if self.history is None and self.z is not None:
            self.history = np.array(self.z, copy=True)
        if self.candidates is None and self.x is not None:
            chosen_candidate = np.array(self.x[-ACTION_VECTOR_DIM :], copy=True)
            self.candidates = chosen_candidate[None, :]
        if self.target != 0.0 and self.mc_return == 0.0:
            self.mc_return = float(self.target)
        if self.x is None and self.state is not None and self.candidates is not None:
            self.x = np.concatenate((self.state, self.candidates[self.chosen_index]), axis=0)
        if self.z is None and self.history is not None:
            self.z = np.array(self.history, copy=True)


@dataclass
class EpisodeResult:
    samples: list[EpisodeSample]
    scores: dict[int, int]


@dataclass(frozen=True)
class ActorSnapshot:
    state_dict: dict[str, torch.Tensor]
    state_dim: int
    action_dim: int
    hidden_dim: int
    history_hidden_dim: int


@dataclass
class ActorRuntime:
    processes: list[Any]
    result_queue: Any
    control_queues: list[Any]
    stop_event: Any
    backend: str


def generate_episode(
    model: CandidateScoringModel,
    seed: int | None = None,
    epsilon: float = 0.1,
    device: str = "cpu",
) -> EpisodeResult:
    rng = Random(seed)
    env = PaodekuaiEnv(seed=seed)
    obs = env.observe()
    samples: list[EpisodeSample] = []

    model.eval()
    while True:
        # 1. 模型选择动作
        action_index = _select_action_index(model, obs, rng, epsilon, device)
        # 2. 保存局面和选择的动作
        samples.append(
            EpisodeSample(
                player=int(obs["position"]),
                state=np.array(obs["x_no_action"], copy=True),
                history=np.array(obs["z"], copy=True),
                candidates=np.array(obs["action_batch"], copy=True),
                chosen_index=action_index,
            )
        )
        # 3. 执行动作
        next_obs, _, done, info = env.step(obs["legal_actions"][action_index])
        if done:
            # 4. 游戏结束，得到最终分数
            scores = info["scores"]
            for sample in samples:
                # 用最终分数作为 target！
                sample.mc_return = float(scores[sample.player])
                sample.target = sample.mc_return
                sample.win_target = 1.0 if scores[sample.player] > 0 else 0.0
            return EpisodeResult(samples=samples, scores=scores)
        assert next_obs is not None
        obs = next_obs


def _select_action_index(
    model: CandidateScoringModel,
    obs: dict,
    rng: Random,
    epsilon: float,
    device: str,
) -> int:
    num_actions = len(obs["legal_actions"])
    if num_actions == 1:
        return 0
    # epsilon-greedy 探索
    if rng.random() < epsilon:
        return rng.randrange(num_actions)

    # 模型对所有候选动作打分
    with torch.no_grad():
        x_batch = torch.tensor(obs["x_batch"], dtype=torch.float32, device=device)
        z_batch = torch.tensor(obs["z_batch"], dtype=torch.float32, device=device)
        scores = model(z_batch, x_batch)
    # 选择分数最高的动作
    return int(torch.argmax(scores).item())


def snapshot_actor_model(model: CandidateScoringModel) -> ActorSnapshot:
    return ActorSnapshot(
        state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        state_dim=model.state_dim,
        action_dim=model.action_dim,
        hidden_dim=model.hidden_dim,
        history_hidden_dim=model.history_hidden_dim,
    )


def start_actor_workers(
    snapshot: ActorSnapshot,
    num_workers: int,
    epsilon: float,
    base_seed: int | None,
    *,
    max_episodes_per_actor: int | None = None,
    queue_size: int = 64,
) -> ActorRuntime:
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    try:
        return _start_process_actor_workers(
            snapshot=snapshot,
            num_workers=num_workers,
            epsilon=epsilon,
            base_seed=base_seed,
            max_episodes_per_actor=max_episodes_per_actor,
            queue_size=queue_size,
        )
    except (OSError, PermissionError):
        return _start_thread_actor_workers(
            snapshot=snapshot,
            num_workers=num_workers,
            epsilon=epsilon,
            base_seed=base_seed,
            max_episodes_per_actor=max_episodes_per_actor,
            queue_size=queue_size,
        )


def publish_actor_snapshot(runtime: ActorRuntime, snapshot: ActorSnapshot) -> None:
    for control_queue in runtime.control_queues:
        control_queue.put(snapshot)


def stop_actor_workers(runtime: ActorRuntime) -> None:
    runtime.stop_event.set()
    for control_queue in runtime.control_queues:
        try:
            control_queue.put_nowait(None)
        except Exception:
            pass
    for process in runtime.processes:
        process.join(timeout=5)
        if hasattr(process, "is_alive") and process.is_alive() and hasattr(process, "terminate"):
            process.terminate()
            process.join(timeout=1)


def _start_process_actor_workers(
    *,
    snapshot: ActorSnapshot,
    num_workers: int,
    epsilon: float,
    base_seed: int | None,
    max_episodes_per_actor: int | None,
    queue_size: int,
) -> ActorRuntime:
    ctx = get_context("spawn")
    result_queue: Queue = ctx.Queue(maxsize=queue_size)
    control_queues: list[Queue] = []
    stop_event = ctx.Event()
    processes: list[Process] = []
    for worker_index in range(num_workers):
        control_queue: Queue = ctx.Queue()
        process = ctx.Process(
            target=_actor_worker_loop,
            args=(
                snapshot,
                result_queue,
                control_queue,
                stop_event,
                worker_index,
                epsilon,
                base_seed,
                max_episodes_per_actor,
            ),
        )
        process.daemon = True
        process.start()
        control_queues.append(control_queue)
        processes.append(process)
    return ActorRuntime(
        processes=processes,
        result_queue=result_queue,
        control_queues=control_queues,
        stop_event=stop_event,
        backend="process",
    )


def _start_thread_actor_workers(
    *,
    snapshot: ActorSnapshot,
    num_workers: int,
    epsilon: float,
    base_seed: int | None,
    max_episodes_per_actor: int | None,
    queue_size: int,
) -> ActorRuntime:
    result_queue: queue_module.Queue = queue_module.Queue(maxsize=queue_size)
    control_queues: list[queue_module.Queue] = []
    stop_event = threading.Event()
    processes: list[threading.Thread] = []
    for worker_index in range(num_workers):
        control_queue: queue_module.Queue = queue_module.Queue()
        thread = threading.Thread(
            target=_actor_worker_loop,
            args=(
                snapshot,
                result_queue,
                control_queue,
                stop_event,
                worker_index,
                epsilon,
                base_seed,
                max_episodes_per_actor,
            ),
            daemon=True,
        )
        thread.start()
        control_queues.append(control_queue)
        processes.append(thread)
    return ActorRuntime(
        processes=processes,
        result_queue=result_queue,
        control_queues=control_queues,
        stop_event=stop_event,
        backend="thread",
    )


def _actor_worker_loop(
    snapshot: ActorSnapshot,
    result_queue: Queue,
    control_queue: Queue,
    stop_event: Event,
    worker_index: int,
    epsilon: float,
    base_seed: int | None,
    max_episodes_per_actor: int | None,
) -> None:
    model = CandidateScoringModel(
        state_dim=snapshot.state_dim,
        action_dim=snapshot.action_dim,
        hidden_dim=snapshot.hidden_dim,
        history_hidden_dim=snapshot.history_hidden_dim,
    )
    model.load_state_dict(snapshot.state_dict)
    model.eval()
    episode_count = 0
    while not stop_event.is_set():
        try:
            while True:
                maybe_snapshot = control_queue.get_nowait()
                if maybe_snapshot is None:
                    stop_event.set()
                    break
                model = CandidateScoringModel(
                    state_dim=maybe_snapshot.state_dim,
                    action_dim=maybe_snapshot.action_dim,
                    hidden_dim=maybe_snapshot.hidden_dim,
                    history_hidden_dim=maybe_snapshot.history_hidden_dim,
                )
                model.load_state_dict(maybe_snapshot.state_dict)
                model.eval()
        except queue_module.Empty:
            pass

        if stop_event.is_set():
            break

        seed = None
        if base_seed is not None:
            seed = base_seed + worker_index * 100_000 + episode_count
        episode = generate_episode(model=model, seed=seed, epsilon=epsilon, device="cpu")
        result_queue.put(episode)
        episode_count += 1
        if max_episodes_per_actor is not None and episode_count >= max_episodes_per_actor:
            break


def create_episode_executor(num_workers: int) -> ProcessPoolExecutor | None:
    if num_workers <= 1:
        return None
    try:
        return ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context("spawn"))
    except (OSError, PermissionError):
        return None


def generate_episode_batch(
    model: CandidateScoringModel,
    seeds: list[int | None],
    epsilon: float,
    num_workers: int,
    executor: ProcessPoolExecutor | None = None,
) -> list[EpisodeResult]:
    if not seeds:
        return []
    if num_workers <= 1:
        return [generate_episode(model=model, seed=seed, epsilon=epsilon, device="cpu") for seed in seeds]

    snapshot = snapshot_actor_model(model)
    if executor is None:
        created_executor = create_episode_executor(num_workers)
        if created_executor is None:
            return [generate_episode(model=model, seed=seed, epsilon=epsilon, device="cpu") for seed in seeds]
        with created_executor:
            return list(
                created_executor.map(
                    _generate_episode_from_snapshot,
                    [snapshot] * len(seeds),
                    seeds,
                    [epsilon] * len(seeds),
                )
            )

    return list(
        executor.map(
            _generate_episode_from_snapshot,
            [snapshot] * len(seeds),
            seeds,
            [epsilon] * len(seeds),
        )
    )


def _generate_episode_from_snapshot(
    snapshot: ActorSnapshot,
    seed: int | None,
    epsilon: float,
) -> EpisodeResult:
    model = CandidateScoringModel(
        state_dim=snapshot.state_dim,
        action_dim=snapshot.action_dim,
        hidden_dim=snapshot.hidden_dim,
        history_hidden_dim=snapshot.history_hidden_dim,
    )
    model.load_state_dict(snapshot.state_dict)
    model.eval()
    return generate_episode(model=model, seed=seed, epsilon=epsilon, device="cpu")
