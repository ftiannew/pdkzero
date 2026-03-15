from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from pdkzero.dmc.models import CandidateScoringModel
from pdkzero.dmc.selfplay import EpisodeSample


def build_batch(samples: Sequence[EpisodeSample]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("cannot build a batch from zero samples")
    return {
        "x": torch.from_numpy(np.stack([sample.x for sample in samples])).float(),
        "z": torch.from_numpy(np.stack([sample.z for sample in samples])).float(),
        "target": torch.tensor([sample.target for sample in samples], dtype=torch.float32),
    }


def train_step(
    model: CandidateScoringModel,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad()
    x = batch["x"].to(device)
    z = batch["z"].to(device)
    target = batch["target"].to(device)

    predicted_scores = model(z, x)
    value_loss = torch.nn.functional.mse_loss(predicted_scores, target)
    policy_loss = value_loss.detach() * 0.0
    loss = value_loss
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
    }
