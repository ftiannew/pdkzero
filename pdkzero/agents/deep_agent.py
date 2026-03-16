from __future__ import annotations

from random import Random

import torch

from pdkzero.dmc.models import CandidateScoringModel
from pdkzero.env.observation import build_observation
from pdkzero.game.engine import InfoSet
from pdkzero.game.move import Move


class DeepAgent:
    def __init__(
        self,
        model: CandidateScoringModel,
        epsilon: float = 0.0,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.epsilon = epsilon
        self.device = device
        self._rng = Random(seed)

    def act(self, infoset: InfoSet) -> Move:
        observation = build_observation(infoset)
        if len(observation["legal_actions"]) == 1:
            return observation["legal_actions"][0]
        if self._rng.random() < self.epsilon:
            index = self._rng.randrange(len(observation["legal_actions"]))
            return observation["legal_actions"][index]

        with torch.no_grad():
            x_batch = torch.tensor(observation["x_batch"], dtype=torch.float32, device=self.device)
            z_batch = torch.tensor(observation["z_batch"], dtype=torch.float32, device=self.device)
            scores = self.model(z_batch, x_batch)
        index = int(torch.argmax(scores).item())
        return observation["legal_actions"][index]
