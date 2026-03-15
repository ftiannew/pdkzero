from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        out += residual
        return self.relu(out)


class CandidateScoringModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        history_hidden_dim: int = 64,
        num_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.history_hidden_dim = history_hidden_dim
        self.num_res_blocks = num_res_blocks

        self.history_encoder = nn.LSTM(action_dim, history_hidden_dim, batch_first=True)

        # Residual blocks
        res_block_dim = state_dim + history_hidden_dim
        self.res_blocks = nn.Sequential(
            *[ResBlock(res_block_dim) for _ in range(num_res_blocks)]
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim + history_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_state(
        self, history_batch: torch.Tensor, state_batch: torch.Tensor
    ) -> torch.Tensor:
        _, (hidden_state, _) = self.history_encoder(history_batch)
        history_embedding = hidden_state[-1]
        state_features = torch.cat((state_batch, history_embedding), dim=1)
        # Pass through residual blocks
        res_features = self.res_blocks(state_features)
        return self.state_encoder(res_features)

    def forward(
        self,
        z_batch: torch.Tensor,
        x_batch: torch.Tensor,
    ) -> torch.Tensor:
        if z_batch is None or x_batch is None:
            raise TypeError("forward requires (z_batch, x_batch)")

        if x_batch.shape[1] != self.state_dim + self.action_dim:
            raise ValueError(
                f"expected x_batch width {self.state_dim + self.action_dim}, got {x_batch.shape[1]}"
            )
        state_batch = x_batch[:, : self.state_dim]
        candidate_batch = x_batch[:, self.state_dim :]
        encoded_state = self.encode_state(
            history_batch=z_batch, state_batch=state_batch
        )
        score_features = torch.cat((encoded_state, candidate_batch), dim=1)
        return self.score_head(score_features).squeeze(-1)
