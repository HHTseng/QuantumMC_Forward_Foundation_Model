"""A categorical gate: f_s:R^d -> simplex(C), conditional on T=1.

eta_FD = softmax(f_s)_FD; eta_any = 1-softmax(f_s)_unreconstructed.
This construction guarantees 0 <= eta_FD <= eta_any <= 1.
"""

from __future__ import annotations

import torch
from torch import nn


class HadronEfficiencyNet(nn.Module):
    def __init__(self, n_continuous: int, n_outcomes: int = 5,
                 hidden_width: int = 256, hidden_layers: int = 4,
                 dropout: float = 0.03) -> None:
        super().__init__()
        self.architecture = dict(n_continuous=n_continuous, n_outcomes=n_outcomes,
                                 hidden_width=hidden_width, hidden_layers=hidden_layers,
                                 dropout=dropout)
        layers = []
        for j in range(hidden_layers):
            layers.extend([nn.Linear(n_continuous if j == 0 else hidden_width, hidden_width),
                           nn.SiLU(), nn.LayerNorm(hidden_width)])
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_width, n_outcomes))
        self.network = nn.Sequential(*layers)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, continuous: torch.Tensor) -> torch.Tensor:
        return self.network(continuous)

    def architecture_dict(self):
        return self.architecture.copy()
