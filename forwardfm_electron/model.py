"""Neural Bernoulli and categorical heads for generated electrons."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ElectronEfficiencyOutput:
    trigger_logit: torch.Tensor
    outcome_logits: torch.Tensor


class ElectronEfficiencyNet(nn.Module):
    """Approximate eta_T(x_e)=P(T=1|x_e) from generated truth only.

    The second head estimates P(C_e=c|x_e). In the current dataset C_e is
    exactly FD for T=1 and unreconstructed for T=0, so it is a deliberately
    redundant baseline output that preserves a future multi-region interface.
    """

    def __init__(
        self,
        n_continuous: int,
        n_outcomes: int,
        hidden_width: int = 128,
        hidden_layers: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_continuous = n_continuous
        self.n_outcomes = n_outcomes
        layers: list[nn.Module] = []
        for layer_index in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(
                        n_continuous if layer_index == 0 else hidden_width,
                        hidden_width,
                    ),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_width),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.trigger_head = nn.Linear(hidden_width, 1)
        self.outcome_head = nn.Linear(hidden_width, n_outcomes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, continuous: torch.Tensor) -> ElectronEfficiencyOutput:
        hidden = self.backbone(continuous)
        return ElectronEfficiencyOutput(
            trigger_logit=self.trigger_head(hidden).squeeze(-1),
            outcome_logits=self.outcome_head(hidden),
        )

    def architecture_dict(self) -> dict[str, int | float]:
        linear_layers = [module for module in self.backbone if isinstance(module, nn.Linear)]
        dropout = next(
            (module.p for module in self.backbone if isinstance(module, nn.Dropout)), 0.0
        )
        return {
            "n_continuous": self.n_continuous,
            "n_outcomes": self.n_outcomes,
            "hidden_width": linear_layers[0].out_features,
            "hidden_layers": len(linear_layers),
            "dropout": dropout,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
