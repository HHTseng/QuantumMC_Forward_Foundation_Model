"""Dr. Joo's separate delta-p and (theta, phi, beta) mixture heads.

Given generated features z and species s,
q(u_p, eta, r | z,s) = q_p(u_p | z,s) q_o(eta | z,s) rho_r(z,s),
where u_p is standardized asinh(delta_p / 0.020 GeV) and eta contains
standardized (delta_theta, delta_phi, delta_beta). Separate mixture labels
impose conditional independence of delta-p and the remaining residuals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SplitModelOutput:
    dp_mixture_logits: torch.Tensor
    dp_means: torch.Tensor
    dp_log_scales: torch.Tensor
    other_mixture_logits: torch.Tensor
    other_means: torch.Tensor
    other_log_scales: torch.Tensor
    pid_logits: torch.Tensor


class ConditionalSplitMDN(nn.Module):
    def __init__(
        self, n_continuous, n_species, n_rec_pid_classes,
        hidden_width=256, hidden_layers=4, pid_embedding_dim=16,
        mixture_components=8, target_dim=4, dropout=0.03,
    ):
        super().__init__()
        if target_dim != 4 or n_continuous not in (5, 6):
            raise ValueError("Split response requires 4 targets and 5 or 6 inputs")
        self.n_continuous = n_continuous
        self.n_species = n_species
        self.n_rec_pid_classes = n_rec_pid_classes
        self.mixture_components = mixture_components
        self.target_dim = target_dim
        self.species_embedding = nn.Embedding(n_species, pid_embedding_dim)
        layers = []
        for layer in range(hidden_layers):
            input_width = n_continuous + pid_embedding_dim if layer == 0 else hidden_width
            layers.extend([nn.Linear(input_width, hidden_width), nn.SiLU(), nn.LayerNorm(hidden_width)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.dp_mixture_head = nn.Linear(hidden_width, mixture_components)
        self.dp_mean_head = nn.Linear(hidden_width, mixture_components)
        self.dp_log_scale_head = nn.Linear(hidden_width, mixture_components)
        self.other_mixture_head = nn.Linear(hidden_width, mixture_components)
        self.other_mean_head = nn.Linear(hidden_width, mixture_components * 3)
        self.other_log_scale_head = nn.Linear(hidden_width, mixture_components * 3)
        self.pid_head = nn.Linear(hidden_width, n_rec_pid_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, continuous, species_index):
        embedded = self.species_embedding(species_index)
        hidden = self.backbone(torch.cat([continuous, embedded], dim=-1))
        shape = (-1, self.mixture_components, 3)
        return SplitModelOutput(
            self.dp_mixture_head(hidden), self.dp_mean_head(hidden),
            self.dp_log_scale_head(hidden).clamp(-7, 3),
            self.other_mixture_head(hidden), self.other_mean_head(hidden).reshape(shape),
            self.other_log_scale_head(hidden).reshape(shape).clamp(-7, 3),
            self.pid_head(hidden),
        )

    def architecture_dict(self):
        linear = [m for m in self.backbone if isinstance(m, nn.Linear)]
        dropout = next((m.p for m in self.backbone if isinstance(m, nn.Dropout)), 0.0)
        return {
            "n_continuous": self.n_continuous, "n_species": self.n_species,
            "n_rec_pid_classes": self.n_rec_pid_classes,
            "hidden_width": linear[0].out_features, "hidden_layers": len(linear),
            "pid_embedding_dim": self.species_embedding.embedding_dim,
            "mixture_components": self.mixture_components, "target_dim": 4,
            "dropout": dropout,
        }


def _mdn_nll(logits, means, log_scales, targets):
    standardized = (targets - means) * torch.exp(-log_scales)
    log_prob = -0.5 * standardized.square().sum(-1) - log_scales.sum(-1)
    log_prob -= 0.5 * means.shape[-1] * math.log(2 * math.pi)
    return -torch.logsumexp(torch.log_softmax(logits, -1) + log_prob, -1).mean()


def split_mixture_nll(output, targets):
    """NLL in standardized model coordinates; physical Jacobian is separate."""
    dp = _mdn_nll(output.dp_mixture_logits, output.dp_means.unsqueeze(-1),
                  output.dp_log_scales.unsqueeze(-1), targets[:, None, :1])
    other = _mdn_nll(output.other_mixture_logits, output.other_means,
                     output.other_log_scales, targets[:, None, 1:])
    return dp + other


@torch.no_grad()
def sample_split_standardized_residuals(output, generator=None):
    components = torch.multinomial(torch.softmax(output.dp_mixture_logits, -1), 1, generator=generator).squeeze(1)
    rows = torch.arange(len(components), device=components.device)
    mu = output.dp_means[rows, components]
    sigma = output.dp_log_scales[rows, components].exp()
    dp = mu + sigma * torch.randn(mu.shape, device=mu.device, generator=generator)
    components = torch.multinomial(torch.softmax(output.other_mixture_logits, -1), 1, generator=generator).squeeze(1)
    mu = output.other_means[rows, components]
    sigma = output.other_log_scales[rows, components].exp()
    other = mu + sigma * torch.randn(mu.shape, device=mu.device, generator=generator)
    return torch.cat([dp[:, None], other], dim=1)


@torch.no_grad()
def initialize_raw_momentum_from_control(control, extended):
    """Embed F_5 in F_6: copy every shared weight and set raw-p weight to zero."""
    if control.n_continuous != 5 or extended.n_continuous != 6:
        raise ValueError("Expected a 5-input control and a 6-input extension")
    state = extended.state_dict()
    for name, value in control.state_dict().items():
        if name == "backbone.0.weight":
            state[name].zero_()
            state[name][:, :5] = value[:, :5]
            state[name][:, 6:] = value[:, 5:]
        else:
            if state[name].shape != value.shape:
                raise ValueError(f"Incompatible paired architecture: {name}")
            state[name].copy_(value)
    extended.load_state_dict(state)
