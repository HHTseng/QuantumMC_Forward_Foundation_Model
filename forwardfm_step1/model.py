"""Conditional stochastic physics response model.

The neural network approximates the FD response factor

    P(Delta, s_rec | x, T=1, C=FD)
      = P(Delta | x,T=1,C=FD) P(s_rec | x,T=1,C=FD),

with a Gaussian mixture for the continuous residual vector Delta and a
categorical head for reconstructed PID.  It predicts a distribution rather
than only E[Delta|x], because detector resolution and reconstruction tails are
physical stochastic effects that a mean-squared-error regressor would erase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelOutput:
    mixture_logits: torch.Tensor
    means: torch.Tensor
    log_scales: torch.Tensor
    pid_logits: torch.Tensor


class ConditionalMDN(nn.Module):
    """Shared particle backbone with a joint residual MDN and REC-PID head.

    The conditional residual density is

        p(Delta|x) = sum_{k=1}^K pi_k(x)
                     prod_{j in configured response targets}
                     Normal(Delta_j; mu_kj(x), sigma_kj(x)^2).

    Each component is diagonal internally. Mixture membership can still encode
    joint target dependence, including Delta beta when configured, while
    keeping the baseline stable and auditable. The PID head is
    softmax(logits(x)) = P(s_rec|x).
    """

    def __init__(
        self,
        n_continuous: int,
        n_species: int,
        n_rec_pid_classes: int,
        hidden_width: int = 128,
        hidden_layers: int = 3,
        pid_embedding_dim: int = 8,
        mixture_components: int = 5,
        target_dim: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_continuous = n_continuous
        self.n_species = n_species
        self.n_rec_pid_classes = n_rec_pid_classes
        self.mixture_components = mixture_components
        self.target_dim = target_dim
        self.species_embedding = nn.Embedding(n_species, pid_embedding_dim)

        layers: list[nn.Module] = []
        input_width = n_continuous + pid_embedding_dim
        for layer_index in range(hidden_layers):
            layers.append(nn.Linear(input_width if layer_index == 0 else hidden_width, hidden_width))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(hidden_width))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.mixture_head = nn.Linear(hidden_width, mixture_components)
        self.mean_head = nn.Linear(hidden_width, mixture_components * target_dim)
        self.log_scale_head = nn.Linear(hidden_width, mixture_components * target_dim)
        self.pid_head = nn.Linear(hidden_width, n_rec_pid_classes)
        self.reset_parameters()

    def reset_parameters(self, seed: int | None = None) -> None:
        """Initialize parameters, optionally with component-paired generators.

        A seeded reset gives every module its own deterministic random stream.
        Models whose response heads have different output dimensions therefore
        still start with identical species embeddings, shared-backbone weights,
        mixture weights, and PID-head weights.  This is useful for a paired
        auxiliary-target ablation; the default preserves the original policy.
        """
        for module_index, module in enumerate(self.modules()):
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(
                    int(seed) + 1009 * module_index
                )
            if isinstance(module, nn.Embedding) and seed is not None:
                nn.init.normal_(module.weight, generator=generator)
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, generator=generator)
                nn.init.zeros_(module.bias)
        # Start near unit variance in standardized target space.
        nn.init.constant_(self.log_scale_head.bias, 0.0)

    def forward(self, continuous: torch.Tensor, species_index: torch.Tensor) -> ModelOutput:
        """Evaluate response parameters conditioned on truth x and species s.

        `continuous` represents (log(1+p), theta, sin(phi), cos(phi)); the
        embedding supplies s_gen.  No reconstructed quantity is used as an
        input, avoiding target leakage.
        """
        species = self.species_embedding(species_index)
        hidden = self.backbone(torch.cat([continuous, species], dim=-1))
        batch_size = continuous.shape[0]
        means = self.mean_head(hidden).reshape(
            batch_size, self.mixture_components, self.target_dim
        )
        log_scales = self.log_scale_head(hidden).reshape(
            batch_size, self.mixture_components, self.target_dim
        )
        # Bounds prevent numerical collapse/explosion while retaining a wide
        # range of possible physical resolutions.
        log_scales = torch.clamp(log_scales, min=-7.0, max=3.0)
        return ModelOutput(
            mixture_logits=self.mixture_head(hidden),
            means=means,
            log_scales=log_scales,
            pid_logits=self.pid_head(hidden),
        )

    def architecture_dict(self) -> dict[str, int | float]:
        linear_layers = [module for module in self.backbone if isinstance(module, nn.Linear)]
        hidden_width = linear_layers[0].out_features
        dropout = next(
            (module.p for module in self.backbone if isinstance(module, nn.Dropout)), 0.0
        )
        return {
            "n_continuous": self.n_continuous,
            "n_species": self.n_species,
            "n_rec_pid_classes": self.n_rec_pid_classes,
            "hidden_width": hidden_width,
            "hidden_layers": len(linear_layers),
            "pid_embedding_dim": self.species_embedding.embedding_dim,
            "mixture_components": self.mixture_components,
            "target_dim": self.target_dim,
            "dropout": dropout,
        }


def mixture_nll(output: ModelOutput, targets: torch.Tensor) -> torch.Tensor:
    """Monte Carlo estimate of -E_data[log p_theta(Delta|x)].

    For diagonal component k,

        log N_k = -1/2 sum_j ((Delta_j-mu_kj)/sigma_kj)^2
                  - sum_j log sigma_kj - d/2 log(2 pi),

    followed by log-sum-exp over log(pi_k)+log(N_k). Minimizing this likelihood
    rewards calibrated widths and tails, not just a correct conditional mean.
    """
    targets = targets[:, None, :]
    inverse_scales = torch.exp(-output.log_scales)
    standardized = (targets - output.means) * inverse_scales
    component_log_prob = -0.5 * torch.sum(standardized.square(), dim=-1)
    component_log_prob -= torch.sum(output.log_scales, dim=-1)
    component_log_prob -= 0.5 * output.means.shape[-1] * math.log(2.0 * math.pi)
    log_weights = torch.log_softmax(output.mixture_logits, dim=-1)
    return -torch.logsumexp(log_weights + component_log_prob, dim=-1).mean()


@torch.no_grad()
def sample_standardized_residuals(
    output: ModelOutput,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw Delta by k~Categorical(pi(x)), epsilon~N(0,I), Delta=mu_k+sigma_k*epsilon."""
    probabilities = torch.softmax(output.mixture_logits, dim=-1)
    components = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    rows = torch.arange(components.shape[0], device=components.device)
    means = output.means[rows, components]
    scales = torch.exp(output.log_scales[rows, components])
    noise = torch.randn(means.shape, device=means.device, generator=generator)
    return means + scales * noise


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
