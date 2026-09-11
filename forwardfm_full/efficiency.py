"""Bernoulli trigger-efficiency model and validation-only training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import TriggerSplit


class TriggerEfficiencyNet(nn.Module):
    r"""Map generated-electron coordinates to \(P(T=1\mid x_e)\)."""

    def __init__(
        self,
        n_continuous: int,
        hidden_width: int = 128,
        hidden_layers: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_continuous = n_continuous
        layers: list[nn.Module] = []
        for index in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(
                        n_continuous if index == 0 else hidden_width,
                        hidden_width,
                    ),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_width),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.logit = nn.Linear(hidden_width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, continuous: torch.Tensor) -> torch.Tensor:
        return self.logit(self.backbone(continuous)).squeeze(-1)

    def architecture_dict(self) -> dict[str, int | float]:
        linear = [module for module in self.backbone if isinstance(module, nn.Linear)]
        dropout = next(
            (module.p for module in self.backbone if isinstance(module, nn.Dropout)),
            0.0,
        )
        return {
            "n_continuous": self.n_continuous,
            "hidden_width": linear[0].out_features,
            "hidden_layers": len(linear),
            "dropout": dropout,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@dataclass
class EpochMetrics:
    binary_cross_entropy: float
    accuracy: float
    examples_per_second: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def make_loader(
    split: TriggerSplit,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(split.continuous),
            torch.from_numpy(split.trigger_target),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )


def run_loader_epoch(
    model: TriggerEfficiencyNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 5.0,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    count = 0
    start_time = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for continuous, target in loader:
            continuous = continuous.to(device)
            target = target.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(continuous)
            loss = functional.binary_cross_entropy_with_logits(logits, target)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            n = len(target)
            loss_sum += float(loss.detach()) * n
            correct += int(((logits >= 0) == (target >= 0.5)).sum().detach())
            count += n
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    return EpochMetrics(loss_sum / count, correct / count, count / elapsed)


def run_preloaded_epoch(
    model: TriggerEfficiencyNet,
    continuous: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 5.0,
) -> EpochMetrics:
    """Vectorized batches for multi-million-row accelerator training."""
    training = optimizer is not None
    model.train(training)
    order = (
        torch.randperm(len(target), device=continuous.device, generator=generator)
        if shuffle
        else None
    )
    loss_sum = 0.0
    correct = 0
    start_time = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for start in range(0, len(target), batch_size):
            stop = min(start + batch_size, len(target))
            index = order[start:stop] if order is not None else slice(start, stop)
            batch_x = continuous[index]
            batch_y = target[index]
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = functional.binary_cross_entropy_with_logits(logits, batch_y)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            n = len(batch_y)
            loss_sum += float(loss.detach()) * n
            correct += int(((logits >= 0) == (batch_y >= 0.5)).sum().detach())
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    return EpochMetrics(loss_sum / len(target), correct / len(target), len(target) / elapsed)


def train_model(
    model: TriggerEfficiencyNet,
    splits: dict[str, TriggerSplit],
    config: dict[str, Any],
    device: torch.device,
    epoch_callback: Callable[[int, EpochMetrics, EpochMetrics], None] | None = None,
) -> tuple[TriggerEfficiencyNet, list[dict[str, Any]], int, float]:
    """Minimize unweighted BCE; choose the checkpoint by validation BCE."""
    training = config["training"]
    seed = int(config["project"]["seed"])
    preload = bool(training.get("preload_to_device", False))
    if preload and device.type == "cpu":
        raise ValueError("preload_to_device requires an accelerator")
    if preload:
        train_x = torch.from_numpy(splits["train"].continuous).to(device)
        train_y = torch.from_numpy(splits["train"].trigger_target).to(device)
        val_x = torch.from_numpy(splits["validation"].continuous).to(device)
        val_y = torch.from_numpy(splits["validation"].trigger_target).to(device)
        generator = torch.Generator(device=device.type).manual_seed(seed)
        print("preloaded train/validation trigger tensors to accelerator")
    else:
        train_loader = make_loader(
            splits["train"],
            int(training["batch_size"]),
            True,
            seed,
            int(training.get("num_workers", 0)),
        )
        val_loader = make_loader(
            splits["validation"], int(training["batch_size"]), False, seed
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    schedule = str(training.get("lr_schedule", "none"))
    if schedule not in {"none", "cosine"}:
        raise ValueError("training.lr_schedule must be 'none' or 'cosine'")
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training["epochs"]),
            eta_min=float(training["learning_rate"])
            * float(training.get("lr_min_factor", 0.01)),
        )
        if schedule == "cosine"
        else None
    )
    model.to(device)
    history: list[dict[str, Any]] = []
    best_value = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, int(training["epochs"]) + 1):
        if preload:
            train_metrics = run_preloaded_epoch(
                model,
                train_x,
                train_y,
                int(training["batch_size"]),
                True,
                generator,
                optimizer,
                float(training["gradient_clip_norm"]),
            )
            validation_metrics = run_preloaded_epoch(
                model,
                val_x,
                val_y,
                int(training["batch_size"]),
                False,
                generator,
            )
        else:
            train_metrics = run_loader_epoch(
                model,
                train_loader,
                device,
                optimizer,
                float(training["gradient_clip_norm"]),
            )
            validation_metrics = run_loader_epoch(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics.as_dict(),
                "validation": validation_metrics.as_dict(),
            }
        )
        if epoch_callback is not None:
            epoch_callback(epoch, train_metrics, validation_metrics)
        print(
            f"epoch={epoch:02d} train_bce={train_metrics.binary_cross_entropy:.6f} "
            f"val_bce={validation_metrics.binary_cross_entropy:.6f} "
            f"val_accuracy={validation_metrics.accuracy:.4f}"
        )
        if validation_metrics.binary_cross_entropy < best_value - 1e-6:
            best_value = validation_metrics.binary_cross_entropy
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(training["early_stopping_patience"]):
                print(f"early stopping after epoch {epoch}; best epoch={best_epoch}")
                break
        if scheduler is not None:
            scheduler.step()
    if best_state is None:
        raise RuntimeError("Trigger training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return model, history, best_epoch, best_value
