"""Optimization for trigger efficiency and reconstruction outcome.

The unweighted calibrated objective is

  L = -lambda_T E[y log eta + (1-y) log(1-eta)]
      + lambda_C CE(C, softmax(z_C)).

Class weighting is intentionally absent because it changes the probability
target unless an explicit recalibration step is added.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from forwardfm_step1.training import seed_everything  # re-exported utility

from .data import ElectronEfficiencySplit
from .model import ElectronEfficiencyNet


@dataclass
class EfficiencyEpochMetrics:
    total_loss: float
    trigger_bce: float
    outcome_cross_entropy: float
    trigger_accuracy: float
    outcome_accuracy: float
    examples_per_second: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def make_loader(
    split: ElectronEfficiencySplit,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(split.continuous),
        torch.from_numpy(split.trigger_target),
        torch.from_numpy(split.outcome_target),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )


def run_epoch(
    model: ElectronEfficiencyNet,
    loader: DataLoader,
    device: torch.device,
    trigger_loss_weight: float,
    outcome_loss_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 5.0,
) -> EfficiencyEpochMetrics:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "trigger": 0.0, "outcome": 0.0, "tc": 0, "oc": 0, "n": 0}
    start = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for continuous, trigger_target, outcome_target in loader:
            continuous = continuous.to(device)
            trigger_target = trigger_target.to(device)
            outcome_target = outcome_target.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(continuous)
            trigger_bce = functional.binary_cross_entropy_with_logits(
                output.trigger_logit, trigger_target
            )
            outcome_ce = functional.cross_entropy(output.outcome_logits, outcome_target)
            loss = trigger_loss_weight * trigger_bce + outcome_loss_weight * outcome_ce
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            n = len(trigger_target)
            totals["loss"] += float(loss.detach()) * n
            totals["trigger"] += float(trigger_bce.detach()) * n
            totals["outcome"] += float(outcome_ce.detach()) * n
            totals["tc"] += int(
                ((output.trigger_logit >= 0) == (trigger_target >= 0.5)).sum().detach()
            )
            totals["oc"] += int(
                (output.outcome_logits.argmax(dim=-1) == outcome_target).sum().detach()
            )
            totals["n"] += n
    elapsed = max(time.perf_counter() - start, 1e-9)
    return EfficiencyEpochMetrics(
        total_loss=totals["loss"] / totals["n"],
        trigger_bce=totals["trigger"] / totals["n"],
        outcome_cross_entropy=totals["outcome"] / totals["n"],
        trigger_accuracy=totals["tc"] / totals["n"],
        outcome_accuracy=totals["oc"] / totals["n"],
        examples_per_second=totals["n"] / elapsed,
    )


def preload_split(
    split: ElectronEfficiencySplit,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move the compact all-event tensors to a large GPU once per run."""
    return (
        torch.from_numpy(split.continuous).to(device),
        torch.from_numpy(split.trigger_target).to(device),
        torch.from_numpy(split.outcome_target).to(device),
    )


def run_preloaded_epoch(
    model: ElectronEfficiencyNet,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
    trigger_loss_weight: float,
    outcome_loss_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 5.0,
) -> EfficiencyEpochMetrics:
    """Vectorized GPU batching with identical losses to run_epoch().

    This avoids Python's per-row DataLoader collation on multi-million-row
    in-memory datasets. It changes only data movement, not the physics sample
    or optimization objective.
    """
    continuous, trigger_target, outcome_target = tensors
    n_rows = len(trigger_target)
    order = (
        torch.randperm(n_rows, device=continuous.device, generator=generator)
        if shuffle
        else None
    )
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "trigger": 0.0, "outcome": 0.0, "tc": 0, "oc": 0, "n": 0}
    start_time = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            index = order[start:stop] if order is not None else slice(start, stop)
            batch_continuous = continuous[index]
            batch_trigger = trigger_target[index]
            batch_outcome = outcome_target[index]
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch_continuous)
            trigger_bce = functional.binary_cross_entropy_with_logits(
                output.trigger_logit, batch_trigger
            )
            outcome_ce = functional.cross_entropy(output.outcome_logits, batch_outcome)
            loss = trigger_loss_weight * trigger_bce + outcome_loss_weight * outcome_ce
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            batch_n = len(batch_trigger)
            totals["loss"] += float(loss.detach()) * batch_n
            totals["trigger"] += float(trigger_bce.detach()) * batch_n
            totals["outcome"] += float(outcome_ce.detach()) * batch_n
            totals["tc"] += int(
                ((output.trigger_logit >= 0) == (batch_trigger >= 0.5)).sum().detach()
            )
            totals["oc"] += int(
                (output.outcome_logits.argmax(dim=-1) == batch_outcome).sum().detach()
            )
            totals["n"] += batch_n
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    return EfficiencyEpochMetrics(
        total_loss=totals["loss"] / totals["n"],
        trigger_bce=totals["trigger"] / totals["n"],
        outcome_cross_entropy=totals["outcome"] / totals["n"],
        trigger_accuracy=totals["tc"] / totals["n"],
        outcome_accuracy=totals["oc"] / totals["n"],
        examples_per_second=totals["n"] / elapsed,
    )


def train_model(
    model: ElectronEfficiencyNet,
    splits: dict[str, ElectronEfficiencySplit],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[ElectronEfficiencyNet, list[dict[str, Any]], int]:
    training = config["training"]
    seed = int(config["project"]["seed"])
    preload = bool(training.get("preload_to_device", False))
    if preload and device.type == "cpu":
        raise ValueError("preload_to_device is intended for an accelerator")
    if preload:
        train_data = preload_split(splits["train"], device)
        validation_data = preload_split(splits["validation"], device)
        batch_generator = torch.Generator(device=device.type).manual_seed(seed)
        print("preloaded train/validation efficiency tensors to accelerator")
    else:
        train_loader = make_loader(
            splits["train"], int(training["batch_size"]), True, seed,
            int(training.get("num_workers", 0))
        )
        validation_loader = make_loader(
            splits["validation"], int(training["batch_size"]), False, seed,
            int(training.get("num_workers", 0))
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"])
    )
    model.to(device)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, int(training["epochs"]) + 1):
        common = (
            device,
            float(training["trigger_loss_weight"]),
            float(training["outcome_loss_weight"]),
        )
        if preload:
            train_metrics = run_preloaded_epoch(
                model, train_data, int(training["batch_size"]), True,
                batch_generator, float(training["trigger_loss_weight"]),
                float(training["outcome_loss_weight"]), optimizer=optimizer,
                gradient_clip_norm=float(training["gradient_clip_norm"])
            )
            validation_metrics = run_preloaded_epoch(
                model, validation_data, int(training["batch_size"]), False,
                batch_generator, float(training["trigger_loss_weight"]),
                float(training["outcome_loss_weight"])
            )
        else:
            train_metrics = run_epoch(
                model, train_loader, *common, optimizer=optimizer,
                gradient_clip_norm=float(training["gradient_clip_norm"])
            )
            validation_metrics = run_epoch(model, validation_loader, *common)
        history.append(
            {"epoch": epoch, "train": train_metrics.as_dict(),
             "validation": validation_metrics.as_dict()}
        )
        print(
            f"epoch={epoch:02d} train_loss={train_metrics.total_loss:.6f} "
            f"val_loss={validation_metrics.total_loss:.6f} "
            f"val_bce={validation_metrics.trigger_bce:.6f} "
            f"val_trigger_acc={validation_metrics.trigger_accuracy:.4f}"
        )
        if validation_metrics.total_loss < best_loss - 1e-6:
            best_loss = validation_metrics.total_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(training["early_stopping_patience"]):
                print(f"early stopping after epoch {epoch}; best epoch was {best_epoch}")
                break
    if best_state is None:
        raise RuntimeError("Efficiency training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return model, history, best_epoch
