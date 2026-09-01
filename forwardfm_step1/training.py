"""Optimization of the factorized continuous-response and PID likelihoods.

On the conditional T=1,C=FD sample, the step-one objective is

    L = L_response + lambda_PID L_PID
      = -E[log p_theta(Delta|x)]
        -lambda_PID E[log P_theta(s_rec|x)].

The response term is evaluated only because every row has a valid FD response.
The separate trigger-efficiency package uses its own all-event denominator;
the general particle reconstruction-outcome factor remains future work.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import PreparedSplit
from .model import ConditionalMDN, mixture_nll


@dataclass
class EpochMetrics:
    total_loss: float
    residual_nll: float
    pid_cross_entropy: float
    pid_accuracy: float
    examples_per_second: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total_loss": self.total_loss,
            "residual_nll": self.residual_nll,
            "pid_cross_entropy": self.pid_cross_entropy,
            "pid_accuracy": self.pid_accuracy,
            "examples_per_second": self.examples_per_second,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    split: PreparedSplit,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(split.continuous),
        torch.from_numpy(split.species_index),
        torch.from_numpy(split.targets),
        torch.from_numpy(split.rec_pid_index),
        torch.from_numpy(split.pid_loss_mask),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )


def run_epoch(
    model: ConditionalMDN,
    loader: DataLoader,
    device: torch.device,
    pid_loss_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 5.0,
) -> EpochMetrics:
    """Accumulate the joint negative log likelihood for one data pass."""
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "nll": 0.0,
        "pid_ce": 0.0,
        "correct": 0,
        "pid_n": 0,
        "n": 0,
    }
    start = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for continuous, species_index, targets, rec_pid_index, pid_loss_mask in loader:
            continuous = continuous.to(device)
            species_index = species_index.to(device)
            targets = targets.to(device)
            rec_pid_index = rec_pid_index.to(device)
            pid_loss_mask = pid_loss_mask.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(continuous, species_index)
            # Continuous detector response: -log P(Delta|x,T=1,C=FD).
            nll = mixture_nll(output, targets)
            # PID contamination response: -log P(s_rec|x,T=1,C=FD).
            per_row_pid_ce = functional.cross_entropy(
                output.pid_logits, rec_pid_index, reduction="none"
            )
            if bool(pid_loss_mask.any()):
                pid_ce = per_row_pid_ce[pid_loss_mask].mean()
            else:
                # Preserve a differentiable zero when a configuration masks
                # every row in a batch from the categorical objective.
                pid_ce = output.pid_logits.sum() * 0.0
            # Joint factorized objective L_R + lambda_PID L_PID.
            loss = nll + pid_loss_weight * pid_ce
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            batch_n = len(targets)
            batch_pid_n = int(pid_loss_mask.sum().detach())
            totals["loss"] += float(loss.detach()) * batch_n
            totals["nll"] += float(nll.detach()) * batch_n
            totals["pid_ce"] += float(pid_ce.detach()) * batch_pid_n
            totals["correct"] += int(
                (
                    (output.pid_logits.argmax(dim=-1) == rec_pid_index)
                    & pid_loss_mask
                ).sum().detach()
            )
            totals["pid_n"] += batch_pid_n
            totals["n"] += batch_n
    elapsed = max(time.perf_counter() - start, 1e-9)
    return EpochMetrics(
        total_loss=totals["loss"] / totals["n"],
        residual_nll=totals["nll"] / totals["n"],
        pid_cross_entropy=(
            totals["pid_ce"] / totals["pid_n"] if totals["pid_n"] else 0.0
        ),
        pid_accuracy=(totals["correct"] / totals["pid_n"] if totals["pid_n"] else 0.0),
        examples_per_second=totals["n"] / elapsed,
    )


def preload_split(
    split: PreparedSplit,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move compact selected-response tensors to a large GPU once per run."""
    return (
        torch.from_numpy(split.continuous).to(device),
        torch.from_numpy(split.species_index).to(device),
        torch.from_numpy(split.targets).to(device),
        torch.from_numpy(split.rec_pid_index).to(device),
        torch.from_numpy(split.pid_loss_mask).to(device),
    )


def run_preloaded_epoch(
    model: ConditionalMDN,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
    pid_loss_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 5.0,
) -> EpochMetrics:
    """Vectorized accelerator batching equivalent to run_epoch()."""
    continuous, species_index, targets, rec_pid_index, pid_loss_mask = tensors
    n_rows = len(targets)
    order = (
        torch.randperm(n_rows, device=continuous.device, generator=generator)
        if shuffle
        else None
    )
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0, "nll": 0.0, "pid_ce": 0.0,
        "correct": 0, "pid_n": 0, "n": 0,
    }
    start_time = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            index = order[start:stop] if order is not None else slice(start, stop)
            batch_continuous = continuous[index]
            batch_species = species_index[index]
            batch_targets = targets[index]
            batch_rec_pid = rec_pid_index[index]
            batch_pid_mask = pid_loss_mask[index]
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch_continuous, batch_species)
            nll = mixture_nll(output, batch_targets)
            per_row_pid_ce = functional.cross_entropy(
                output.pid_logits, batch_rec_pid, reduction="none"
            )
            pid_ce = (
                per_row_pid_ce[batch_pid_mask].mean()
                if bool(batch_pid_mask.any())
                else output.pid_logits.sum() * 0.0
            )
            loss = nll + pid_loss_weight * pid_ce
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            batch_n = len(batch_targets)
            batch_pid_n = int(batch_pid_mask.sum().detach())
            totals["loss"] += float(loss.detach()) * batch_n
            totals["nll"] += float(nll.detach()) * batch_n
            totals["pid_ce"] += float(pid_ce.detach()) * batch_pid_n
            totals["correct"] += int(
                ((output.pid_logits.argmax(dim=-1) == batch_rec_pid) & batch_pid_mask)
                .sum().detach()
            )
            totals["pid_n"] += batch_pid_n
            totals["n"] += batch_n
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    return EpochMetrics(
        total_loss=totals["loss"] / totals["n"],
        residual_nll=totals["nll"] / totals["n"],
        pid_cross_entropy=(totals["pid_ce"] / totals["pid_n"] if totals["pid_n"] else 0.0),
        pid_accuracy=(totals["correct"] / totals["pid_n"] if totals["pid_n"] else 0.0),
        examples_per_second=totals["n"] / elapsed,
    )


def train_model(
    model: ConditionalMDN,
    splits: dict[str, PreparedSplit],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[ConditionalMDN, list[dict[str, Any]], int]:
    """Fit on event-disjoint training data and select by validation likelihood."""
    training = config["training"]
    seed = int(config["project"]["seed"])
    preload = bool(training.get("preload_to_device", False))
    if preload and device.type == "cpu":
        raise ValueError("preload_to_device is intended for an accelerator")
    if preload:
        train_data = preload_split(splits["train"], device)
        validation_data = preload_split(splits["validation"], device)
        batch_generator = torch.Generator(device=device.type).manual_seed(seed)
        print("preloaded train/validation response tensors to accelerator")
    else:
        train_loader = make_loader(
            splits["train"],
            int(training["batch_size"]),
            shuffle=True,
            seed=seed,
            num_workers=int(training["num_workers"]),
        )
        validation_loader = make_loader(
            splits["validation"],
            int(training["batch_size"]),
            shuffle=False,
            seed=seed,
            num_workers=int(training["num_workers"]),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    model.to(device)
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, int(training["epochs"]) + 1):
        if preload:
            train_metrics = run_preloaded_epoch(
                model, train_data, int(training["batch_size"]), True,
                batch_generator, float(training["pid_loss_weight"]),
                optimizer=optimizer,
                gradient_clip_norm=float(training["gradient_clip_norm"]),
            )
            validation_metrics = run_preloaded_epoch(
                model, validation_data, int(training["batch_size"]), False,
                batch_generator, float(training["pid_loss_weight"]),
            )
        else:
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                float(training["pid_loss_weight"]),
                optimizer=optimizer,
                gradient_clip_norm=float(training["gradient_clip_norm"]),
            )
            validation_metrics = run_epoch(
                model,
                validation_loader,
                device,
                float(training["pid_loss_weight"]),
            )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics.as_dict(),
                "validation": validation_metrics.as_dict(),
            }
        )
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics.total_loss:.5f} "
            f"val_loss={validation_metrics.total_loss:.5f} "
            f"val_nll={validation_metrics.residual_nll:.5f} "
            f"val_pid_acc={validation_metrics.pid_accuracy:.4f}"
        )
        if validation_metrics.total_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_metrics.total_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["early_stopping_patience"]):
                print(f"early stopping after epoch {epoch}; best epoch was {best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return model, history, best_epoch
