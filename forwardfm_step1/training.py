"""Optimization of the factorized continuous-response and PID likelihoods.

On the conditional T=1,C=FD sample, the step-one objective is

    L = L_response + lambda_PID L_PID
      = -E[log p_theta(Delta|x)]
        -lambda_PID E[log P_theta(s_rec|x)].

The response term is evaluated only because every row has a valid FD response.
Later trigger/outcome heads must use their own all-event denominators.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from collections.abc import Callable
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
    totals = {"loss": 0.0, "nll": 0.0, "pid_ce": 0.0, "correct": 0, "n": 0}
    start = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for continuous, species_index, targets, rec_pid_index in loader:
            continuous = continuous.to(device)
            species_index = species_index.to(device)
            targets = targets.to(device)
            rec_pid_index = rec_pid_index.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(continuous, species_index)
            # Continuous detector response: -log P(Delta|x,T=1,C=FD).
            nll = mixture_nll(output, targets)
            # PID contamination response: -log P(s_rec|x,T=1,C=FD).
            pid_ce = functional.cross_entropy(output.pid_logits, rec_pid_index)
            # Joint factorized objective L_R + lambda_PID L_PID.
            loss = nll + pid_loss_weight * pid_ce
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            batch_n = len(targets)
            totals["loss"] += float(loss.detach()) * batch_n
            totals["nll"] += float(nll.detach()) * batch_n
            totals["pid_ce"] += float(pid_ce.detach()) * batch_n
            totals["correct"] += int(
                (output.pid_logits.argmax(dim=-1) == rec_pid_index).sum().detach()
            )
            totals["n"] += batch_n
    elapsed = max(time.perf_counter() - start, 1e-9)
    return EpochMetrics(
        total_loss=totals["loss"] / totals["n"],
        residual_nll=totals["nll"] / totals["n"],
        pid_cross_entropy=totals["pid_ce"] / totals["n"],
        pid_accuracy=totals["correct"] / totals["n"],
        examples_per_second=totals["n"] / elapsed,
    )


def train_model(
    model: ConditionalMDN,
    splits: dict[str, PreparedSplit],
    config: dict[str, Any],
    device: torch.device,
    epoch_callback: Callable[[int, EpochMetrics, EpochMetrics], None] | None = None,
) -> tuple[
    ConditionalMDN,
    list[dict[str, Any]],
    int,
    dict[str, dict[str, Any]],
]:
    """Fit on event-disjoint data and retain two validation-only checkpoints.

    Candidate epochs minimize either

        L_val = L_response,val + lambda_PID L_PID,val

    or ``L_PID,val`` alone. ``training.checkpoint_metric`` declares which
    candidate is primary before held-out test evaluation.
    """
    training = config["training"]
    seed = int(config["project"]["seed"])
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
    checkpoint_metric = str(training.get("checkpoint_metric", "total_loss"))
    checkpoint_metrics = ("total_loss", "pid_cross_entropy")
    if checkpoint_metric not in checkpoint_metrics:
        raise ValueError(
            "training.checkpoint_metric must be 'total_loss' or "
            "'pid_cross_entropy'"
        )
    best_values = {name: float("inf") for name in checkpoint_metrics}
    best_epochs = {name: 0 for name in checkpoint_metrics}
    best_states: dict[str, dict[str, torch.Tensor]] = {}
    stale_epochs = 0

    for epoch in range(1, int(training["epochs"]) + 1):
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
        if epoch_callback is not None:
            epoch_callback(epoch, train_metrics, validation_metrics)
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
        validation_values = {
            "total_loss": validation_metrics.total_loss,
            "pid_cross_entropy": validation_metrics.pid_cross_entropy,
        }
        primary_improved = False
        for name, value in validation_values.items():
            if value < best_values[name] - 1e-5:
                best_values[name] = value
                best_epochs[name] = epoch
                best_states[name] = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                }
                if name == checkpoint_metric:
                    primary_improved = True
        if primary_improved:
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["early_stopping_patience"]):
                print(
                    f"early stopping after epoch {epoch}; "
                    f"best {checkpoint_metric} epoch was {best_epochs[checkpoint_metric]}"
                )
                break
        if scheduler is not None:
            scheduler.step()

    if set(best_states) != set(checkpoint_metrics):
        raise RuntimeError("Training did not produce a checkpoint")
    best_epoch = best_epochs[checkpoint_metric]
    model.load_state_dict(best_states[checkpoint_metric])
    model.to(device)
    candidates = {
        name: {
            "epoch": best_epochs[name],
            "validation_value": best_values[name],
            "model_state": best_states[name],
        }
        for name in checkpoint_metrics
    }
    return model, history, best_epoch, candidates
