"""Optimization of the factorized continuous-response and PID likelihoods.

On the conditional T=1,C=FD sample, the step-one objective is

    L = L_response + lambda_PID L_PID
      = -E[log p_theta(Delta|x)]
        -lambda_PID E[log P_theta(s_rec|x)].

The response term is evaluated only because every row has a valid FD response.
Later trigger/outcome heads must use their own all-event denominators.

Two opt-in mechanisms exist for hyper-parameter search and long schedules.
They are disabled by default so that the published baseline behaviour is bit-
for-bit unchanged:

``training.fast_loader``
    Keep the split tensors resident on the compute device and slice minibatches
    directly.  The per-sample ``TensorDataset`` collation is the throughput
    bottleneck for this small network, not the arithmetic.

``training.lr_schedule``
    ``cosine`` anneals the AdamW learning rate over the configured epoch
    budget, which matters once the epoch budget is large.

``training.selection_metric``
    ``total_loss`` (default) selects the checkpoint by ``L_R+lambda_PID L_PID``.
    ``joint_nll`` selects by ``L_R+L_PID``, the lambda-independent held-out
    joint negative log likelihood ``-log q(Delta,s_rec|x)``.  Only the latter is
    comparable between runs that used different ``pid_loss_weight`` values.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable

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

    @property
    def joint_nll(self) -> float:
        """Lambda-independent held-out joint NLL -log q(Delta,s_rec|x).

        ``total_loss`` uses the training ``pid_loss_weight``, so it cannot rank
        two runs that were trained with different weights.  The unweighted sum
        is a proper log density of the factorized model and can.
        """
        return self.residual_nll + self.pid_cross_entropy

    def as_dict(self) -> dict[str, float]:
        return {
            "total_loss": self.total_loss,
            "residual_nll": self.residual_nll,
            "pid_cross_entropy": self.pid_cross_entropy,
            "pid_accuracy": self.pid_accuracy,
            "joint_nll": self.joint_nll,
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


class DeviceResidentLoader:
    """Minibatch iterator over split tensors held on the compute device.

    ``TensorDataset`` collates one row at a time, which dominates wall clock for
    a network this small: the published full run reported about 1e5 examples/s
    on an H100 that can evaluate this backbone orders of magnitude faster.  The
    whole selected sample is a few tens of megabytes, so keeping it resident and
    slicing with an index tensor removes that bottleneck without changing the
    loss, the optimizer, or the batch composition semantics.

    Like ``make_loader`` this keeps ``drop_last=False`` and draws the shuffling
    permutation from a seeded generator, so epochs remain reproducible.
    """

    def __init__(
        self,
        split: PreparedSplit,
        batch_size: int,
        shuffle: bool,
        seed: int,
        device: torch.device,
    ) -> None:
        self.device = device
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.tensors = (
            torch.from_numpy(split.continuous).to(device),
            torch.from_numpy(split.species_index).to(device),
            torch.from_numpy(split.targets).to(device),
            torch.from_numpy(split.rec_pid_index).to(device),
        )
        self.n = len(split)
        self._generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.n, generator=self._generator).to(self.device)
        else:
            order = torch.arange(self.n, device=self.device)
        for start in range(0, self.n, self.batch_size):
            index = order[start : start + self.batch_size]
            yield tuple(tensor[index] for tensor in self.tensors)


def build_loader(
    split: PreparedSplit,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
    fast: bool,
):
    """Return the device-resident loader when requested, else the DataLoader."""
    if fast:
        return DeviceResidentLoader(split, batch_size, shuffle, seed, device)
    return make_loader(split, batch_size, shuffle, seed, num_workers)


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


def selection_value(metrics: EpochMetrics, selection_metric: str) -> float:
    """Return the scalar that ranks checkpoints under the configured rule."""
    if selection_metric == "total_loss":
        return metrics.total_loss
    if selection_metric == "joint_nll":
        return metrics.joint_nll
    raise ValueError(f"Unknown selection_metric {selection_metric!r}")


def train_model(
    model: ConditionalMDN,
    splits: dict[str, PreparedSplit],
    config: dict[str, Any],
    device: torch.device,
    epoch_callback: Callable[[int, EpochMetrics, EpochMetrics], None] | None = None,
) -> tuple[ConditionalMDN, list[dict[str, Any]], int]:
    """Fit on event-disjoint training data and select by validation likelihood.

    ``epoch_callback(epoch, train_metrics, validation_metrics)`` runs after each
    validation pass.  A hyper-parameter search uses it to report intermediate
    values and may raise to abort an unpromising trial.
    """
    training = config["training"]
    seed = int(config["project"]["seed"])
    fast_loader = bool(training.get("fast_loader", False))
    selection_metric = str(training.get("selection_metric", "total_loss"))
    max_epochs = int(training["epochs"])
    train_loader = build_loader(
        splits["train"],
        int(training["batch_size"]),
        shuffle=True,
        seed=seed,
        num_workers=int(training["num_workers"]),
        device=device,
        fast=fast_loader,
    )
    validation_loader = build_loader(
        splits["validation"],
        int(training["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(training["num_workers"]),
        device=device,
        fast=fast_loader,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    schedule = str(training.get("lr_schedule", "none"))
    if schedule == "cosine":
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=float(training["learning_rate"]) * float(
                    training.get("lr_min_factor", 0.01)
                ),
            )
        )
    elif schedule == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown lr_schedule {schedule!r}")
    model.to(device)
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
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
        if scheduler is not None:
            scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": train_metrics.as_dict(),
                "validation": validation_metrics.as_dict(),
            }
        )
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics.total_loss:.5f} "
            f"val_loss={validation_metrics.total_loss:.5f} "
            f"val_nll={validation_metrics.residual_nll:.5f} "
            f"val_joint_nll={validation_metrics.joint_nll:.5f} "
            f"val_pid_acc={validation_metrics.pid_accuracy:.4f}",
            flush=True,
        )
        current = selection_value(validation_metrics, selection_metric)
        if current < best_validation_loss - 1e-5:
            best_validation_loss = current
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch_callback is not None:
            epoch_callback(epoch, train_metrics, validation_metrics)
        if stale_epochs >= int(training["early_stopping_patience"]):
            print(f"early stopping after epoch {epoch}; best epoch was {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return model, history, best_epoch
