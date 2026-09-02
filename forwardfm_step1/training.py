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

``training.lr_warmup_epochs``
    Non-zero switches the learning rate to a *per-step* schedule: a linear
    warm-up over this many epochs' worth of optimizer steps, followed by the
    configured decay.  Warm-up has to act inside the first epoch, so it cannot
    be expressed by the epoch-granularity scheduler above.

``training.pid_head_lr_multiplier`` / ``training.backbone_lr_multiplier``
    Separate learning rates for the categorical head and the shared trunk.
    ``pid_loss_weight`` alone couples "how fast the PID head learns" to "how
    hard the PID term perturbs the trunk"; these separate the two.

``training.freeze_backbone``
    Optimize only the heads, for refitting PID on a fixed residual density.

``training.selection_metric``
    ``total_loss`` (default) selects the checkpoint by ``L_R+lambda_PID L_PID``.
    ``joint_nll`` selects by ``L_R+L_PID``, the lambda-independent held-out
    joint negative log likelihood ``-log q(Delta,s_rec|x)``.  Only the latter is
    comparable between runs that used different ``pid_loss_weight`` values.
"""

from __future__ import annotations

import math
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
    step_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> EpochMetrics:
    """Accumulate the joint negative log likelihood for one data pass.

    ``step_scheduler`` is advanced after every optimizer step rather than once
    per epoch.  Warm-up has to act on the first few hundred updates, which is a
    fraction of one epoch here, so an epoch-granularity schedule cannot express
    it.
    """
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
                if step_scheduler is not None:
                    step_scheduler.step()
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


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """The base learning rate, independent of how many groups exist."""
    for group in optimizer.param_groups:
        if group.get("name") == "density_heads":
            return float(group["lr"])
    return float(optimizer.param_groups[0]["lr"])


def build_optimizer(
    model: ConditionalMDN, training: dict[str, Any]
) -> torch.optim.Optimizer:
    """AdamW, optionally with a different learning rate for trunk and PID head.

    The shared trunk and the PID head are coupled through one loss weight:
    raising ``pid_loss_weight`` makes the PID term learn faster *and* perturbs
    the trunk harder, and the second effect is what destabilizes training. Two
    multipliers separate those knobs, so the PID head can be driven hard while
    the trunk takes smaller steps:

    ``training.pid_head_lr_multiplier``   scales the categorical head only
    ``training.backbone_lr_multiplier``   scales the backbone and the species
                                          embedding only

    With both at 1.0 no parameter groups are created at all, so the default
    optimizer is byte-for-byte the one this project has always used.

    ``training.freeze_backbone`` excludes the trunk from optimization entirely,
    for refitting the PID head on a fixed residual density.
    """
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    pid_multiplier = float(training.get("pid_head_lr_multiplier", 1.0))
    backbone_multiplier = float(training.get("backbone_lr_multiplier", 1.0))
    freeze_backbone = bool(training.get("freeze_backbone", False))

    if pid_multiplier == 1.0 and backbone_multiplier == 1.0 and not freeze_backbone:
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    trunk, pid_head, density = [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith(("backbone.", "species_embedding.")):
            trunk.append(parameter)
        elif name.startswith("pid_head."):
            pid_head.append(parameter)
        else:
            density.append(parameter)

    if freeze_backbone:
        for parameter in trunk:
            parameter.requires_grad_(False)

    groups: list[dict[str, Any]] = [
        {"params": density, "lr": learning_rate, "name": "density_heads"},
        {
            "params": pid_head,
            "lr": learning_rate * pid_multiplier,
            "name": "pid_head",
        },
    ]
    if not freeze_backbone:
        groups.insert(
            0,
            {
                "params": trunk,
                "lr": learning_rate * backbone_multiplier,
                "name": "backbone",
            },
        )
    return torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)


def warmup_cosine_factor(
    step: int,
    warmup_steps: int,
    total_steps: int,
    minimum_factor: float,
    decay: bool,
) -> float:
    """Learning-rate multiplier: linear warm-up, then optional cosine decay.

    The multiplier rises linearly from ``1/warmup_steps`` to 1 over the first
    ``warmup_steps`` optimizer steps, then either holds at 1 or decays as a
    cosine to ``minimum_factor``.

    The motivation is specific and measured, not decorative.  At the searched
    learning rate a large ``pid_loss_weight`` makes the first epochs unstable:
    one seed in six diverged far enough that early stopping returned a model
    from epoch 2.  Warm-up exists to test whether that early phase, rather than
    the weight itself, is what makes the setting unusable.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if not decay:
        return 1.0
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_factor + (1.0 - minimum_factor) * cosine


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
    optimizer = build_optimizer(model, training)
    schedule = str(training.get("lr_schedule", "none"))
    warmup_epochs = float(training.get("lr_warmup_epochs", 0.0))
    minimum_factor = float(training.get("lr_min_factor", 0.01))
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    step_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if warmup_epochs > 0.0:
        # Warm-up is a per-step schedule.  The instability it targets appears in
        # the first few hundred updates, well inside epoch one, so an
        # epoch-granularity scheduler cannot express it.
        steps_per_epoch = max(len(train_loader), 1)
        total_steps = steps_per_epoch * max_epochs
        warmup_steps = max(1, int(round(steps_per_epoch * warmup_epochs)))
        if warmup_steps >= total_steps:
            raise ValueError(
                f"lr_warmup_epochs={warmup_epochs} consumes the whole "
                f"{max_epochs}-epoch budget"
            )
        step_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: warmup_cosine_factor(
                step, warmup_steps, total_steps, minimum_factor, schedule == "cosine"
            ),
        )
    elif schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=float(training["learning_rate"]) * minimum_factor,
        )
    elif schedule != "none":
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
            step_scheduler=step_scheduler,
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
                "learning_rate": current_learning_rate(optimizer),
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
