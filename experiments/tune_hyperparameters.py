#!/usr/bin/env python3
"""Optuna search over the step-one FD response model's capacity and schedule.

Motivation
----------
The published baselines fix a four-layer width-256 backbone, eight mixture
components, thirty epochs, and ``pid_loss_weight=0.20``.  None of those numbers
was ever searched.  This driver searches them jointly on the same deterministic,
event-disjoint splits.

Objective
---------
Trials are ranked by the *validation joint negative log likelihood*

    J = -1/N sum_i log q_theta(Delta_i | x_i)
        -1/N sum_i log q_theta(s_rec,i | x_i)
      = residual_nll + pid_cross_entropy,

evaluated at each trial's own early-stopping checkpoint.

Using ``total_loss = residual_nll + lambda_PID * pid_cross_entropy`` would be
wrong here: ``lambda_PID`` is itself a search dimension, so a trial could lower
its score simply by shrinking the weight on the PID term rather than by fitting
the data better.  ``J`` is the unweighted log density of the factorized model
q(Delta|x) q(s_rec|x) and is therefore comparable across every trial.

``J`` also has the property the search needs: it is smooth, low variance, and
available every epoch, so it supports median pruning.

Physics closure is *not* used to drive the search, because the sampled closure
statistics are Monte-Carlo noisy and much more expensive.  Instead every trial
records them as user attributes:

``pid_closure_tv``
    Particle-weighted mean total-variation distance between the COATJAVA
    reconstructed-class fractions and the mean PID-head softmax response, over
    generated species and fixed 1 GeV generated-momentum bins.

``moment_closure_error``
    Mean over generated species and the three residual targets of
    ``|mean_model - mean_obs| / std_obs + |std_model / std_obs - 1|``, i.e. a
    dimensionless first- and second-moment closure error.

The final checkpoint is then chosen from the top-``--closure-top-k`` trials by
``J`` using the closure composite; see ``experiments/analyze_tuning.py``.

Parallelism
-----------
Run one process per GPU against a shared SQLite study:

    python experiments/tune_hyperparameters.py --config configs/gpu_optuna_search.yaml \
        --storage sqlite:///runs/optuna/study.db --device cuda:0 --n-trials 40 &
    python experiments/tune_hyperparameters.py --config configs/gpu_optuna_search.yaml \
        --storage sqlite:///runs/optuna/study.db --device cuda:1 --n-trials 40 &

Every worker loads the splits once and reuses them for all of its trials, so the
DuckDB selection cost is paid once per process rather than once per trial.
"""
from __future__ import annotations

import argparse
import sys
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch

# Allow execution as `python experiments/<script>.py` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from forwardfm_step1.config import load_config, resolve_run_dir
from forwardfm_step1.data import (
    CONTINUOUS_FEATURES,
    SPECIES,
    TARGET_COLUMNS,
    PreparedSplit,
    Standardizer,
    load_all_splits,
)
from forwardfm_step1.evaluation import (
    SPECIES_LABELS,
    _raw_kinematics,
    closure_rows,
    conditional_pid_response_rows,
    predict_test_sample,
)
from forwardfm_step1.model import ConditionalMDN, count_parameters
from forwardfm_step1.training import (
    EpochMetrics,
    build_loader,
    choose_device,
    run_epoch,
    seed_everything,
    train_model,
)


# Ranked capacity ladder.  The published baseline (256 wide, 4 deep, K=8,
# embedding 16) sits inside the grid so the search can reproduce it.
HIDDEN_WIDTHS = (128, 256, 384, 512, 768, 1024)
HIDDEN_LAYERS = (3, 4, 5, 6, 7, 8)
EMBEDDING_DIMS = (8, 16, 32)
MIXTURE_COMPONENTS = (5, 8, 12, 16, 24)
BATCH_SIZES = (4096, 8192, 16384)
EPOCH_BUDGETS = (40, 70, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Base YAML configuration")
    parser.add_argument("--storage", required=True, help="Optuna storage URL")
    parser.add_argument("--study-name", default="forwardfm-step1-capacity")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=None, help="Seconds")
    parser.add_argument(
        "--worker-tag", default="worker", help="Recorded on every trial for provenance"
    )
    return parser.parse_args()


def suggest_config(base: dict[str, Any], trial: optuna.Trial) -> dict[str, Any]:
    """Overlay one sampled point of the search space onto the base config."""
    config = copy.deepcopy(base)
    model = config["model"]
    training = config["training"]

    model["hidden_width"] = trial.suggest_categorical("hidden_width", HIDDEN_WIDTHS)
    model["hidden_layers"] = trial.suggest_categorical("hidden_layers", HIDDEN_LAYERS)
    model["pid_embedding_dim"] = trial.suggest_categorical(
        "pid_embedding_dim", EMBEDDING_DIMS
    )
    model["mixture_components"] = trial.suggest_categorical(
        "mixture_components", MIXTURE_COMPONENTS
    )
    model["dropout"] = trial.suggest_float("dropout", 0.0, 0.15)

    training["epochs"] = trial.suggest_categorical("epochs", EPOCH_BUDGETS)
    training["batch_size"] = trial.suggest_categorical("batch_size", BATCH_SIZES)
    training["learning_rate"] = trial.suggest_float("learning_rate", 2e-4, 4e-3, log=True)
    training["weight_decay"] = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    # The physics motivation for widening this range is that the PID term is the
    # only supervision for reconstructed-species contamination, and the baseline
    # value of 0.20 was assumed rather than measured.
    training["pid_loss_weight"] = trial.suggest_float(
        "pid_loss_weight", 0.05, 10.0, log=True
    )
    training["lr_schedule"] = trial.suggest_categorical("lr_schedule", ("none", "cosine"))
    return config


def moment_closure_error(rows: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    """Dimensionless first/second-moment closure error of sampled residuals.

    For generated species ``s`` and residual target ``t`` the per-cell error is

        |E[Delta]_model - E[Delta]_obs| / Std[Delta]_obs
        + |Std[Delta]_model / Std[Delta]_obs - 1|,

    which is invariant under the GeV/radian units of each target and therefore
    can be averaged over the three targets.
    """
    per_cell: dict[str, float] = {}
    values: list[float] = []
    for row in rows:
        observed_std = float(row["observed_std"])
        if not np.isfinite(observed_std) or observed_std <= 0:
            continue
        bias = abs(float(row["sampled_mean"]) - float(row["observed_mean"])) / observed_std
        width = abs(float(row["std_ratio"]) - 1.0)
        error = bias + width
        per_cell[f"{row['species']}|{row['target']}"] = error
        values.append(error)
    return (float(np.mean(values)) if values else float("nan")), per_cell


def pid_closure_total_variation(
    summary_rows: list[dict[str, Any]],
) -> tuple[float, float, dict[str, float]]:
    """Particle-weighted mean, maximum, and per-species mean TV distance."""
    weights = np.array([row["n"] for row in summary_rows], dtype=np.float64)
    distances = np.array(
        [row["total_variation_distance"] for row in summary_rows], dtype=np.float64
    )
    per_species: dict[str, float] = {}
    for label in SPECIES_LABELS.values():
        mask = np.array([row["generated_species"] == label for row in summary_rows])
        if mask.any():
            per_species[label] = float(
                np.average(distances[mask], weights=weights[mask])
            )
    return (
        float(np.average(distances, weights=weights)),
        float(distances.max()),
        per_species,
    )


def evaluate_validation_closure(
    model: ConditionalMDN,
    split: PreparedSplit,
    feature_scaler: Standardizer,
    target_scaler: Standardizer,
    rec_pid_vocabulary: list[int],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Physics closure diagnostics on the validation split.

    The test split is deliberately untouched during the search so that the
    reported held-out numbers of the final model are not selection biased.
    """
    seed = int(config["project"]["seed"])
    batch_size = int(config["training"]["batch_size"])
    sampled_targets, _sampled_pid, pid_probabilities = predict_test_sample(
        model, split, target_scaler, device, batch_size, seed
    )
    rows = closure_rows(split, target_scaler, sampled_targets)
    moment_error, moment_cells = moment_closure_error(rows)

    pid_labels: list[int | str] = [*rec_pid_vocabulary, "OTHER"]
    edges = np.asarray(
        config["evaluation"]["pid_momentum_edges_gev"], dtype=np.float64
    )
    generated_momentum = _raw_kinematics(split, feature_scaler)["gen_p"]
    _response_rows, summary_rows = conditional_pid_response_rows(
        split.raw_species,
        generated_momentum,
        split.rec_pid_index,
        pid_probabilities,
        pid_labels,
        edges,
    )
    weighted_tv, max_tv, per_species_tv = pid_closure_total_variation(summary_rows)

    observed_fractions = np.bincount(
        split.rec_pid_index, minlength=len(pid_labels)
    ) / len(split)
    marginal_discrepancy = float(
        np.max(np.abs(observed_fractions - pid_probabilities.mean(axis=0)))
    )

    correct_probability: dict[str, float] = {}
    label_to_index = {label: index for index, label in enumerate(pid_labels)}
    for species, label in SPECIES_LABELS.items():
        if species not in label_to_index:
            continue
        mask = split.raw_species == species
        correct_probability[label] = float(
            pid_probabilities[mask, label_to_index[species]].mean()
        )

    return {
        "moment_closure_error": moment_error,
        "moment_closure_cells": moment_cells,
        "pid_closure_tv": weighted_tv,
        "pid_closure_tv_max": max_tv,
        "pid_closure_tv_by_species": per_species_tv,
        "pid_marginal_discrepancy": marginal_discrepancy,
        "pid_correct_mean_probability": correct_probability,
        "closure_rows": rows,
    }


class TrialRunner:
    """Holds the splits so that every trial in a worker reuses one data load."""

    def __init__(self, base_config: dict[str, Any], device: torch.device, tag: str) -> None:
        self.base_config = base_config
        self.device = device
        self.tag = tag
        start = time.perf_counter()
        (
            self.splits,
            self.feature_scaler,
            self.target_scaler,
            self.rec_pid_vocabulary,
            self.audit,
        ) = load_all_splits(base_config)
        print(
            "loaded "
            + ", ".join(f"{name}={len(split):,}" for name, split in self.splits.items())
            + f" in {time.perf_counter() - start:.2f}s",
            flush=True,
        )

    def __call__(self, trial: optuna.Trial) -> float:
        config = suggest_config(self.base_config, trial)
        seed = int(config["project"]["seed"])
        # One fixed seed for every trial, so score differences are attributable
        # to the hyper-parameters rather than to initialization noise.  Seed
        # sensitivity of the selected configuration is measured separately.
        seed_everything(seed)
        torch.set_num_threads(int(config["training"]["torch_threads"]))

        model = ConditionalMDN(
            n_continuous=len(CONTINUOUS_FEATURES),
            n_species=len(SPECIES),
            n_rec_pid_classes=len(self.rec_pid_vocabulary) + 1,
            hidden_width=int(config["model"]["hidden_width"]),
            hidden_layers=int(config["model"]["hidden_layers"]),
            pid_embedding_dim=int(config["model"]["pid_embedding_dim"]),
            mixture_components=int(config["model"]["mixture_components"]),
            target_dim=len(TARGET_COLUMNS),
            dropout=float(config["model"]["dropout"]),
        )
        parameters = count_parameters(model)
        trial.set_user_attr("trainable_parameters", parameters)
        trial.set_user_attr("worker", self.tag)
        trial.set_user_attr("device", str(self.device))

        def epoch_callback(
            epoch: int, _train: EpochMetrics, validation: EpochMetrics
        ) -> None:
            trial.report(validation.joint_nll, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"pruned at epoch {epoch} with joint_nll={validation.joint_nll:.5f}"
                )

        wall_start = time.perf_counter()
        model, history, best_epoch = train_model(
            model, self.splits, config, self.device, epoch_callback=epoch_callback
        )
        train_seconds = time.perf_counter() - wall_start

        validation_loader = build_loader(
            self.splits["validation"],
            int(config["training"]["batch_size"]),
            shuffle=False,
            seed=seed,
            num_workers=0,
            device=self.device,
            fast=bool(config["training"].get("fast_loader", False)),
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            self.device,
            float(config["training"]["pid_loss_weight"]),
        )
        closure = evaluate_validation_closure(
            model,
            self.splits["validation"],
            self.feature_scaler,
            self.target_scaler,
            self.rec_pid_vocabulary,
            config,
            self.device,
        )

        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("epochs_run", len(history))
        trial.set_user_attr("train_seconds", train_seconds)
        trial.set_user_attr("validation_residual_nll", validation_metrics.residual_nll)
        trial.set_user_attr(
            "validation_pid_cross_entropy", validation_metrics.pid_cross_entropy
        )
        trial.set_user_attr("validation_pid_accuracy", validation_metrics.pid_accuracy)
        trial.set_user_attr("validation_joint_nll", validation_metrics.joint_nll)
        for key in (
            "moment_closure_error",
            "pid_closure_tv",
            "pid_closure_tv_max",
            "pid_marginal_discrepancy",
        ):
            trial.set_user_attr(key, closure[key])
        trial.set_user_attr(
            "pid_closure_tv_by_species", json.dumps(closure["pid_closure_tv_by_species"])
        )
        trial.set_user_attr(
            "pid_correct_mean_probability",
            json.dumps(closure["pid_correct_mean_probability"]),
        )
        trial.set_user_attr("moment_closure_cells", json.dumps(closure["moment_closure_cells"]))
        trial.set_user_attr("history", json.dumps(history))

        print(
            f"trial={trial.number} params={parameters:,} best_epoch={best_epoch} "
            f"J={validation_metrics.joint_nll:.5f} "
            f"pid_tv={closure['pid_closure_tv']:.5f} "
            f"moment={closure['moment_closure_error']:.5f} "
            f"({train_seconds:.0f}s)",
            flush=True,
        )
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return validation_metrics.joint_nll


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    run_dir = resolve_run_dir(base_config)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    sampler = optuna.samplers.TPESampler(
        seed=int(base_config["project"]["seed"]) % (2**31),
        multivariate=True,
        group=True,
        n_startup_trials=16,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=8, n_warmup_steps=12, interval_steps=2
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    runner = TrialRunner(base_config, device, args.worker_tag)
    study.optimize(
        runner,
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        catch=(torch.cuda.OutOfMemoryError,),
    )
    print(f"worker {args.worker_tag} finished; study has {len(study.trials)} trials")


if __name__ == "__main__":
    main()
