#!/usr/bin/env python3
"""Validation-only Optuna search for the fixed β-informed D10 model.

The physics map, data split, targets, mixture size, and λ_PID=1 are invariant.
Trials minimize the particle-weighted momentum-binned PID total variation

    T_val = sum_(s,b) N_(s,b) TV_(s,b) / sum_(s,b) N_(s,b).

The test split is never evaluated by this program.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forwardfm_step1.config import load_config, resolve_run_dir
from forwardfm_step1.data import BASE_CONTINUOUS_FEATURES, SPECIES, load_all_splits
from forwardfm_step1.evaluation import (
    _raw_kinematics,
    conditional_pid_response_rows,
)
from forwardfm_step1.model import (
    ConditionalMDN,
    count_parameters,
    initialize_beta_input_from_control,
)
from forwardfm_step1.training import make_loader, run_epoch, seed_everything, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--study-name", default="beta-response-capacity")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--worker-tag", default="worker")
    parser.add_argument("--sampler-seed", type=int)
    return parser.parse_args()


def suggest_config(base: dict[str, Any], trial: optuna.Trial) -> dict[str, Any]:
    config = copy.deepcopy(base)
    model = config["model"]
    training = config["training"]
    model["hidden_width"] = trial.suggest_categorical("hidden_width", [256, 512, 768])
    model["hidden_layers"] = trial.suggest_categorical("hidden_layers", [4, 6])
    model["dropout"] = trial.suggest_float("dropout", 0.02, 0.15)
    training["batch_size"] = trial.suggest_categorical("batch_size", [4096, 8192])
    training["learning_rate"] = trial.suggest_float(
        "learning_rate", 1.0e-3, 3.5e-3, log=True
    )
    training["weight_decay"] = trial.suggest_float(
        "weight_decay", 1.0e-5, 3.0e-4, log=True
    )
    training["lr_schedule"] = trial.suggest_categorical(
        "lr_schedule", ["none", "cosine"]
    )
    return config


@torch.no_grad()
def pid_probabilities(
    model: ConditionalMDN,
    split: Any,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    loader = make_loader(split, batch_size, False, seed, 0)
    values: list[np.ndarray] = []
    model.eval()
    for continuous, species, _, _ in loader:
        logits = model(continuous.to(device), species.to(device)).pid_logits
        values.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(values)


class TrialRunner:
    def __init__(self, base: dict[str, Any], device: torch.device, worker: str) -> None:
        self.base = base
        self.device = device
        self.worker = worker
        (
            self.splits,
            self.feature_scaler,
            _target_scaler,
            self.rec_pid_vocabulary,
            _audit,
        ) = load_all_splits(base)

    def __call__(self, trial: optuna.Trial) -> float:
        config = suggest_config(self.base, trial)
        seed = int(config["project"]["seed"])
        seed_everything(seed)
        torch.set_num_threads(int(config["training"]["torch_threads"]))
        feature_names = self.splits["train"].feature_names
        target_names = self.splits["train"].target_names
        arguments = {
            "n_continuous": len(feature_names),
            "n_species": len(SPECIES),
            "n_rec_pid_classes": len(self.rec_pid_vocabulary) + 1,
            "hidden_width": int(config["model"]["hidden_width"]),
            "hidden_layers": int(config["model"]["hidden_layers"]),
            "pid_embedding_dim": int(config["model"]["pid_embedding_dim"]),
            "mixture_components": int(config["model"]["mixture_components"]),
            "target_dim": len(target_names),
            "dropout": float(config["model"]["dropout"]),
        }
        model = ConditionalMDN(**arguments)
        model.reset_parameters(seed=seed)
        control_arguments = dict(arguments)
        control_arguments["n_continuous"] = len(BASE_CONTINUOUS_FEATURES)
        control = ConditionalMDN(**control_arguments)
        control.reset_parameters(seed=seed)
        initialize_beta_input_from_control(control, model)
        seed_everything(seed)

        start = time.perf_counter()
        model, history, best_epoch, _candidates = train_model(
            model, self.splits, config, self.device
        )
        validation = self.splits["validation"]
        loader = make_loader(
            validation,
            int(config["training"]["batch_size"]),
            False,
            seed,
            0,
        )
        metrics = run_epoch(
            model,
            loader,
            self.device,
            float(config["training"]["pid_loss_weight"]),
        )
        probabilities = pid_probabilities(
            model,
            validation,
            int(config["training"]["batch_size"]),
            seed,
            self.device,
        )
        momentum = _raw_kinematics(validation, self.feature_scaler)["gen_p"]
        labels: list[int | str] = [*self.rec_pid_vocabulary, "OTHER"]
        _rows, summary = conditional_pid_response_rows(
            validation.raw_species,
            momentum,
            validation.rec_pid_index,
            probabilities,
            labels,
            np.asarray(config["evaluation"]["pid_momentum_edges_gev"]),
        )
        counts = np.asarray([row["n"] for row in summary], dtype=np.float64)
        tv = np.asarray(
            [row["total_variation_distance"] for row in summary], dtype=np.float64
        )
        objective = float(np.average(tv, weights=counts))
        species_tv = {
            label: float(
                np.average(
                    tv[[row["generated_species"] == label for row in summary]],
                    weights=counts[
                        [row["generated_species"] == label for row in summary]
                    ],
                )
            )
            for label in ("pi-", "pi+", "proton")
        }
        trial.set_user_attr("worker", self.worker)
        trial.set_user_attr("parameters", count_parameters(model))
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("epochs_run", len(history))
        trial.set_user_attr("validation_pid_cross_entropy", metrics.pid_cross_entropy)
        trial.set_user_attr("validation_pid_accuracy", metrics.pid_accuracy)
        trial.set_user_attr("validation_residual_nll", metrics.residual_nll)
        trial.set_user_attr("validation_pid_tv_by_species", json.dumps(species_tv))
        trial.set_user_attr("seconds", time.perf_counter() - start)
        print(
            f"trial={trial.number} T_val={objective:.6f} "
            f"CE_val={metrics.pid_cross_entropy:.6f} epoch={best_epoch}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
        return objective


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    if float(base["training"]["pid_loss_weight"]) != 1.0:
        raise ValueError("This controlled search requires pid_loss_weight=1")
    if not base["data"]["beta_response"]["include_generated_beta"]:
        raise ValueError("This controlled search requires beta_gen input")
    if not base["data"]["beta_response"]["enabled"]:
        raise ValueError("This controlled search requires delta_beta target")
    resolve_run_dir(base).mkdir(parents=True, exist_ok=True)
    sampler_seed = args.sampler_seed
    if sampler_seed is None:
        offset = int(hashlib.sha256(args.worker_tag.encode()).hexdigest()[:8], 16)
        sampler_seed = (int(base["project"]["seed"]) + offset) % (2**31)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=sampler_seed, multivariate=True, group=True, n_startup_trials=8
        ),
        load_if_exists=True,
    )
    study.optimize(
        TrialRunner(base, torch.device(args.device), args.worker_tag),
        n_trials=args.n_trials,
        gc_after_trial=True,
        catch=(torch.cuda.OutOfMemoryError,),
    )


if __name__ == "__main__":
    main()
