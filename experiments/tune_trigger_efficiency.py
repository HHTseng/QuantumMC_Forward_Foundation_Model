#!/usr/bin/env python3
"""Validation-BCE Optuna search for P(T=1 | x_e); never reads test data."""
from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import optuna
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forwardfm_full.data import TRIGGER_FEATURES, load_trigger_splits
from forwardfm_full.efficiency import TriggerEfficiencyNet, count_parameters, train_model
from forwardfm_step1.config import load_config, resolve_run_dir
from forwardfm_step1.training import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--study-name", default="trigger-efficiency-capacity")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--worker-tag", default="worker")
    parser.add_argument("--sampler-seed", type=int)
    return parser.parse_args()


def suggest_config(base: dict[str, Any], trial: optuna.Trial) -> dict[str, Any]:
    config = copy.deepcopy(base)
    model = config["model"]
    training = config["training"]
    model["hidden_width"] = trial.suggest_categorical(
        "hidden_width", [128, 256, 512, 768]
    )
    model["hidden_layers"] = trial.suggest_categorical("hidden_layers", [3, 4, 6])
    model["dropout"] = trial.suggest_float("dropout", 0.0, 0.15)
    training["batch_size"] = trial.suggest_categorical(
        "batch_size", [8192, 16384, 32768]
    )
    training["learning_rate"] = trial.suggest_float(
        "learning_rate", 5.0e-4, 3.5e-3, log=True
    )
    training["weight_decay"] = trial.suggest_float(
        "weight_decay", 1.0e-6, 3.0e-4, log=True
    )
    training["lr_schedule"] = trial.suggest_categorical(
        "lr_schedule", ["none", "cosine"]
    )
    return config


class TrialRunner:
    def __init__(self, base: dict[str, Any], device: torch.device, worker: str) -> None:
        self.base = base
        self.device = device
        self.worker = worker
        self.splits, _scaler, _audit = load_trigger_splits(base)

    def __call__(self, trial: optuna.Trial) -> float:
        config = suggest_config(self.base, trial)
        seed = int(config["project"]["seed"])
        seed_everything(seed)
        torch.set_num_threads(int(config["training"]["torch_threads"]))
        model = TriggerEfficiencyNet(
            n_continuous=len(TRIGGER_FEATURES),
            hidden_width=int(config["model"]["hidden_width"]),
            hidden_layers=int(config["model"]["hidden_layers"]),
            dropout=float(config["model"]["dropout"]),
        )
        model, history, best_epoch, best_value = train_model(
            model, self.splits, config, self.device
        )
        trial.set_user_attr("worker", self.worker)
        trial.set_user_attr("parameters", count_parameters(model))
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("epochs_run", len(history))
        print(
            f"trial={trial.number} BCE_val={best_value:.7f} epoch={best_epoch}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
        return best_value


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
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
