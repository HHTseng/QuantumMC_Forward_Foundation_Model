#!/usr/bin/env python3
"""Export a completed β-response Optuna study and its selected YAML."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import optuna
import yaml

from tune_beta_response import suggest_config

from forwardfm_step1.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--study-name", default="beta-response-capacity")
    parser.add_argument("--output-dir", default="runs/beta_optuna_analysis")
    args = parser.parse_args()
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    complete = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        raise RuntimeError("The study has no completed trials")
    best = min(complete, key=lambda trial: float(trial.value))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "validation_pid_tv",
        "validation_pid_cross_entropy",
        "validation_pid_accuracy",
        "best_epoch",
        "parameters",
        *sorted(best.params),
    ]
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in complete:
            row = {
                "trial": trial.number,
                "validation_pid_tv": trial.value,
                "validation_pid_cross_entropy": trial.user_attrs[
                    "validation_pid_cross_entropy"
                ],
                "validation_pid_accuracy": trial.user_attrs["validation_pid_accuracy"],
                "best_epoch": trial.user_attrs["best_epoch"],
                "parameters": trial.user_attrs["parameters"],
                **trial.params,
            }
            writer.writerow(row)
    summary = {
        "study_name": study.study_name,
        "objective": "validation particle-weighted momentum-binned PID total variation",
        "selection_used_test_split": False,
        "n_trials": len(study.trials),
        "n_complete": len(complete),
        "best_trial": best.number,
        "best_validation_pid_tv": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = suggest_config(load_config(args.config), best)
    config.pop("_config_path", None)
    config["project"]["name"] = "beta-response-optuna-best"
    config["training"]["epochs"] = 70
    config["training"]["early_stopping_patience"] = 70
    config["output"]["run_dir"] = "runs/beta_optuna_best"
    with (output / "best_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
