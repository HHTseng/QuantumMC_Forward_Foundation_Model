#!/usr/bin/env python3
"""Export a completed trigger-efficiency Optuna study and selected YAML."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import optuna
import yaml

from tune_trigger_efficiency import suggest_config

from forwardfm_step1.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--study-name", default="trigger-efficiency-capacity")
    parser.add_argument("--output-dir", default="runs/trigger_optuna_analysis")
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
    fieldnames = ["trial", "validation_bce", "best_epoch", "parameters", *sorted(best.params)]
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in complete:
            writer.writerow(
                {
                    "trial": trial.number,
                    "validation_bce": trial.value,
                    "best_epoch": trial.user_attrs["best_epoch"],
                    "parameters": trial.user_attrs["parameters"],
                    **trial.params,
                }
            )
    summary = {
        "study_name": study.study_name,
        "objective": "validation binary cross-entropy",
        "selection_used_test_split": False,
        "n_trials": len(study.trials),
        "n_complete": len(complete),
        "best_trial": best.number,
        "best_validation_bce": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = suggest_config(load_config(args.config), best)
    config.pop("_config_path", None)
    config["project"]["name"] = "trigger-electron-optuna-best"
    config["training"]["epochs"] = 50
    config["training"]["early_stopping_patience"] = 10
    config["output"]["run_dir"] = "runs/trigger_optuna_best"
    with (output / "best_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
