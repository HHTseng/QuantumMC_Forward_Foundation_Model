#!/usr/bin/env python3
"""How much of the closure metrics is Monte-Carlo noise rather than model quality?

The two closure statistics used to select and compare models are not equally
reliable, and the difference matters when the reported gaps are small.

``pid_closure_tv`` is computed from the *mean PID-head softmax probability*, so
for a fixed checkpoint it is deterministic: re-sampling cannot move it.

``moment_closure_error`` is computed from one stochastic draw per particle, and
its width term ``Std[Delta]_model / Std[Delta]_obs`` is a second moment. A
mixture density with heavy tails produces occasional very large draws, and a
sample standard deviation is sensitive to exactly those, so this statistic can
carry real Monte-Carlo noise even with 10^5 particles.

This script re-samples a saved checkpoint under several sampling seeds, holding
the weights and the data fixed, and reports the spread of every closure number.
That spread is the resolution floor: differences smaller than it say nothing
about model quality.

Usage:

    python experiments/closure_sampling_uncertainty.py \
        --run-dir runs/optuna_best --split validation --draws 8 \
        --output-dir runs/optuna_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.tune_hyperparameters import (
    moment_closure_error,
    pid_closure_total_variation,
)
from forwardfm_step1.config import load_config
from forwardfm_step1.data import (
    CONTINUOUS_FEATURES,
    SPECIES,
    TARGET_COLUMNS,
    Standardizer,
    load_all_splits,
)
from forwardfm_step1.evaluation import (
    _raw_kinematics,
    closure_rows,
    conditional_pid_response_rows,
    predict_test_sample,
)
from forwardfm_step1.model import ConditionalMDN
from forwardfm_step1.training import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import yaml

    with (run_dir / "resolved_config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str((REPOSITORY_ROOT / "configs" / "placeholder.yaml").resolve())
    device = choose_device(args.device)

    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = ConditionalMDN(
        n_continuous=len(CONTINUOUS_FEATURES),
        n_species=len(SPECIES),
        n_rec_pid_classes=architecture["n_rec_pid_classes"],
        hidden_width=architecture["hidden_width"],
        hidden_layers=architecture["hidden_layers"],
        pid_embedding_dim=architecture["pid_embedding_dim"],
        mixture_components=architecture["mixture_components"],
        target_dim=architecture["target_dim"],
        dropout=architecture["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    splits, feature_scaler, target_scaler, rec_pid_vocabulary, _audit = load_all_splits(config)
    split = splits[args.split]
    feature_scaler = Standardizer.from_dict(checkpoint["feature_scaler"])
    target_scaler = Standardizer.from_dict(checkpoint["target_scaler"])
    pid_labels: list[int | str] = [*rec_pid_vocabulary, "OTHER"]
    edges = np.asarray(config["evaluation"]["pid_momentum_edges_gev"], dtype=np.float64)
    generated_momentum = _raw_kinematics(split, feature_scaler)["gen_p"]
    batch_size = int(config["training"]["batch_size"])

    rows: list[dict[str, Any]] = []
    for draw in range(args.draws):
        sampled, _predicted, probabilities = predict_test_sample(
            model, split, target_scaler, device, batch_size, seed=1000 + draw
        )
        moment, _cells = moment_closure_error(closure_rows(split, target_scaler, sampled))
        _response, summary = conditional_pid_response_rows(
            split.raw_species,
            generated_momentum,
            split.rec_pid_index,
            probabilities,
            pid_labels,
            edges,
        )
        weighted_tv, max_tv, _by_species = pid_closure_total_variation(summary)
        rows.append(
            {
                "draw": draw,
                "sampling_seed": 1000 + draw,
                "moment_closure_error": moment,
                "pid_closure_tv": weighted_tv,
                "pid_closure_tv_max": max_tv,
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    name = run_dir.name
    csv_path = output_dir / f"closure_sampling_uncertainty_{name}_{args.split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_stats = {"run_dir": str(run_dir), "split": args.split, "draws": args.draws}
    for key in ("moment_closure_error", "pid_closure_tv", "pid_closure_tv_max"):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        summary_stats[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "peak_to_peak": float(values.max() - values.min()),
        }
    with (output_dir / f"closure_sampling_uncertainty_{name}_{args.split}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary_stats, handle, indent=2)
    print(json.dumps(summary_stats, indent=2))


if __name__ == "__main__":
    main()
