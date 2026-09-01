#!/usr/bin/env python3
"""Isolate the effect of ``pid_loss_weight`` at a fixed architecture.

The Optuna search varies every hyper-parameter at once, so it cannot answer
"does a larger PID weight help?" on its own: a good trial might simply have a
better width or schedule.  This scan holds the whole configuration fixed and
moves only

    L = L_response + lambda_PID L_PID,

which makes the trade-off between the two terms directly measurable.

Everything is evaluated on the *validation* split.  The test split stays
untouched, because a scan is a selection procedure and reporting it on test
would bias the final held-out numbers.

Usage:

    python experiments/scan_pid_weight.py --config configs/gpu_optuna_best.yaml \
        --weights 0.05,0.2,0.5,1,2,5,10 --device cuda:0 \
        --output-dir runs/optuna_analysis
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.tune_hyperparameters import evaluate_validation_closure
from forwardfm_step1.config import load_config
from forwardfm_step1.data import CONTINUOUS_FEATURES, SPECIES, TARGET_COLUMNS, load_all_splits
from forwardfm_step1.model import ConditionalMDN, count_parameters
from forwardfm_step1.training import (
    build_loader,
    choose_device,
    run_epoch,
    seed_everything,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default="0.05,0.1,0.2,0.5,1,2,5,10")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="scan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_config(args.config)
    device = choose_device(args.device)
    weights = [float(value) for value in args.weights.split(",")]

    splits, feature_scaler, target_scaler, rec_pid_vocabulary, _audit = load_all_splits(base)
    print(
        "loaded " + ", ".join(f"{name}={len(split):,}" for name, split in splits.items()),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for weight in weights:
        config = copy.deepcopy(base)
        config["training"]["pid_loss_weight"] = weight
        seed = int(config["project"]["seed"])
        seed_everything(seed)
        torch.set_num_threads(int(config["training"]["torch_threads"]))
        model = ConditionalMDN(
            n_continuous=len(CONTINUOUS_FEATURES),
            n_species=len(SPECIES),
            n_rec_pid_classes=len(rec_pid_vocabulary) + 1,
            hidden_width=int(config["model"]["hidden_width"]),
            hidden_layers=int(config["model"]["hidden_layers"]),
            pid_embedding_dim=int(config["model"]["pid_embedding_dim"]),
            mixture_components=int(config["model"]["mixture_components"]),
            target_dim=len(TARGET_COLUMNS),
            dropout=float(config["model"]["dropout"]),
        )
        start = time.perf_counter()
        model, history, best_epoch = train_model(model, splits, config, device)
        loader = build_loader(
            splits["validation"],
            int(config["training"]["batch_size"]),
            shuffle=False,
            seed=seed,
            num_workers=0,
            device=device,
            fast=bool(config["training"].get("fast_loader", False)),
        )
        metrics = run_epoch(model, loader, device, weight)
        closure = evaluate_validation_closure(
            model,
            splits["validation"],
            feature_scaler,
            target_scaler,
            rec_pid_vocabulary,
            config,
            device,
        )
        row = {
            "pid_loss_weight": weight,
            "trainable_parameters": count_parameters(model),
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "train_seconds": time.perf_counter() - start,
            "validation_residual_nll": metrics.residual_nll,
            "validation_pid_cross_entropy": metrics.pid_cross_entropy,
            "validation_joint_nll": metrics.joint_nll,
            "validation_pid_accuracy": metrics.pid_accuracy,
            "pid_closure_tv": closure["pid_closure_tv"],
            "pid_closure_tv_max": closure["pid_closure_tv_max"],
            "pid_marginal_discrepancy": closure["pid_marginal_discrepancy"],
            "moment_closure_error": closure["moment_closure_error"],
        }
        for species, value in closure["pid_closure_tv_by_species"].items():
            row[f"pid_tv_{species}"] = value
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = output_dir / f"pid_weight_{args.tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    values = np.array([row["pid_loss_weight"] for row in rows])
    figure, axes = plt.subplots(1, 4, figsize=(19.0, 4.3))
    panels = (
        ("validation_residual_nll", "validation residual NLL", "#2f855a"),
        ("validation_pid_cross_entropy", "validation PID cross entropy", "#c05621"),
        ("pid_closure_tv", "PID weighted mean TV", "#2b6cb0"),
        ("moment_closure_error", "moment closure error", "#6b46c1"),
    )
    for panel, (key, title, color) in zip(axes, panels):
        series = np.array([row[key] for row in rows])
        panel.plot(values, series, "-o", color=color)
        best = int(np.argmin(series))
        panel.plot(values[best], series[best], "*", ms=16, color="#c53030")
        panel.set_xscale("log")
        panel.set_xlabel(r"$\lambda_{\mathrm{PID}}$")
        panel.set_ylabel(title)
        panel.grid(alpha=0.3)
        panel.set_title(f"minimum at $\\lambda={values[best]:g}$", fontsize=10)
    figure.suptitle(
        r"Effect of $\lambda_{\mathrm{PID}}$ at a fixed architecture "
        "(validation split; the star marks the minimum)",
        y=1.0,
    )
    figure.tight_layout()
    figure.savefig(output_dir / f"pid_weight_{args.tag}.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
