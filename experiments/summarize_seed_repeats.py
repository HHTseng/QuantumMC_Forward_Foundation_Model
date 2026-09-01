#!/usr/bin/env python3
"""Aggregate repeated-seed runs of the baseline and the tuned configuration.

A single seed cannot separate a real architectural gain from initialization
luck.  This reads the ``metrics.json`` of several runs per configuration and
reports, for every held-out quantity, the mean, the sample standard deviation,
and whether the tuned mean separates from the baseline mean by more than the
combined spread.

Usage:

    python experiments/summarize_seed_repeats.py \
        --group baseline=runs/seed_baseline_20260822,runs/seed_baseline_7,... \
        --group tuned=runs/seed_tuned_20260822,runs/seed_tuned_7,... \
        --output-dir runs/optuna_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

METRICS = (
    ("test_residual_nll", "residual NLL", True),
    ("test_pid_cross_entropy", "PID cross entropy", True),
    ("test_joint_nll", "joint NLL", True),
    ("test_pid_accuracy", "PID top-1 accuracy", False),
    ("pid_weighted_mean_tv", "PID weighted mean TV", True),
    ("moment_closure_error", "moment closure error", True),
    ("physical_sample_fraction", r"physical $(p,\theta)$ fraction", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", required=True, metavar="LABEL=DIR,DIR,...")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def run_values(path: Path) -> dict[str, float]:
    metrics = json.loads((path / "metrics.json").read_text())
    test = metrics["test"]
    summary = metrics["pid_conditional_closure"]["bin_summary"]
    weights = np.array([row["n"] for row in summary], dtype=np.float64)
    tv = np.array([row["total_variation_distance"] for row in summary], dtype=np.float64)
    moment = []
    for row in metrics["closure"]:
        observed_std = float(row["observed_std"])
        moment.append(
            abs(float(row["sampled_mean"]) - float(row["observed_mean"])) / observed_std
            + abs(float(row["std_ratio"]) - 1.0)
        )
    return {
        "test_residual_nll": test["residual_nll"],
        "test_pid_cross_entropy": test["pid_cross_entropy"],
        "test_joint_nll": test["residual_nll"] + test["pid_cross_entropy"],
        "test_pid_accuracy": test["pid_accuracy"],
        "pid_weighted_mean_tv": float(np.average(tv, weights=weights)),
        "pid_max_bin_tv": float(tv.max()),
        "moment_closure_error": float(np.mean(moment)),
        "physical_sample_fraction": metrics["joint_and_physical"][
            "physical_sample_fraction"
        ],
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[dict[str, float]]] = {}
    per_run_rows: list[dict[str, Any]] = []
    for entry in args.group:
        label, _, paths = entry.partition("=")
        groups[label] = []
        for raw in paths.split(","):
            path = Path(raw.strip())
            values = run_values(path)
            groups[label].append(values)
            per_run_rows.append({"group": label, "run_dir": str(path), **values})

    summary_rows: list[dict[str, Any]] = []
    keys = [key for key, _title, _lower in METRICS] + ["pid_max_bin_tv"]
    reference = list(groups)[0]
    for key in keys:
        row: dict[str, Any] = {"metric": key}
        for label, runs in groups.items():
            values = np.array([run[key] for run in runs], dtype=np.float64)
            row[f"{label}_mean"] = float(values.mean())
            row[f"{label}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{label}_n"] = len(values)
        for label in groups:
            if label == reference:
                continue
            difference = row[f"{label}_mean"] - row[f"{reference}_mean"]
            spread = np.hypot(row[f"{label}_std"], row[f"{reference}_std"])
            row[f"{label}_minus_{reference}"] = difference
            row[f"{label}_separation_in_combined_std"] = (
                float(abs(difference) / spread) if spread > 0 else float("inf")
            )
        summary_rows.append(row)

    with (output_dir / "seed_repeat_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_run_rows[0]))
        writer.writeheader()
        writer.writerows(per_run_rows)
    with (output_dir / "seed_repeat_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields: list[str] = []
        for row in summary_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    labels = list(groups)
    plotted = [item for item in METRICS]
    figure, axes_grid = plt.subplots(2, 4, figsize=(18.0, 7.4))
    for axes, (key, title, lower_better) in zip(axes_grid.ravel(), plotted):
        for index, label in enumerate(labels):
            values = [run[key] for run in groups[label]]
            axes.scatter([index] * len(values), values, s=42, color="#2b6cb0", zorder=3)
            axes.hlines(
                np.mean(values), index - 0.22, index + 0.22, color="#c53030", lw=2, zorder=4
            )
        axes.set_xticks(range(len(labels)), labels, fontsize=9)
        axes.set_title(f"{title}\n({'lower' if lower_better else 'higher'} is better)", fontsize=10)
        axes.grid(alpha=0.25, axis="y")
    axes_grid.ravel()[len(plotted)].axis("off")
    figure.suptitle(
        "Seed repeats: each point is one run, the bar is the group mean", y=1.0
    )
    figure.tight_layout()
    figure.savefig(output_dir / "seed_repeat_spread.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
