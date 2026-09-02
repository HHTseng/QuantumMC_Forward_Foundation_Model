#!/usr/bin/env python3
"""Compare strategies for reaching the large-PID-weight solution.

Section 13.4 established that ``pid_loss_weight >= 2`` sometimes reaches a
distinctly better solution and sometimes destabilizes. This compares the
attempts to make that solution reachable deliberately rather than by accident:

* the released recipe, as the reference;
* a large weight from scratch, the accidental version;
* a large weight applied by fine-tuning an already-trained released checkpoint,
  so there is no fragile early phase to survive;
* a large weight with the shared trunk on a smaller learning rate than the
  heads, so the PID term can be strong without large trunk updates.

Each group is one point per seed.  A run counts as having reached the better
solution when its held-out PID top-1 clears ``--accuracy-threshold``, and as a
failure when its held-out joint NLL is more than ``--failure-margin`` nats worse
than the mean of the first group, which is treated as the reference.

Failure is deliberately defined on the outcome rather than on early stopping.
An early best epoch means opposite things in the two settings compared here: for
a from-scratch run it signals divergence, but for a run warm started from an
already-trained checkpoint it only means fine-tuning did not improve on the
starting point, which is a null result and not a failure.
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

PALETTE = ("#2b6cb0", "#c53030", "#2f855a", "#6b46c1", "#c05621")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", required=True, metavar="LABEL=DIR,DIR,...")
    parser.add_argument("--accuracy-threshold", type=float, default=0.70)
    parser.add_argument(
        "--failure-margin",
        type=float,
        default=0.5,
        help="Nats of held-out joint NLL worse than the reference group's mean "
        "at which a run counts as a failure",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_yaml(path: Path) -> Any:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_run(path: Path) -> dict[str, Any]:
    metrics = json.loads((path / "metrics.json").read_text())
    history = json.loads((path / "history.json").read_text())
    config = _load_yaml(path / "resolved_config.yaml")
    best_epoch = next(
        int(line.split(":")[1])
        for line in (path / "MODEL_CARD.md").read_text().splitlines()
        if line.startswith("- Best validation epoch")
    )
    test = metrics["test"]
    summary = metrics["pid_conditional_closure"]["bin_summary"]
    weights = np.array([row["n"] for row in summary], dtype=np.float64)
    tv = np.array([row["total_variation_distance"] for row in summary])
    moment = np.mean(
        [
            abs(row["sampled_mean"] - row["observed_mean"]) / row["observed_std"]
            + abs(row["std_ratio"] - 1.0)
            for row in metrics["closure"]
        ]
    )
    return {
        "run": path.name,
        "seed": int(config["project"]["seed"]),
        "epochs_run": len(history),
        "epoch_budget": int(config["training"]["epochs"]),
        "best_epoch": best_epoch,
        "pid_loss_weight": float(config["training"]["pid_loss_weight"]),
        "backbone_lr_multiplier": float(
            config["training"].get("backbone_lr_multiplier", 1.0)
        ),
        "test_residual_nll": test["residual_nll"],
        "test_pid_cross_entropy": test["pid_cross_entropy"],
        "test_joint_nll": test["residual_nll"] + test["pid_cross_entropy"],
        "test_pid_accuracy": test["pid_accuracy"],
        "pid_weighted_mean_tv": float(np.average(tv, weights=weights)),
        "moment_closure_error": float(moment),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for spec in args.group:
        label, _, paths = spec.partition("=")
        runs = [load_run(Path(p.strip())) for p in paths.split(",")]
        runs.sort(key=lambda row: row["seed"])
        for row in runs:
            row["reached_better_solution"] = (
                row["test_pid_accuracy"] >= args.accuracy_threshold
            )
        groups.append((label, runs))

    reference_mean = float(
        np.mean([row["test_joint_nll"] for row in groups[0][1]])
    )
    for _label, runs in groups:
        for row in runs:
            row["failed"] = row["test_joint_nll"] > reference_mean + args.failure_margin

    rows = [{"group": label, **row} for label, runs in groups for row in runs]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_dir / "pid_strategy_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for label, runs in groups:
        entry: dict[str, Any] = {"group": label, "n": len(runs)}
        for key in (
            "test_joint_nll",
            "test_pid_accuracy",
            "pid_weighted_mean_tv",
            "moment_closure_error",
        ):
            values = np.array([row[key] for row in runs], dtype=float)
            entry[f"{key}_mean"] = float(values.mean())
            entry[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        entry["reached_better_solution"] = sum(r["reached_better_solution"] for r in runs)
        entry["failed"] = sum(r["failed"] for r in runs)
        summary_rows.append(entry)
    for entry in summary_rows:
        entry["reference_joint_nll_mean"] = reference_mean
        entry["failure_margin"] = args.failure_margin
    with (output_dir / "pid_strategy_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    panels = (
        ("test_joint_nll", r"held-out joint NLL $J$", True),
        ("test_pid_accuracy", "held-out PID top-1 accuracy", False),
        ("pid_weighted_mean_tv", "PID closure: weighted mean TV", True),
    )
    figure, axes_grid = plt.subplots(1, 3, figsize=(16.5, 5.2))
    labels = [label for label, _ in groups]
    for axes, (key, title, lower_better) in zip(axes_grid, panels):
        for index, (label, runs) in enumerate(groups):
            values = np.array([row[key] for row in runs], dtype=float)
            jitter = np.linspace(-0.14, 0.14, len(values))
            for offset, row, value in zip(jitter, runs, values):
                if row["failed"]:
                    marker, colour, size = "X", "#c53030", 120
                elif row["reached_better_solution"]:
                    marker, colour, size = "*", "#2f855a", 230
                else:
                    marker, colour, size = "o", PALETTE[index % len(PALETTE)], 80
                axes.scatter(
                    index + offset, value, marker=marker, s=size, color=colour,
                    zorder=4, edgecolors="white", linewidths=0.6,
                )
            axes.hlines(values.mean(), index - 0.28, index + 0.28, color="#1a202c", lw=2, zorder=5)
        axes.set_xticks(range(len(labels)), labels, rotation=18, ha="right", fontsize=8)
        axes.set_ylabel(title)
        axes.set_title(f"{title}\n({'lower' if lower_better else 'higher'} is better)", fontsize=10)
        axes.grid(alpha=0.25, axis="y")

    handles = [
        plt.Line2D([], [], marker="*", ls="", ms=15, color="#2f855a", label="reached the better solution"),
        plt.Line2D([], [], marker="o", ls="", ms=9, color="#718096", label="ordinary solution"),
        plt.Line2D([], [], marker="X", ls="", ms=11, color="#c53030",
                   label=f"failed (>{args.failure_margin:g} nats worse than reference)"),
        plt.Line2D([], [], color="#1a202c", lw=2, label="group mean"),
    ]
    figure.legend(handles=handles, loc="lower center", ncols=4, frameon=False, fontsize=9)
    figure.suptitle(
        "Attempts to reach the large-PID-weight solution deliberately: one point per seed",
        y=1.0,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(output_dir / "pid_strategy_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
