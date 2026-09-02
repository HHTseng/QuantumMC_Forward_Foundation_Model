#!/usr/bin/env python3
"""Show that the large-lambda_PID gain is bimodal, not a shift.

A mean and a standard deviation are the wrong summary for this comparison. The
lambda_PID >= 2 runs do not scatter around a better centre; they land in one of
two places -- a distinctly better solution or the ordinary one -- with a third
possibility that training destabilizes and early stopping returns an undertrained
model. Averaging those hides exactly the thing a reader needs to see before
adopting the setting.

Usage:

    python experiments/plot_pid_weight_stability.py \
        --reference "released lambda=0.397"=runs/optuna_best,runs/seed_tuned_20260823,... \
        --variant "lambda=2"=runs/optuna_best_pidweight,runs/seed_pidweight_20260823,... \
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, metavar="LABEL=DIR,DIR,...")
    parser.add_argument("--variant", required=True, metavar="LABEL=DIR,DIR,...")
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.70,
        help="Test PID top-1 above this counts as having reached the better solution",
    )
    parser.add_argument(
        "--destabilized-fraction",
        type=float,
        default=0.2,
        help="A run counts as destabilized when its best epoch falls inside this "
        "fraction of the budget, i.e. it peaked immediately and never recovered. "
        "Ordinary early stopping late in the schedule is not destabilization.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_group(spec: str) -> tuple[str, list[dict[str, Any]]]:
    label, _, paths = spec.partition("=")
    rows: list[dict[str, Any]] = []
    for raw in paths.split(","):
        path = Path(raw.strip())
        metrics = json.loads((path / "metrics.json").read_text())
        history = json.loads((path / "history.json").read_text())
        config = json.loads(json.dumps(_load_yaml(path / "resolved_config.yaml")))
        best_epoch = next(
            int(line.split(":")[1])
            for line in (path / "MODEL_CARD.md").read_text().splitlines()
            if line.startswith("- Best validation epoch")
        )
        test = metrics["test"]
        summary = metrics["pid_conditional_closure"]["bin_summary"]
        weights = np.array([row["n"] for row in summary], dtype=np.float64)
        tv = np.array([row["total_variation_distance"] for row in summary])
        rows.append(
            {
                "run": path.name,
                "seed": int(config["project"]["seed"]),
                "epochs_run": len(history),
                "best_epoch": best_epoch,
                "epoch_budget": int(config["training"]["epochs"]),
                "pid_loss_weight": float(config["training"]["pid_loss_weight"]),
                "test_joint_nll": test["residual_nll"] + test["pid_cross_entropy"],
                "test_pid_accuracy": test["pid_accuracy"],
                "pid_weighted_mean_tv": float(np.average(tv, weights=weights)),
            }
        )
    rows.sort(key=lambda row: row["seed"])
    return label, rows


def _load_yaml(path: Path) -> Any:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_label, reference = read_group(args.reference)
    variant_label, variant = read_group(args.variant)

    panels = (
        ("test_joint_nll", r"held-out joint NLL $J$", True),
        ("test_pid_accuracy", "held-out PID top-1 accuracy", False),
        ("pid_weighted_mean_tv", "PID closure: weighted mean TV", True),
    )
    figure, axes_grid = plt.subplots(1, 3, figsize=(16.0, 4.8))
    for axes, (key, title, lower_better) in zip(axes_grid, panels):
        values = np.array([row[key] for row in reference], dtype=float)
        centre, spread = values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0
        axes.axhspan(
            centre - spread,
            centre + spread,
            color="#2b6cb0",
            alpha=0.16,
            label=f"{reference_label}: mean $\\pm$ s.d. ({len(values)} seeds)",
        )
        axes.axhline(centre, color="#2b6cb0", lw=1.6)

        seeds = [row["seed"] for row in variant]
        positions = np.arange(len(seeds))
        for index, row in enumerate(variant):
            diverged = row["best_epoch"] <= args.destabilized_fraction * row["epoch_budget"]
            better = row["test_pid_accuracy"] >= args.accuracy_threshold
            colour = "#c53030" if diverged else ("#2f855a" if better else "#a0aec0")
            marker = "X" if diverged else ("*" if better else "o")
            axes.scatter(
                index,
                row[key],
                s=210 if marker == "*" else 110,
                marker=marker,
                color=colour,
                zorder=4,
                edgecolors="white",
                linewidths=0.6,
            )
        axes.set_xticks(positions, [str(seed) for seed in seeds], rotation=45, fontsize=8)
        axes.set_xlabel("seed")
        axes.set_ylabel(title)
        axes.set_title(f"{title}\n({'lower' if lower_better else 'higher'} is better)", fontsize=10)
        axes.grid(alpha=0.25, axis="y")

    handles = [
        plt.Line2D([], [], marker="*", ls="", ms=15, color="#2f855a",
                   label="reached the better solution"),
        plt.Line2D([], [], marker="o", ls="", ms=9, color="#a0aec0",
                   label="ordinary solution"),
        plt.Line2D([], [], marker="X", ls="", ms=11, color="#c53030",
                   label="destabilized, early-stopped undertrained"),
        plt.Line2D([], [], color="#2b6cb0", lw=6, alpha=0.4,
                   label=f"{reference_label}, mean $\\pm$ s.d."),
    ]
    figure.legend(handles=handles, loc="lower center", ncols=4, frameon=False, fontsize=9)
    reached = sum(
        row["test_pid_accuracy"] >= args.accuracy_threshold for row in variant
    )
    unstable = sum(
        row["best_epoch"] <= args.destabilized_fraction * row["epoch_budget"]
        for row in variant
    )
    figure.suptitle(
        f"{variant_label}: one point per independently trained seed. "
        f"{reached} of {len(variant)} reached the better solution, "
        f"{unstable} destabilized.",
        y=1.0,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output_dir / "pid_weight_stability.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    rows = [{"group": reference_label, **row} for row in reference]
    rows += [{"group": variant_label, **row} for row in variant]
    for row in rows:
        row["reached_better_solution"] = row["test_pid_accuracy"] >= args.accuracy_threshold
        row["destabilized"] = (
            row["best_epoch"] <= args.destabilized_fraction * row["epoch_budget"]
        )
    with (output_dir / "pid_weight_stability.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    variant_values = np.array([row["test_joint_nll"] for row in variant])
    reference_values = np.array([row["test_joint_nll"] for row in reference])
    kept = np.array(
        [
            row["best_epoch"] > args.destabilized_fraction * row["epoch_budget"]
            for row in variant
        ]
    )
    print(
        json.dumps(
            {
                "reference": {
                    "label": reference_label,
                    "n": len(reference_values),
                    "joint_nll_mean": float(reference_values.mean()),
                    "joint_nll_std": float(reference_values.std(ddof=1)),
                },
                "variant": {
                    "label": variant_label,
                    "n": len(variant_values),
                    "joint_nll_mean": float(variant_values.mean()),
                    "joint_nll_std": float(variant_values.std(ddof=1)),
                    "reached_better_solution": int(
                        sum(row["test_pid_accuracy"] >= args.accuracy_threshold for row in variant)
                    ),
                    "destabilized": int((~kept).sum()),
                    "joint_nll_mean_excluding_destabilized": float(
                        variant_values[kept].mean()
                    ),
                    "joint_nll_std_excluding_destabilized": float(
                        variant_values[kept].std(ddof=1)
                    ),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
