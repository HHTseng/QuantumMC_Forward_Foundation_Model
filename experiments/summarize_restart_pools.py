#!/usr/bin/env python3
"""Aggregate several multi-restart pools run at different data partitions.

One pool shows the procedure worked once. Several pools, each at its own pinned
partition, measure the two things that decide whether it is usable:

* the **landing rate** -- how often an independently initialized run at the large
  PID weight reaches the better solution, and therefore how many training runs
  are needed per usable model;
* whether **validation selection is reliable** -- whether picking the best
  restart on validation actually picks a better-basin run, and how much held-out
  accuracy that choice gives up against an oracle that cheated and looked at the
  test split.

Each pool contributes its own released-recipe reference from the same partition,
so every comparison is within-partition.
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
    parser.add_argument(
        "--pool",
        action="append",
        required=True,
        metavar="DIR",
        help="Directory holding one pool's restart_selection.json and restart_runs.csv",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, which stays inside [0,1] for small samples."""
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = (z / denominator) * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pools = []
    all_runs: list[dict[str, Any]] = []
    for raw in args.pool:
        path = Path(raw)
        summary = json.loads((path / "restart_selection.json").read_text())
        with (path / "restart_runs.csv").open("r", encoding="utf-8") as handle:
            runs = list(csv.DictReader(handle))
        for row in runs:
            all_runs.append({"pool": str(summary["split_seed"]), **row})
        pools.append(summary)

    total_restarts = sum(p["n_restarts"] for p in pools)
    total_landed = sum(p["landed_in_better_basin"] for p in pools)
    rate = total_landed / total_restarts
    low, high = wilson_interval(total_landed, total_restarts)
    picked_better = sum(bool(p["selection_picked_a_better_basin_run"]) for p in pools)
    regrets = [p["selection_regret_vs_test_oracle"] for p in pools]

    aggregate = {
        "n_pools": len(pools),
        "total_restarts": total_restarts,
        "total_landed_in_better_basin": total_landed,
        "landing_rate": rate,
        "landing_rate_wilson_95": [low, high],
        "expected_runs_per_usable_model": (1.0 / rate) if rate else None,
        "probability_at_least_one_success": {
            str(k): 1.0 - (1.0 - rate) ** k for k in (4, 6, 8, 12)
        },
        "pools_where_validation_selection_picked_a_better_basin_run": picked_better,
        "max_selection_regret_vs_test_oracle": max(regrets),
        "per_pool": [
            {
                "split_seed": p["split_seed"],
                "n_restarts": p["n_restarts"],
                "landed": p["landed_in_better_basin"],
                "selected_run": p["selected_run"],
                "selected_test_pid_accuracy": p["selected_test_pid_accuracy"],
                "selected_test_joint_nll": p["selected_test_joint_nll"],
                "selected_test_pid_weighted_mean_tv": p["selected_test_pid_weighted_mean_tv"],
                "reference_test_pid_accuracy": p["reference_test_pid_accuracy"],
                "reference_test_joint_nll": p["reference_test_joint_nll"],
                "reference_test_pid_weighted_mean_tv": p["reference_test_pid_weighted_mean_tv"],
                "accuracy_gain": p["selected_test_pid_accuracy"] - p["reference_test_pid_accuracy"],
                "joint_nll_gain": p["selected_test_joint_nll"] - p["reference_test_joint_nll"],
                "tv_gain": p["selected_test_pid_weighted_mean_tv"]
                - p["reference_test_pid_weighted_mean_tv"],
                "selection_regret_vs_test_oracle": p["selection_regret_vs_test_oracle"],
            }
            for p in pools
        ],
    }
    with (output_dir / "restart_pools_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    with (output_dir / "restart_pools.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate["per_pool"][0]))
        writer.writeheader()
        writer.writerows(aggregate["per_pool"])

    figure, axes_grid = plt.subplots(1, 2, figsize=(13.5, 5.0))

    axes = axes_grid[0]
    for index, pool in enumerate(pools):
        runs = [r for r in all_runs if r["pool"] == str(pool["split_seed"])]
        accuracy = np.array([float(r["test_pid_accuracy"]) for r in runs])
        offsets = np.linspace(-0.16, 0.16, len(accuracy))
        for offset, row, value in zip(offsets, runs, accuracy):
            better = row["reached_better_solution"] == "True"
            axes.scatter(
                index + offset, value, s=70,
                color="#2f855a" if better else "#a0aec0", zorder=3,
                edgecolors="white", linewidths=0.5,
            )
            if row["selected"] == "True":
                axes.scatter(index + offset, value, s=300, marker="*",
                             color="#c53030", zorder=5)
        axes.hlines(
            pool["reference_test_pid_accuracy"], index - 0.28, index + 0.28,
            color="#2b6cb0", lw=2.4, zorder=4,
        )
    axes.set_xticks(range(len(pools)), [str(p["split_seed"]) for p in pools])
    axes.set_xlabel("data partition (split seed)")
    axes.set_ylabel("held-out PID top-1 accuracy")
    axes.set_title(
        f"{total_landed} of {total_restarts} restarts landed in the better basin", fontsize=10
    )
    axes.grid(alpha=0.25, axis="y")
    axes.legend(
        handles=[
            plt.Line2D([], [], marker="*", ls="", ms=15, color="#c53030",
                       label="selected on validation"),
            plt.Line2D([], [], marker="o", ls="", ms=9, color="#2f855a",
                       label="better basin"),
            plt.Line2D([], [], marker="o", ls="", ms=9, color="#a0aec0",
                       label="ordinary basin"),
            plt.Line2D([], [], color="#2b6cb0", lw=2.4,
                       label="released recipe, same partition"),
        ],
        frameon=False, fontsize=8, loc="center right",
    )

    axes = axes_grid[1]
    labels = [str(p["split_seed"]) for p in pools]
    reference = [p["reference_test_pid_accuracy"] for p in pools]
    selected = [p["selected_test_pid_accuracy"] for p in pools]
    positions = np.arange(len(pools))
    axes.bar(positions - 0.19, reference, width=0.36, color="#2b6cb0", label="released recipe")
    axes.bar(positions + 0.19, selected, width=0.36, color="#2f855a", label="restart-selected")
    for position, low_value, high_value in zip(positions, reference, selected):
        axes.text(position + 0.19, high_value, f"+{100 * (high_value - low_value):.2f} pp",
                  ha="center", va="bottom", fontsize=9)
    axes.set_xticks(positions, labels)
    axes.set_xlabel("data partition (split seed)")
    axes.set_ylabel("held-out PID top-1 accuracy")
    axes.set_ylim(0.66, max(selected) + 0.012)
    axes.set_title("Within-partition gain, selection never saw the test split", fontsize=10)
    axes.grid(alpha=0.25, axis="y")
    axes.legend(frameon=False, fontsize=9, loc="upper left")

    figure.suptitle(
        "Multi-restart replicated at three independent data partitions", y=1.0
    )
    figure.tight_layout()
    figure.savefig(output_dir / "restart_pools.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
