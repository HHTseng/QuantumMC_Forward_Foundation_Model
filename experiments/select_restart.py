#!/usr/bin/env python3
"""Multi-restart selection for the large-PID-weight solution.

Sections 13.4.1 and 13.4.2 showed the better solution cannot be reached by
stabilizing training: warm-up, fine-tuning and a smaller trunk learning rate each
remove it. What remains is to stop pretending it is a recipe and treat it as what
it is -- a search. Train several independently initialized runs at the large
weight and keep the one that lands well.

Two properties make this an honest procedure rather than a way of fooling
ourselves:

1. **All restarts share one data partition.** ``data.split_seed`` pins the
   partition while ``project.seed`` varies the initialization and batch order,
   so the runs are mutually comparable and the winner can be reported on a test
   split that none of them trained on.

2. **Selection uses validation only.** Picking the restart with the best *test*
   score would report the maximum of several noisy test evaluations, which is
   biased upward and is not reproducible in use. The selection metric here is
   computed from each run's own validation history; the test numbers are read
   out afterwards and never influence the choice.

The report includes what the procedure costs: how many restarts landed in the
better basin, and therefore how many training runs are needed per usable model.
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

SELECTION_METRICS = {
    "validation_joint_nll": "validation joint NLL",
    "validation_pid_accuracy": "validation PID top-1 accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart", action="append", required=True, metavar="DIR")
    parser.add_argument(
        "--reference",
        required=True,
        metavar="DIR",
        help="Released-recipe run on the same partition, for comparison",
    )
    parser.add_argument(
        "--selection-metric",
        default="validation_joint_nll",
        choices=sorted(SELECTION_METRICS),
    )
    parser.add_argument("--accuracy-threshold", type=float, default=0.70)
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
    audit = json.loads((path / "data_audit.json").read_text())
    best_epoch = next(
        int(line.split(":")[1])
        for line in (path / "MODEL_CARD.md").read_text().splitlines()
        if line.startswith("- Best validation epoch")
    )
    # The selected epoch's validation numbers: what the procedure is allowed to
    # see. Everything named test_* below is read out only after selection.
    chosen = history[best_epoch - 1]["validation"]
    test = metrics["test"]
    summary = metrics["pid_conditional_closure"]["bin_summary"]
    weights = np.array([row["n"] for row in summary], dtype=np.float64)
    tv = np.array([row["total_variation_distance"] for row in summary])
    return {
        "run": path.name,
        "training_seed": int(config["project"]["seed"]),
        "split_seed": int(audit.get("split_seed", config["project"]["seed"])),
        "pid_loss_weight": float(config["training"]["pid_loss_weight"]),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "validation_joint_nll": chosen["residual_nll"] + chosen["pid_cross_entropy"],
        "validation_pid_accuracy": chosen["pid_accuracy"],
        "test_joint_nll": test["residual_nll"] + test["pid_cross_entropy"],
        "test_residual_nll": test["residual_nll"],
        "test_pid_cross_entropy": test["pid_cross_entropy"],
        "test_pid_accuracy": test["pid_accuracy"],
        "test_pid_weighted_mean_tv": float(np.average(tv, weights=weights)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    restarts = [load_run(Path(p)) for p in args.restart]
    reference = load_run(Path(args.reference))

    split_seeds = {row["split_seed"] for row in restarts} | {reference["split_seed"]}
    if len(split_seeds) != 1:
        raise SystemExit(
            f"restarts span {len(split_seeds)} different partitions ({sorted(split_seeds)}); "
            "a restart study must pin data.split_seed so every run shares one split"
        )
    restarts.sort(key=lambda row: row["training_seed"])

    metric = args.selection_metric
    higher_is_better = metric.endswith("accuracy")
    chosen = (max if higher_is_better else min)(restarts, key=lambda row: row[metric])
    for row in restarts:
        row["selected"] = row["run"] == chosen["run"]
        row["reached_better_solution"] = (
            row["test_pid_accuracy"] >= args.accuracy_threshold
        )

    landed = sum(row["reached_better_solution"] for row in restarts)
    rate = landed / len(restarts)
    test_values = np.array([row["test_pid_accuracy"] for row in restarts])
    summary = {
        "n_restarts": len(restarts),
        "split_seed": sorted(split_seeds)[0],
        "selection_metric": metric,
        "selection_used_test_split": False,
        "landed_in_better_basin": landed,
        "landing_rate": rate,
        "expected_runs_per_usable_model": (1.0 / rate) if rate else None,
        "selected_run": chosen["run"],
        "selected_training_seed": chosen["training_seed"],
        "selected_validation_metric": chosen[metric],
        "selected_test_joint_nll": chosen["test_joint_nll"],
        "selected_test_pid_accuracy": chosen["test_pid_accuracy"],
        "selected_test_pid_weighted_mean_tv": chosen["test_pid_weighted_mean_tv"],
        "selection_picked_a_better_basin_run": bool(chosen["reached_better_solution"]),
        "best_possible_test_pid_accuracy": float(test_values.max()),
        "selection_regret_vs_test_oracle": float(
            test_values.max() - chosen["test_pid_accuracy"]
        ),
        "reference_run": reference["run"],
        "reference_test_joint_nll": reference["test_joint_nll"],
        "reference_test_pid_accuracy": reference["test_pid_accuracy"],
        "reference_test_pid_weighted_mean_tv": reference["test_pid_weighted_mean_tv"],
    }

    fields: list[str] = []
    for row in restarts:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_dir / "restart_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(restarts)
    with (output_dir / "restart_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    figure, axes_grid = plt.subplots(1, 2, figsize=(13.0, 5.0))
    validation = np.array([row[metric] for row in restarts])
    accuracy = np.array([row["test_pid_accuracy"] for row in restarts])
    joint = np.array([row["test_joint_nll"] for row in restarts])
    selected_index = [row["run"] for row in restarts].index(chosen["run"])

    axes = axes_grid[0]
    axes.scatter(validation, accuracy, s=90, color="#718096", label="restart")
    axes.scatter(
        validation[selected_index], accuracy[selected_index], s=320, marker="*",
        color="#c53030", zorder=5, label="selected on validation",
    )
    axes.axhline(
        reference["test_pid_accuracy"], color="#2b6cb0", lw=1.8,
        label="released recipe, same split",
    )
    axes.set_xlabel(f"{SELECTION_METRICS[metric]} (selection is made on this axis)")
    axes.set_ylabel("held-out PID top-1 accuracy")
    axes.set_title("Validation selection versus held-out outcome", fontsize=10)
    axes.grid(alpha=0.25)
    axes.legend(frameon=False, fontsize=8)

    axes = axes_grid[1]
    order = np.arange(len(restarts))
    colours = [
        "#2f855a" if row["reached_better_solution"] else "#718096" for row in restarts
    ]
    # Stems anchored on the reference rather than bars anchored on zero: the
    # quantity is a negative log likelihood, so zero is not a meaningful floor
    # and bars from it would waste the axis and imply a false baseline.
    axes.axhline(
        reference["test_joint_nll"], color="#2b6cb0", lw=1.8,
        label="released recipe, same split",
    )
    axes.vlines(order, reference["test_joint_nll"], joint, colors=colours, lw=3, alpha=0.75)
    axes.scatter(order, joint, s=90, color=colours, zorder=4)
    axes.plot(selected_index, joint[selected_index], "*", ms=22, color="#c53030",
              zorder=5, label="selected on validation")
    axes.set_xticks(order, [str(row["training_seed"]) for row in restarts],
                    rotation=45, fontsize=8)
    axes.set_xlabel("training seed (partition held fixed)")
    axes.set_ylabel("held-out joint NLL $J$")
    axes.set_title(
        f"{landed} of {len(restarts)} restarts landed in the better basin", fontsize=10
    )
    axes.grid(alpha=0.25, axis="y")
    axes.legend(frameon=False, fontsize=8)

    figure.suptitle(
        "Multi-restart at a fixed data partition: selection on validation, "
        "reporting on test",
        y=1.0,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "restart_selection.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
