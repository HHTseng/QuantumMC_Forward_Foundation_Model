#!/usr/bin/env python3
"""Summarize the Optuna capacity study and emit the selected configuration.

Two-stage selection
-------------------
Stage 1 (search): trials are ranked by the validation joint negative log
likelihood ``J = residual_nll + pid_cross_entropy``.  ``J`` is a proper log
density of the factorized model q(Delta|x) q(s_rec|x), so it is comparable
across trials that trained with different ``pid_loss_weight`` values, and it is
cheap and smooth enough to prune on.

Stage 2 (final pick): among the ``--top-k`` trials by ``J`` -- all of which are
statistically close -- the checkpoint is chosen by the physics closure
composite

    C = pid_closure_tv / median(pid_closure_tv over top-k)
      + moment_closure_error / median(moment_closure_error over top-k),

which balances the reconstructed-PID response closure against first- and
second-moment closure of the residual variables.  Normalizing by the top-k
median makes the two terms dimensionless and equally weighted without hiding an
arbitrary absolute scale factor.

Every figure and table is written under ``--output-dir``; the selected point is
written as a runnable YAML configuration.
"""
from __future__ import annotations

import argparse
import sys
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import yaml

# Allow execution as `python experiments/<script>.py` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from forwardfm_step1.config import load_config


SEARCHED_MODEL_KEYS = (
    "hidden_width",
    "hidden_layers",
    "pid_embedding_dim",
    "mixture_components",
    "dropout",
)
SEARCHED_TRAINING_KEYS = (
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "pid_loss_weight",
    "lr_schedule",
)
PRETTY = {
    "hidden_width": "hidden width",
    "hidden_layers": "hidden layers",
    "pid_embedding_dim": "species embedding dim",
    "mixture_components": "mixture components $K$",
    "dropout": "dropout",
    "epochs": "epoch budget",
    "batch_size": "batch size",
    "learning_rate": "learning rate",
    "weight_decay": "weight decay",
    "pid_loss_weight": r"$\lambda_{\mathrm{PID}}$",
    "lr_schedule": "LR schedule",
}
LOG_PARAMS = {"learning_rate", "weight_decay", "pid_loss_weight"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--study-name", default="forwardfm-step1-capacity")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--best-config", required=True, help="YAML path to write")
    parser.add_argument("--best-run-dir", default="runs/optuna_best")
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def trial_table(study: optuna.Study) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": trial.number,
            "state": trial.state.name,
            "objective_joint_nll": trial.value,
        }
        row.update({key: trial.params.get(key) for key in PRETTY})
        for key in (
            "trainable_parameters",
            "best_epoch",
            "epochs_run",
            "train_seconds",
            "validation_residual_nll",
            "validation_pid_cross_entropy",
            "validation_pid_accuracy",
            "moment_closure_error",
            "pid_closure_tv",
            "pid_closure_tv_max",
            "pid_marginal_discrepancy",
            "worker",
        ):
            row[key] = trial.user_attrs.get(key)
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def closure_composite(rows: list[dict[str, Any]]) -> tuple[list[float], float, float]:
    """Median-normalized sum of the PID and moment closure errors."""
    tv = np.array([row["pid_closure_tv"] for row in rows], dtype=np.float64)
    moment = np.array([row["moment_closure_error"] for row in rows], dtype=np.float64)
    tv_scale = float(np.median(tv))
    moment_scale = float(np.median(moment))
    composite = (tv / tv_scale + moment / moment_scale).tolist()
    return composite, tv_scale, moment_scale


def plot_optimization_history(rows: list[dict[str, Any]], path: Path) -> None:
    complete = [row for row in rows if row["state"] == "COMPLETE"]
    pruned = [row for row in rows if row["state"] == "PRUNED"]
    figure, axes = plt.subplots(figsize=(8.0, 4.6))
    if complete:
        numbers = [row["number"] for row in complete]
        values = [row["objective_joint_nll"] for row in complete]
        axes.scatter(numbers, values, s=26, color="#2b6cb0", label="complete trial")
        running = np.minimum.accumulate(values)
        axes.step(numbers, running, where="post", color="#c53030", lw=2, label="best so far")
    if pruned:
        top = max(row["objective_joint_nll"] for row in complete) if complete else 0.0
        axes.scatter(
            [row["number"] for row in pruned],
            [top] * len(pruned),
            s=18,
            marker="x",
            color="#a0aec0",
            label=f"pruned ({len(pruned)})",
        )
    axes.set_xlabel("trial number")
    axes.set_ylabel(r"validation $J=\mathrm{NLL}_\Delta+\mathrm{CE}_{\mathrm{PID}}$")
    axes.set_title("Optuna search history (lower is better)")
    axes.grid(alpha=0.3)
    axes.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_importances(study: optuna.Study, path: Path) -> tuple[dict[str, float], str]:
    """Rank hyper-parameters by how much of the variation in J they explain.

    fANOVA is used explicitly rather than the library default so that the
    reported quantity is a documented variance decomposition and stays the same
    if the Optuna default evaluator changes.
    """
    evaluator = optuna.importance.FanovaImportanceEvaluator(
        seed=0, n_trees=64, max_depth=64
    )
    try:
        importances = optuna.importance.get_param_importances(
            study, evaluator=evaluator
        )
    except (ValueError, RuntimeError) as error:  # too few completed trials
        print(f"skipping importances: {error}")
        return {}, type(evaluator).__name__
    names = list(importances)[::-1]
    values = [importances[name] for name in names]
    figure, axes = plt.subplots(figsize=(7.4, 0.42 * len(names) + 1.6))
    axes.barh([PRETTY.get(name, name) for name in names], values, color="#2c7a7b")
    for index, value in enumerate(values):
        axes.text(value, index, f" {value:.3f}", va="center", fontsize=9)
    axes.set_xlabel("fANOVA importance for validation $J$")
    axes.set_title("Which hyper-parameters actually move the objective")
    axes.set_xlim(0, max(values) * 1.22)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return dict(importances), type(evaluator).__name__


def plot_slices(rows: list[dict[str, Any]], path: Path) -> None:
    complete = [row for row in rows if row["state"] == "COMPLETE"]
    names = [name for name in PRETTY if any(row[name] is not None for row in complete)]
    columns = 3
    lines = (len(names) + columns - 1) // columns
    figure, axes_grid = plt.subplots(lines, columns, figsize=(4.4 * columns, 3.3 * lines))
    axes_list = np.atleast_1d(axes_grid).ravel()
    values = [row["objective_joint_nll"] for row in complete]
    for axes, name in zip(axes_list, names):
        raw = [row[name] for row in complete]
        if isinstance(raw[0], str):
            categories = sorted(set(raw))
            positions = [categories.index(item) for item in raw]
            axes.set_xticks(range(len(categories)), categories)
            x = positions
        else:
            x = raw
            if name in LOG_PARAMS:
                axes.set_xscale("log")
        scatter = axes.scatter(x, values, c=values, cmap="viridis_r", s=26)
        axes.set_xlabel(PRETTY[name])
        axes.set_ylabel("$J$")
        axes.grid(alpha=0.25)
    for axes in axes_list[len(names) :]:
        axes.axis("off")
    figure.colorbar(scatter, ax=axes_list.tolist(), label="$J$", shrink=0.6)
    figure.suptitle("Objective versus each searched hyper-parameter", y=1.0)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_capacity_and_tradeoff(
    rows: list[dict[str, Any]], top_numbers: set[int], selected: int, path: Path
) -> None:
    complete = [row for row in rows if row["state"] == "COMPLETE"]
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))

    parameters = np.array([row["trainable_parameters"] for row in complete], dtype=float)
    objective = np.array([row["objective_joint_nll"] for row in complete], dtype=float)
    axes[0].scatter(parameters, objective, s=28, color="#2b6cb0")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("trainable parameters")
    axes[0].set_ylabel("$J$")
    axes[0].set_title("Capacity versus objective")
    axes[0].grid(alpha=0.3)

    weight = np.array([row["pid_loss_weight"] for row in complete], dtype=float)
    pid_ce = np.array(
        [row["validation_pid_cross_entropy"] for row in complete], dtype=float
    )
    residual = np.array([row["validation_residual_nll"] for row in complete], dtype=float)
    axes[1].scatter(weight, pid_ce, s=28, color="#c05621", label="PID cross entropy")
    twin = axes[1].twinx()
    twin.scatter(weight, residual, s=28, marker="^", color="#2f855a", label="residual NLL")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\lambda_{\mathrm{PID}}$")
    axes[1].set_ylabel("validation PID cross entropy", color="#c05621")
    twin.set_ylabel("validation residual NLL", color="#2f855a")
    axes[1].set_title(r"The $\lambda_{\mathrm{PID}}$ trade-off")
    axes[1].grid(alpha=0.3)

    tv = np.array([row["pid_closure_tv"] for row in complete], dtype=float)
    moment = np.array([row["moment_closure_error"] for row in complete], dtype=float)
    numbers = [row["number"] for row in complete]
    axes[2].scatter(tv, moment, s=28, color="#a0aec0", label="all complete")
    mask = np.array([number in top_numbers for number in numbers])
    axes[2].scatter(tv[mask], moment[mask], s=46, color="#2b6cb0", label="top-k by $J$")
    chosen = np.array([number == selected for number in numbers])
    axes[2].scatter(
        tv[chosen], moment[chosen], s=190, marker="*", color="#c53030", label="selected"
    )
    axes[2].set_xlabel("PID closure: particle-weighted mean TV")
    axes[2].set_ylabel("moment closure error")
    axes[2].set_title("Physics closure of the candidate trials")
    axes[2].grid(alpha=0.3)
    axes[2].legend(frameon=False, fontsize=9)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_learning_curves(study: optuna.Study, top_numbers: list[int], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(top_numbers)))
    by_number = {trial.number: trial for trial in study.trials}
    for color, number in zip(colors, top_numbers):
        raw = by_number[number].user_attrs.get("history")
        if not raw:
            continue
        history = json.loads(raw)
        epochs = [entry["epoch"] for entry in history]
        axes[0].plot(
            epochs,
            [entry["validation"]["joint_nll"] for entry in history],
            color=color,
            label=f"trial {number}",
        )
        axes[1].plot(
            epochs,
            [entry["validation"]["pid_cross_entropy"] for entry in history],
            color=color,
        )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation $J$")
    axes[0].set_title("Learning curves of the top trials")
    axes[0].legend(frameon=False, fontsize=8, ncols=2)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation PID cross entropy")
    axes[1].set_title("PID term of the same trials")
    for panel in axes:
        panel.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_best_config(
    base_config: dict[str, Any], params: dict[str, Any], run_dir: str
) -> dict[str, Any]:
    config = {
        key: value for key, value in base_config.items() if not key.startswith("_")
    }
    config = json.loads(json.dumps(config))
    config["project"]["name"] = "clas12-forward-fm-step1-optuna-best"
    for key in SEARCHED_MODEL_KEYS:
        config["model"][key] = params[key]
    for key in SEARCHED_TRAINING_KEYS:
        config["training"][key] = params[key]
    # The final run trains a releasable checkpoint, so it selects the epoch by
    # the same lambda-independent criterion the search used.
    config["training"]["selection_metric"] = "joint_nll"
    config["output"]["run_dir"] = run_dir
    return config


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    rows = trial_table(study)
    write_csv(rows, output_dir / "optuna_trials.csv")

    complete = [row for row in rows if row["state"] == "COMPLETE"]
    if not complete:
        raise SystemExit("No completed trials")
    ranked = sorted(complete, key=lambda row: row["objective_joint_nll"])
    top = ranked[: args.top_k]
    composite, tv_scale, moment_scale = closure_composite(top)
    for row, value in zip(top, composite):
        row["closure_composite"] = value
    selected = min(zip(top, composite), key=lambda pair: pair[1])[0]
    write_csv(top, output_dir / "optuna_top_trials.csv")

    plot_optimization_history(rows, output_dir / "optuna_history.png")
    importances, importance_evaluator = plot_importances(
        study, output_dir / "optuna_importances.png"
    )
    plot_slices(rows, output_dir / "optuna_slices.png")
    plot_capacity_and_tradeoff(
        rows, {row["number"] for row in top}, selected["number"],
        output_dir / "optuna_capacity_and_tradeoff.png",
    )
    plot_learning_curves(
        study, [row["number"] for row in top], output_dir / "optuna_top_learning_curves.png"
    )

    base_config = load_config(args.base_config)
    best_trial = next(trial for trial in study.trials if trial.number == selected["number"])
    best_config = build_best_config(base_config, best_trial.params, args.best_run_dir)
    best_path = Path(args.best_config)
    with best_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_config, handle, sort_keys=False)

    summary = {
        "study_name": args.study_name,
        "n_trials": len(rows),
        "n_complete": len(complete),
        "n_pruned": sum(1 for row in rows if row["state"] == "PRUNED"),
        "n_failed": sum(1 for row in rows if row["state"] == "FAIL"),
        "objective": "validation residual_nll + pid_cross_entropy",
        "best_by_objective": ranked[0]["number"],
        "best_objective_value": ranked[0]["objective_joint_nll"],
        "top_k": args.top_k,
        "closure_composite_tv_scale": tv_scale,
        "closure_composite_moment_scale": moment_scale,
        "selected_trial": selected["number"],
        "selected_objective": selected["objective_joint_nll"],
        "selected_closure_composite": selected["closure_composite"],
        "selected_params": best_trial.params,
        "selected_user_attrs": {
            key: value
            for key, value in best_trial.user_attrs.items()
            if key != "history"
        },
        "param_importances": importances,
        "param_importance_evaluator": importance_evaluator,
        "written_config": str(best_path),
        "total_search_seconds": sum(
            row["train_seconds"] or 0.0 for row in rows if row["train_seconds"]
        ),
    }
    with (output_dir / "optuna_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2)[:2400])


if __name__ == "__main__":
    main()
