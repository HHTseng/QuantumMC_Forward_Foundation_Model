"""Calibration and phase-space closure for trigger-electron efficiency."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import ElectronEfficiencySplit, OUTCOME_LABELS
from .model import ElectronEfficiencyNet
from .training import make_loader, run_epoch


def _binary_log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(probability.astype(np.float64), 1e-12, 1.0 - 1e-12)
    y = target.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    """Tie-aware Mann-Whitney estimate of ROC AUC without sklearn."""
    y = target.astype(bool)
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = ranks[y].sum()
    return float(
        (positive_rank_sum - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    y = target.astype(np.int64)
    positive_count = int(y.sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
    return float(precision[sorted_y == 1].sum() / positive_count)


def calibration_rows(
    target: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 15,
) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probability >= low) & (
            probability <= high if index == n_bins - 1 else probability < high
        )
        if not mask.any():
            continue
        rows.append(
            {
                "bin_index": index,
                "probability_low": float(low),
                "probability_high": float(high),
                "n": int(mask.sum()),
                "mean_predicted_probability": float(probability[mask].mean()),
                "observed_trigger_rate": float(target[mask].mean()),
                "absolute_calibration_gap": float(
                    abs(probability[mask].mean() - target[mask].mean())
                ),
            }
        )
    return rows


def expected_calibration_error(rows: list[dict[str, Any]]) -> float:
    total = sum(int(row["n"]) for row in rows)
    return float(
        sum(int(row["n"]) * float(row["absolute_calibration_gap"]) for row in rows)
        / total
    )


def efficiency_closure_rows(
    values: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    edges: np.ndarray,
    variable: str,
    min_bin_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (values >= low) & (values <= high if index == len(edges) - 2 else values < high)
        n = int(mask.sum())
        if n < min_bin_count:
            continue
        observed = float(targets[mask].mean())
        predicted = float(probabilities[mask].mean())
        observed_se = float(np.sqrt(observed * (1.0 - observed) / n))
        predicted_se = float(
            probabilities[mask].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        )
        rows.append(
            {
                "variable": variable,
                "bin_index": index,
                "low": float(low),
                "high": float(high),
                "upper_edge_inclusive": index == len(edges) - 2,
                "n": n,
                "observed_efficiency": observed,
                "fm_mean_probability": predicted,
                "signed_difference_fm_minus_data": predicted - observed,
                "absolute_difference": abs(predicted - observed),
                "observed_standard_error": observed_se,
                "fm_mean_standard_error": predicted_se,
            }
        )
    return rows


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_efficiency(rows: list[dict[str, Any]], path: Path, xlabel: str) -> None:
    centers = np.asarray([(row["low"] + row["high"]) / 2 for row in rows])
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.errorbar(
        centers, [row["observed_efficiency"] for row in rows],
        yerr=[row["observed_standard_error"] for row in rows],
        marker="o", capsize=2, label="full simulation"
    )
    axis.errorbar(
        centers, [row["fm_mean_probability"] for row in rows],
        yerr=[row["fm_mean_standard_error"] for row in rows],
        marker="s", capsize=2, label="efficiency model"
    )
    axis.set(xlabel=xlabel, ylabel=r"trigger efficiency $P(T=1\mid x_e)$", ylim=(-0.03, 1.03))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_calibration(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 5.2))
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.4", label="ideal")
    axis.plot(
        [row["mean_predicted_probability"] for row in rows],
        [row["observed_trigger_rate"] for row in rows],
        marker="o", label="held-out test"
    )
    axis.set(
        xlabel="mean predicted trigger probability",
        ylabel="observed trigger rate",
        xlim=(0, 1), ylim=(0, 1), title="Trigger-probability calibration"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_history(history: list[dict[str, Any]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "validation"):
        axes[0].plot(epochs, [row[split]["total_loss"] for row in history], label=split)
        axes[1].plot(epochs, [row[split]["trigger_bce"] for row in history], label=split)
    axes[0].set(xlabel="epoch", ylabel="joint loss", title="Efficiency training")
    axes[1].set(xlabel="epoch", ylabel="binary cross entropy", title="Trigger head")
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


@torch.no_grad()
def _predict(
    model: ElectronEfficiencyNet,
    split: ElectronEfficiencySplit,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(split, batch_size, False, seed, 0)
    trigger: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    model.eval()
    for continuous, _, _ in loader:
        output = model(continuous.to(device))
        trigger.append(torch.sigmoid(output.trigger_logit).cpu().numpy())
        outcomes.append(torch.softmax(output.outcome_logits, dim=-1).cpu().numpy())
    return np.concatenate(trigger), np.concatenate(outcomes)


def evaluate_and_write(
    model: ElectronEfficiencyNet,
    splits: dict[str, ElectronEfficiencySplit],
    history: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    training = config["training"]
    evaluation = config["evaluation"]
    seed = int(config["project"]["seed"])
    test = splits["test"]
    test_loader = make_loader(test, int(training["batch_size"]), False, seed, 0)
    test_epoch = run_epoch(
        model, test_loader, device,
        float(training["trigger_loss_weight"]),
        float(training["outcome_loss_weight"])
    )
    trigger_probability, outcome_probability = _predict(
        model, test, device, int(training["batch_size"]), seed
    )
    y = test.trigger_target
    calibration = calibration_rows(y, trigger_probability, int(evaluation.get("calibration_bins", 15)))

    variables = {
        "gen_p": (test.raw_gen_p, np.asarray(evaluation["momentum_edges_gev"], dtype=float),
                  r"generated electron momentum $p_{e,\rm gen}$ [GeV]"),
        "gen_theta": (
            np.rad2deg(test.raw_gen_theta), np.asarray(evaluation["theta_edges_deg"], dtype=float),
            r"generated electron polar angle $\theta_{e,\rm gen}$ [deg]"
        ),
        "gen_phi": (
            np.rad2deg(test.raw_gen_phi), np.asarray(evaluation["phi_edges_deg"], dtype=float),
            r"generated electron azimuth $\phi_{e,\rm gen}$ [deg]"
        ),
        "gen_vz": (
            test.raw_gen_vz, np.asarray(evaluation["vz_edges_cm"], dtype=float),
            r"generated vertex $v_{z,e}$ [cm]"
        ),
    }
    closure: dict[str, list[dict[str, Any]]] = {}
    for variable, (values, edges, xlabel) in variables.items():
        rows = efficiency_closure_rows(
            values, y, trigger_probability, edges, variable,
            int(evaluation["min_bin_count"])
        )
        closure[variable] = rows
        _write_rows(rows, run_dir / f"efficiency_vs_{variable}.csv")
        _plot_efficiency(rows, run_dir / f"efficiency_vs_{variable}.png", xlabel)

    outcome_rows: list[dict[str, Any]] = []
    predicted_outcome = outcome_probability.argmax(axis=1)
    for true_index, true_label in enumerate(OUTCOME_LABELS):
        true_mask = test.outcome_target == true_index
        n = int(true_mask.sum())
        if n == 0:
            continue
        for predicted_index, predicted_label in enumerate(OUTCOME_LABELS):
            outcome_rows.append(
                {
                    "observed_outcome": true_label,
                    "predicted_outcome": predicted_label,
                    "observed_rows": n,
                    "argmax_count": int(np.sum(predicted_outcome[true_mask] == predicted_index)),
                    "mean_predicted_probability": float(
                        outcome_probability[true_mask, predicted_index].mean()
                    ),
                }
            )
    _write_rows(outcome_rows, run_dir / "outcome_response_matrix.csv")
    _write_rows(calibration, run_dir / "calibration_curve.csv")
    _plot_calibration(calibration, run_dir / "calibration_curve.png")
    _plot_history(history, run_dir / "training_history.png")

    clipped_outcome = np.clip(
        outcome_probability[np.arange(len(test)), test.outcome_target], 1e-12, 1.0
    )
    one_hot = np.eye(len(OUTCOME_LABELS), dtype=np.float64)[test.outcome_target]
    all_closure_rows = [row for rows in closure.values() for row in rows]
    metrics = {
        "test": test_epoch.as_dict(),
        "trigger": {
            "log_loss": _binary_log_loss(y, trigger_probability),
            "brier_score": float(np.mean((trigger_probability - y) ** 2)),
            "expected_calibration_error": expected_calibration_error(calibration),
            "roc_auc": _binary_auc(y, trigger_probability),
            "average_precision": _average_precision(y, trigger_probability),
            "observed_rate": float(y.mean()),
            "mean_predicted_probability": float(trigger_probability.mean()),
            "maximum_absolute_binned_difference": float(
                max(row["absolute_difference"] for row in all_closure_rows)
            ),
        },
        "outcome": {
            "cross_entropy": float(-np.log(clipped_outcome).mean()),
            "multiclass_brier_score": float(np.mean(np.sum((outcome_probability - one_hot) ** 2, axis=1))),
            "argmax_accuracy": float(np.mean(predicted_outcome == test.outcome_target)),
            "note": "Current teacher has only unreconstructed and FD, exactly aligned with T=0 and T=1.",
        },
        "efficiency_closure": closure,
        "calibration": calibration,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    return metrics
