r"""Probability calibration and phase-space closure for \(P(T=1\mid x_e)\)."""

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

from .data import TriggerSplit
from .efficiency import TriggerEfficiencyNet, make_loader


def binary_log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(target, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def calibration_rows(
    target: np.ndarray,
    probability: np.ndarray,
    n_bins: int,
) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probability >= low) & (
            probability <= high if index == n_bins - 1 else probability < high
        )
        if not mask.any():
            continue
        predicted = float(probability[mask].mean())
        observed = float(target[mask].mean())
        rows.append(
            {
                "bin_index": index,
                "probability_low": float(low),
                "probability_high": float(high),
                "n": int(mask.sum()),
                "mean_predicted_probability": predicted,
                "observed_trigger_rate": observed,
                "absolute_calibration_gap": abs(predicted - observed),
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
    target: np.ndarray,
    probability: np.ndarray,
    edges: np.ndarray,
    variable: str,
    min_bin_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (values >= low) & (
            values <= high if index == len(edges) - 2 else values < high
        )
        n = int(mask.sum())
        if n < min_bin_count:
            continue
        observed = float(target[mask].mean())
        predicted = float(probability[mask].mean())
        rows.append(
            {
                "variable": variable,
                "bin_index": index,
                "low": float(low),
                "high": float(high),
                "n": n,
                "observed_efficiency": observed,
                "fm_mean_probability": predicted,
                "signed_difference_fm_minus_mc": predicted - observed,
                "absolute_difference": abs(predicted - observed),
                "observed_binomial_standard_error": float(
                    np.sqrt(observed * (1.0 - observed) / n)
                ),
                "fm_mean_standard_error": float(
                    probability[mask].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
                ),
            }
        )
    return rows


def efficiency_closure_2d_rows(
    momentum: np.ndarray,
    theta_deg: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    momentum_edges: np.ndarray,
    theta_edges: np.ndarray,
    min_bin_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p_index, (p_low, p_high) in enumerate(
        zip(momentum_edges[:-1], momentum_edges[1:])
    ):
        p_mask = (momentum >= p_low) & (
            momentum <= p_high
            if p_index == len(momentum_edges) - 2
            else momentum < p_high
        )
        for theta_index, (theta_low, theta_high) in enumerate(
            zip(theta_edges[:-1], theta_edges[1:])
        ):
            theta_mask = (theta_deg >= theta_low) & (
                theta_deg <= theta_high
                if theta_index == len(theta_edges) - 2
                else theta_deg < theta_high
            )
            mask = p_mask & theta_mask
            n = int(mask.sum())
            if n < min_bin_count:
                continue
            observed = float(target[mask].mean())
            predicted = float(probability[mask].mean())
            rows.append(
                {
                    "p_bin": p_index,
                    "theta_bin": theta_index,
                    "p_low_gev": float(p_low),
                    "p_high_gev": float(p_high),
                    "theta_low_deg": float(theta_low),
                    "theta_high_deg": float(theta_high),
                    "n": n,
                    "observed_efficiency": observed,
                    "fm_mean_probability": predicted,
                    "signed_difference_fm_minus_mc": predicted - observed,
                    "absolute_difference": abs(predicted - observed),
                    "observed_binomial_standard_error": float(
                        np.sqrt(observed * (1.0 - observed) / n)
                    ),
                }
            )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No populated bins for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _roc_auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = target.astype(bool)
    n_positive = int(positive.sum())
    n_negative = len(target) - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    y = target.astype(np.int64)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    sorted_y = y[np.argsort(-score, kind="mergesort")]
    precision = np.cumsum(sorted_y) / np.arange(1, len(sorted_y) + 1)
    return float(precision[sorted_y == 1].sum() / positives)


@torch.no_grad()
def predict_probabilities(
    model: TriggerEfficiencyNet,
    split: TriggerSplit,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    loader = make_loader(split, batch_size, False, seed)
    values: list[np.ndarray] = []
    model.eval()
    for continuous, _ in loader:
        values.append(torch.sigmoid(model(continuous.to(device))).cpu().numpy())
    return np.concatenate(values)


def _closure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = np.asarray([row["n"] for row in rows], dtype=np.float64)
    gaps = np.asarray([row["absolute_difference"] for row in rows])
    worst = rows[int(np.argmax(gaps))]
    return {
        "populated_bins": len(rows),
        "particle_weighted_mean_absolute_error": float(np.average(gaps, weights=counts)),
        "maximum_absolute_error": float(gaps.max()),
        "worst_bin": worst,
    }


def _plot_efficiency(rows: list[dict[str, Any]], path: Path, xlabel: str) -> None:
    centers = np.asarray([(row["low"] + row["high"]) / 2 for row in rows])
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.errorbar(
        centers,
        [row["observed_efficiency"] for row in rows],
        yerr=[row["observed_binomial_standard_error"] for row in rows],
        marker="o",
        capsize=2,
        label="full simulation",
    )
    axis.errorbar(
        centers,
        [row["fm_mean_probability"] for row in rows],
        yerr=[row["fm_mean_standard_error"] for row in rows],
        marker="s",
        capsize=2,
        label="Forward FM",
    )
    axis.set(xlabel=xlabel, ylabel=r"$P(T=1\mid x_e)$", ylim=(-0.03, 1.03))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_calibration(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.4, 5.2))
    axis.plot([0, 1], [0, 1], "--", color="0.4", label="exact closure")
    axis.plot(
        [row["mean_predicted_probability"] for row in rows],
        [row["observed_trigger_rate"] for row in rows],
        "o-",
        label="held-out test",
    )
    axis.set(
        xlabel="mean predicted probability",
        ylabel="observed trigger rate",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Trigger-probability reliability",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_2d_closure(
    rows: list[dict[str, Any]],
    momentum_edges: np.ndarray,
    theta_edges: np.ndarray,
    path: Path,
) -> None:
    shape = (len(theta_edges) - 1, len(momentum_edges) - 1)
    observed = np.full(shape, np.nan)
    predicted = np.full(shape, np.nan)
    for row in rows:
        index = (int(row["theta_bin"]), int(row["p_bin"]))
        observed[index] = row["observed_efficiency"]
        predicted[index] = row["fm_mean_probability"]
    difference = predicted - observed
    limit = max(float(np.nanmax(np.abs(difference))), 1e-4)
    figure, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.3), sharex=True, sharey=True, constrained_layout=True
    )
    extent = [momentum_edges[0], momentum_edges[-1], theta_edges[0], theta_edges[-1]]
    first = axes[0].imshow(observed, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    second = axes[1].imshow(predicted, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    third = axes[2].imshow(
        difference,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    for axis, title in zip(axes, ("full simulation", "Forward FM", "FM − simulation")):
        axis.set(title=title, xlabel=r"$p_{e,\rm gen}$ [GeV]")
    axes[0].set_ylabel(r"$\theta_{e,\rm gen}$ [deg]")
    figure.colorbar(first, ax=axes[0], label="trigger efficiency", shrink=0.82)
    figure.colorbar(second, ax=axes[1], label="trigger efficiency", shrink=0.82)
    figure.colorbar(third, ax=axes[2], label="closure residual", shrink=0.82)
    figure.suptitle(r"Two-dimensional closure of $P(T=1\mid x_e)$")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_history(history: list[dict[str, Any]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    for split in ("train", "validation"):
        axis.plot(
            epochs,
            [row[split]["binary_cross_entropy"] for row in history],
            label=split,
        )
    axis.set(xlabel="epoch", ylabel="binary cross-entropy", title="Trigger-efficiency training")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def evaluate_and_write(
    model: TriggerEfficiencyNet,
    splits: dict[str, TriggerSplit],
    history: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    test = splits["test"]
    probability = predict_probabilities(
        model,
        test,
        device,
        int(config["training"]["batch_size"]),
        int(config["project"]["seed"]),
    )
    target = test.trigger_target
    calibration = calibration_rows(
        target, probability, int(evaluation["calibration_bins"])
    )
    _write_csv(calibration, run_dir / "calibration_curve.csv")
    _plot_calibration(calibration, run_dir / "calibration_curve.png")

    variables = {
        "gen_p": (
            test.raw_gen_p,
            np.asarray(evaluation["momentum_edges_gev"], dtype=float),
            r"$p_{e,\rm gen}$ [GeV]",
        ),
        "gen_theta": (
            np.rad2deg(test.raw_gen_theta),
            np.asarray(evaluation["theta_edges_deg"], dtype=float),
            r"$\theta_{e,\rm gen}$ [deg]",
        ),
        "gen_phi": (
            np.rad2deg(test.raw_gen_phi),
            np.asarray(evaluation["phi_edges_deg"], dtype=float),
            r"$\phi_{e,\rm gen}$ [deg]",
        ),
        "gen_vz": (
            test.raw_gen_vz,
            np.asarray(evaluation["vz_edges_cm"], dtype=float),
            r"$v_{z,e}^{\rm gen}$ [cm]",
        ),
    }
    closure_summary: dict[str, Any] = {}
    for variable, (values, edges, xlabel) in variables.items():
        rows = efficiency_closure_rows(
            values,
            target,
            probability,
            edges,
            variable,
            int(evaluation["min_bin_count"]),
        )
        _write_csv(rows, run_dir / f"efficiency_vs_{variable}.csv")
        _plot_efficiency(rows, run_dir / f"efficiency_vs_{variable}.png", xlabel)
        closure_summary[variable] = _closure_summary(rows)

    momentum_edges = np.asarray(evaluation["momentum_edges_gev"], dtype=float)
    theta_edges = np.asarray(evaluation["theta_edges_deg"], dtype=float)
    closure_2d = efficiency_closure_2d_rows(
        test.raw_gen_p,
        np.rad2deg(test.raw_gen_theta),
        target,
        probability,
        momentum_edges,
        theta_edges,
        int(evaluation["min_bin_count_2d"]),
    )
    _write_csv(closure_2d, run_dir / "efficiency_vs_gen_p_theta.csv")
    _plot_2d_closure(
        closure_2d,
        momentum_edges,
        theta_edges,
        run_dir / "efficiency_vs_gen_p_theta.png",
    )
    closure_summary["gen_p_theta"] = _closure_summary(closure_2d)
    _plot_history(history, run_dir / "training_history.png")

    metrics = {
        "test": {
            "n": len(target),
            "observed_trigger_rate": float(target.mean()),
            "mean_predicted_probability": float(probability.mean()),
            "signed_integrated_difference": float(probability.mean() - target.mean()),
            "binary_cross_entropy": binary_log_loss(target, probability),
            "brier_score": float(np.mean((probability - target) ** 2)),
            "expected_calibration_error": expected_calibration_error(calibration),
            "roc_auc": _roc_auc(target, probability),
            "average_precision": _average_precision(target, probability),
            "threshold_accuracy": float(np.mean((probability >= 0.5) == (target >= 0.5))),
        },
        "closure": closure_summary,
        "checkpoint_selection": "minimum validation binary cross-entropy",
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metrics
