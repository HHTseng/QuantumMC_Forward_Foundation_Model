"""Held-out closure of P(C | x_h, species, T=1), before residual/PID cuts.

For a bin B, MC efficiency is sum(1{C_i=FD}) / N_B and FM efficiency
is sum(P_FM(C_i=FD | x_i)) / N_B, not the rate of thresholded decisions.
The any-region probability is 1-P(C=unreconstructed), so FD <= any by design.
"""

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

from .data import OUTCOME_LABELS


def wilson_interval(successes: float, n: int, z: float = 1.0) -> tuple[float, float]:
    """Binomial Wilson interval; z=1 gives approximately 68% coverage.

    Unlike sqrt(epsilon*(1-epsilon)/N), the interval remains nonzero at
    zero successes and at N successes. This is an MC sampling interval,
    not an uncertainty on model parameters or on a detector systematic.
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return float(max(0, center - half)), float(min(1, center + half))


def reliability_rows(
    target: np.ndarray, probability: np.ndarray, n_bins: int = 20
) -> list[dict[str, Any]]:
    if len(target) != len(probability) or len(target) == 0:
        raise ValueError("Reliability requires equal nonempty arrays")
    if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be finite and within [0,1]")
    rows = []
    edges = np.linspace(0, 1, n_bins + 1)
    assignment = np.minimum((probability * n_bins).astype(np.int64), n_bins - 1)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = assignment == index
        n = int(mask.sum())
        observed = float(np.mean(target[mask])) if n else None
        predicted = float(np.mean(probability[mask], dtype=np.float64)) if n else None
        lo, hi = wilson_interval(float(np.sum(target[mask])), n)
        rows.append({
            "bin_index": index, "probability_low": float(low),
            "probability_high": float(high), "upper_edge_inclusive": index == n_bins - 1,
            "n": n, "mean_predicted_probability": predicted,
            "observed_efficiency": observed,
            "wilson_low": lo if n else None, "wilson_high": hi if n else None,
            "absolute_calibration_gap": abs(predicted - observed) if n else None,
        })
    return rows


def binary_metrics(
    target: np.ndarray, probability: np.ndarray, train_rate: float, n_bins: int = 20
) -> dict[str, Any]:
    rows = reliability_rows(target, probability, n_bins)
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    baseline = float(np.clip(train_rate, 1e-12, 1 - 1e-12))
    baseline_brier = float(np.mean((y - train_rate) ** 2))
    brier = float(np.mean((y - p) ** 2))
    occupied = [row for row in rows if row["n"]]
    lo, hi = wilson_interval(float(y.sum()), len(y))
    return {
        "n": len(y), "successes": int(y.sum()), "observed_rate": float(y.mean()),
        "mean_predicted_probability": float(p.mean()), "mean_bias": float(p.mean() - y.mean()),
        "observed_wilson_low": lo, "observed_wilson_high": hi,
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "brier_score": brier,
        "expected_calibration_error": float(sum(
            row["n"] * row["absolute_calibration_gap"] for row in occupied
        ) / len(y)),
        "maximum_calibration_gap": max(row["absolute_calibration_gap"] for row in occupied),
        "occupied_probability_bins": len(occupied), "calibration_bins": n_bins,
        "constant_train_rate": float(train_rate), "constant_baseline_brier": baseline_brier,
        "constant_baseline_log_loss": float(-np.mean(
            y * np.log(baseline) + (1 - y) * np.log1p(-baseline)
        )),
        "brier_skill_vs_constant": 1 - brier / baseline_brier if baseline_brier > 0 else None,
    }


def binned_rows(
    values: np.ndarray, target: np.ndarray, probability: np.ndarray, edges: np.ndarray,
    variable: str, min_bin_count: int = 100,
) -> list[dict[str, Any]]:
    edges = np.asarray(edges, dtype=np.float64)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("Bin edges must be strictly increasing")
    rows = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (values >= low) & (values <= high if index == len(edges) - 2 else values < high)
        n = int(mask.sum())
        successes = int(np.sum(target[mask]))
        observed = successes / n if n else None
        predicted = float(np.mean(probability[mask], dtype=np.float64)) if n else None
        lo, hi = wilson_interval(successes, n)
        rows.append({
            "variable": variable, "bin_index": index, "low": float(low), "high": float(high),
            "upper_edge_inclusive": index == len(edges) - 2, "n": n, "successes": successes,
            "eligible": n >= min_bin_count, "observed_efficiency": observed,
            "fm_mean_probability": predicted, "wilson_low": lo if n else None,
            "wilson_high": hi if n else None,
            "signed_difference_fm_minus_data": predicted - observed if n else None,
            "absolute_difference": abs(predicted - observed) if n else None,
        })
    return rows


def closure_summary(rows: list[dict[str, Any]], total_rows: int) -> dict[str, Any]:
    accepted = [row for row in rows if row["eligible"]]
    accepted_count = sum(row["n"] for row in accepted)
    covered_count = sum(row["n"] for row in rows)
    return {
        "eligible_bins": len(accepted), "all_bins": len(rows),
        "eligible_rows": accepted_count, "low_count_rows": covered_count - accepted_count,
        "out_of_range_rows": total_rows - covered_count,
        "weighted_mae": sum(row["n"] * row["absolute_difference"] for row in accepted) / accepted_count
        if accepted_count else None,
        "unweighted_mae": float(np.mean([row["absolute_difference"] for row in accepted]))
        if accepted else None,
        "maximum_absolute_difference": max(row["absolute_difference"] for row in accepted)
        if accepted else None,
    }


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def predict(model: torch.nn.Module, split: Any, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    results = []
    for start in range(0, len(split), batch_size):
        batch = torch.from_numpy(split.continuous[start:start + batch_size]).to(device)
        results.append(torch.softmax(model(batch), dim=-1).cpu().numpy())
    probability = np.concatenate(results)
    if not np.all(np.isfinite(probability)) or not np.allclose(probability.sum(axis=1), 1, atol=2e-6):
        raise ValueError("Invalid categorical probabilities")
    return probability


def _plot_closure(rows: list[dict[str, Any]], path: Path, xlabel: str, target_name: str) -> None:
    rows = [row for row in rows if row["eligible"]]
    if not rows:
        return
    x = np.array([(row["low"] + row["high"]) / 2 for row in rows])
    observed = np.array([row["observed_efficiency"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].errorbar(x, observed, yerr=[
        observed - np.array([row["wilson_low"] for row in rows]),
        np.array([row["wilson_high"] for row in rows]) - observed,
    ], fmt="o", capsize=2, label="Full MC (68% Wilson interval)")
    axes[0].plot(x, [row["fm_mean_probability"] for row in rows], "s-", label="FM mean probability")
    axes[0].set(ylabel=f"{target_name} efficiency given electron trigger", ylim=(-.025, 1.025))
    axes[0].legend(fontsize=8)
    axes[1].plot(x, 100 * np.array([row["signed_difference_fm_minus_data"] for row in rows]), "o-")
    axes[1].axhline(0, color=".4", linestyle="--")
    axes[1].set(xlabel=xlabel, ylabel="FM-MC [pp]")
    for axis in axes:
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _two_dimensional_rows(split: Any, target: np.ndarray, probability: np.ndarray,
                          p_edges: np.ndarray, theta_edges: np.ndarray,
                          min_count: int) -> list[dict[str, Any]]:
    rows = []
    theta = np.rad2deg(split.raw_gen_theta)
    for pi, (p_low, p_high) in enumerate(zip(p_edges[:-1], p_edges[1:])):
        mask = (split.raw_gen_p >= p_low) & (
            split.raw_gen_p <= p_high if pi == len(p_edges) - 2 else split.raw_gen_p < p_high
        )
        for row in binned_rows(theta[mask], target[mask], probability[mask], theta_edges, "gen_theta", min_count):
            rows.append({"p_bin_index": pi, "p_low": float(p_low), "p_high": float(p_high), **row})
    return rows


def _plot_two_dimensional(rows: list[dict[str, Any]], p_edges: np.ndarray,
                          theta_edges: np.ndarray, path: Path, label: str) -> None:
    shape = (len(theta_edges) - 1, len(p_edges) - 1)
    observed = np.full(shape, np.nan)
    predicted = np.full(shape, np.nan)
    for row in rows:
        if row["eligible"]:
            position = (row["bin_index"], row["p_bin_index"])
            observed[position] = row["observed_efficiency"]
            predicted[position] = row["fm_mean_probability"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4), sharey=True)
    for axis, matrix, title in zip(axes[:2], (observed, predicted), ("Full MC", "FM mean probability")):
        mesh = axis.pcolormesh(p_edges, theta_edges, matrix, vmin=0, vmax=1, cmap="viridis")
        fig.colorbar(mesh, ax=axis, label="efficiency")
        axis.set_title(title)
    limit = max(.01, float(np.nanmax(np.abs(predicted - observed)))) if np.any(np.isfinite(observed)) else .01
    mesh = axes[2].pcolormesh(p_edges, theta_edges, 100 * (predicted - observed),
                             vmin=-100 * limit, vmax=100 * limit, cmap="RdBu_r")
    fig.colorbar(mesh, ax=axes[2], label="FM-MC [pp]")
    axes[2].set_title("Closure difference")
    for axis in axes:
        axis.set_xlabel("Generated momentum [GeV]")
    axes[0].set_ylabel("Generated polar angle [deg]")
    fig.suptitle(f"{label} reconstruction given electron trigger; blank cells: insufficient count")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def evaluate_and_write(model: torch.nn.Module, splits: dict[str, Any],
                       history: list[dict[str, Any]], config: dict[str, Any],
                       device: torch.device, run_dir: Path) -> dict[str, Any]:
    evaluation = config["evaluation"]
    batch_size = int(config["training"]["batch_size"])
    n_bins = int(evaluation.get("calibration_bins", 20))
    min_count = int(evaluation.get("min_bin_count", 100))
    train_rates = np.bincount(splits["train"].outcome_target, minlength=len(OUTCOME_LABELS)) / len(splits["train"])
    results = {}
    for split_name in ("validation", "test"):
        split = splits[split_name]
        probability = predict(model, split, device, batch_size)
        y = split.outcome_target
        one_hot = np.eye(len(OUTCOME_LABELS))[y]
        metrics = {"split": split_name, "outcome": {
            "cross_entropy": float(-np.log(np.clip(probability[np.arange(len(y)), y], 1e-12, 1)).mean()),
            "multiclass_brier_score": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
            "counts": {label: int(np.sum(y == index)) for index, label in enumerate(OUTCOME_LABELS)},
            "class_metrics": {
                label: binary_metrics(y == index, probability[:, index], float(train_rates[index]), n_bins)
                for index, label in enumerate(OUTCOME_LABELS)
            },
        }, "kinematic_closure": {}}
        prefix = "validation_" if split_name == "validation" else ""
        variables = {
            "gen_p": (split.raw_gen_p, np.asarray(evaluation["momentum_edges_gev"], dtype=float), "Generated momentum [GeV]"),
            "gen_theta": (np.rad2deg(split.raw_gen_theta), np.asarray(evaluation["theta_edges_deg"], dtype=float), "Generated polar angle [deg]"),
            "gen_phi": (np.rad2deg(split.raw_gen_phi), np.asarray(evaluation["phi_edges_deg"], dtype=float), "Generated azimuth [deg]"),
            "gen_vz": (split.raw_gen_vz, np.asarray(evaluation["vz_edges_cm"], dtype=float), "Generated vertex z [cm]"),
        }
        for name, target, prediction, rate in (
            ("fd", y == 1, probability[:, 1], train_rates[1]),
            ("any", y != 0, 1 - probability[:, 0], 1 - train_rates[0]),
        ):
            metrics[name] = binary_metrics(target, prediction, float(rate), n_bins)
            reliability = reliability_rows(target, prediction, n_bins)
            write_rows(reliability, run_dir / f"{prefix}calibration_{name}.csv")
            metrics["kinematic_closure"][name] = {}
            for variable, (values, edges, xlabel) in variables.items():
                rows = binned_rows(values, target, prediction, edges, variable, min_count)
                write_rows(rows, run_dir / f"{prefix}efficiency_{name}_vs_{variable}.csv")
                metrics["kinematic_closure"][name][variable] = closure_summary(rows, len(split))
                if split_name == "test":
                    _plot_closure(rows, run_dir / f"efficiency_{name}_vs_{variable}.png", xlabel, name.upper())
            rows_2d = _two_dimensional_rows(split, target, prediction, variables["gen_p"][1],
                                             variables["gen_theta"][1], min_count)
            write_rows(rows_2d, run_dir / f"{prefix}efficiency_{name}_p_theta.csv")
            metrics["kinematic_closure"][name]["gen_p_theta"] = closure_summary(rows_2d, len(split))
            if split_name == "test":
                _plot_two_dimensional(rows_2d, variables["gen_p"][1], variables["gen_theta"][1],
                                      run_dir / f"efficiency_{name}_p_theta.png", name.upper())
        with (run_dir / ("metrics.json" if split_name == "test" else "validation_metrics.json")).open("w") as stream:
            json.dump(metrics, stream, indent=2, allow_nan=False)
        if split_name == "test":
            np.savez_compressed(run_dir / "predictions.npz", outcome_target=y,
                outcome_probability=probability, event_keys=split.event_keys.astype(np.uint64),
                gen_p=split.raw_gen_p, gen_theta=split.raw_gen_theta,
                gen_phi=split.raw_gen_phi, gen_vz=split.raw_gen_vz)
            fig, axis = plt.subplots(figsize=(5, 5))
            axis.plot([0, 1], [0, 1], "--", color=".5", label="Ideal")
            for name, target, prediction in (("FD", y == 1, probability[:, 1]), ("Any region", y != 0, 1 - probability[:, 0])):
                rows = [row for row in reliability_rows(target, prediction, n_bins) if row["n"]]
                axis.plot([row["mean_predicted_probability"] for row in rows],
                          [row["observed_efficiency"] for row in rows], "o-", label=name)
            axis.set(xlabel="Mean predicted probability", ylabel="Observed reconstruction rate",
                     xlim=(0, 1), ylim=(0, 1), title="Hadron calibration given electron trigger")
            axis.grid(alpha=.25)
            axis.legend()
            fig.tight_layout()
            fig.savefig(run_dir / "calibration.png", dpi=170)
            plt.close(fig)
        results[split_name] = metrics
    if history:
        fig, axis = plt.subplots(figsize=(6, 4))
        epochs = [row["epoch"] for row in history]
        axis.plot(epochs, [row["train_loss"] for row in history], label="Train")
        axis.plot(epochs, [row["validation_loss"] for row in history], label="Validation")
        axis.set(xlabel="Epoch", ylabel="Categorical cross entropy")
        axis.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "training_history.png", dpi=170)
        plt.close(fig)
    return results["test"]
