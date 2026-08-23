"""Held-out statistical and physical closure tests for the learned kernel.

If the learned kernel K_theta reproduces the full-simulation kernel K, samples
at fixed truth kinematics should reproduce conditional bias, resolution, tails,
PID confusion, and target correlations. Aggregate and binned checks here test
that necessary per-particle condition; they do not replace event/analysis-level
closure of derived observables.
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
from torch.utils.data import DataLoader

from .data import PreparedSplit, Standardizer, TARGET_COLUMNS
from .model import ConditionalMDN, sample_standardized_residuals
from .training import make_loader, run_epoch


SPECIES_LABELS = {-211: "pi-", 211: "pi+", 2212: "proton"}


@torch.no_grad()
def predict_test_sample(
    model: ConditionalMDN,
    split: PreparedSplit,
    target_scaler: Standardizer,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = make_loader(split, batch_size, shuffle=False, seed=seed, num_workers=0)
    sampled_targets: list[np.ndarray] = []
    predicted_pid: list[np.ndarray] = []
    pid_probabilities: list[np.ndarray] = []
    generator = (
        torch.Generator(device=device.type).manual_seed(seed + 101)
        if device.type in {"cpu", "cuda"}
        else None
    )
    model.eval()
    for continuous, species_index, _, _ in loader:
        output = model(continuous.to(device), species_index.to(device))
        sampled = sample_standardized_residuals(output, generator=generator)
        sampled_targets.append(sampled.cpu().numpy())
        probabilities = torch.softmax(output.pid_logits, dim=-1)
        pid_probabilities.append(probabilities.cpu().numpy())
        predicted_pid.append(torch.multinomial(probabilities, 1, generator=generator).squeeze(1).cpu().numpy())
    normalized = np.concatenate(sampled_targets, axis=0)
    physical_targets = target_scaler.inverse(normalized)
    physical_targets[:, 2] = (physical_targets[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    return (
        physical_targets,
        np.concatenate(predicted_pid),
        np.concatenate(pid_probabilities),
    )


def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
    }


def _wasserstein_equal_sample(left: np.ndarray, right: np.ndarray) -> float:
    """Empirical W1 = integral_0^1 |F_data^-1(u)-F_model^-1(u)| du."""
    return float(np.mean(np.abs(np.sort(left) - np.sort(right))))


def closure_rows(
    split: PreparedSplit,
    target_scaler: Standardizer,
    sampled_targets: np.ndarray,
) -> list[dict[str, Any]]:
    """Compare bias E[Delta], resolution Std[Delta], quantiles, and W1."""
    observed = target_scaler.inverse(split.targets)
    rows: list[dict[str, Any]] = []
    for species in sorted(SPECIES_LABELS):
        mask = split.raw_species == species
        for target_index, target in enumerate(TARGET_COLUMNS):
            truth = observed[mask, target_index]
            sample = sampled_targets[mask, target_index]
            truth_stats = _distribution_stats(truth)
            sample_stats = _distribution_stats(sample)
            row: dict[str, Any] = {
                "species_pid": species,
                "species": SPECIES_LABELS[species],
                "target": target,
                "n": int(mask.sum()),
                "wasserstein_1d": _wasserstein_equal_sample(truth, sample),
            }
            row.update({f"observed_{key}": value for key, value in truth_stats.items()})
            row.update({f"sampled_{key}": value for key, value in sample_stats.items()})
            row["absolute_mean_difference"] = abs(sample_stats["mean"] - truth_stats["mean"])
            row["std_ratio"] = (
                sample_stats["std"] / truth_stats["std"] if truth_stats["std"] > 0 else float("nan")
            )
            rows.append(row)
    return rows


def _raw_kinematics(split: PreparedSplit, feature_scaler: Standardizer) -> dict[str, np.ndarray]:
    features = feature_scaler.inverse(split.continuous)
    return {
        "gen_p": np.expm1(features[:, 0]),
        "gen_theta": features[:, 1],
        "gen_phi": np.arctan2(features[:, 2], features[:, 3]),
    }


def kinematic_closure_rows(
    split: PreparedSplit,
    feature_scaler: Standardizer,
    target_scaler: Standardizer,
    sampled_targets: np.ndarray,
) -> list[dict[str, Any]]:
    """Test conditional closure versus p_gen, theta_gen, and phi_gen.

    Global agreement can hide phase-space failures. These bins approximate
    moments of p(Delta|p,theta,phi,s) separately for each generated species.
    """
    observed = target_scaler.inverse(split.targets)
    kinematics = _raw_kinematics(split, feature_scaler)
    rows: list[dict[str, Any]] = []
    for species in sorted(SPECIES_LABELS):
        species_mask = split.raw_species == species
        for variable, values in kinematics.items():
            if variable == "gen_phi":
                edges = np.linspace(-np.pi, np.pi, 7)
            else:
                edges = np.unique(np.quantile(values[species_mask], np.linspace(0.0, 1.0, 6)))
            for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
                inclusive_high = bin_index == len(edges) - 2
                bin_mask = species_mask & (values >= low) & (
                    (values <= high) if inclusive_high else (values < high)
                )
                if bin_mask.sum() < 20:
                    continue
                for target_index, target in enumerate(TARGET_COLUMNS):
                    truth = observed[bin_mask, target_index]
                    sample = sampled_targets[bin_mask, target_index]
                    truth_std = float(np.std(truth))
                    sample_std = float(np.std(sample))
                    rows.append(
                        {
                            "species_pid": species,
                            "species": SPECIES_LABELS[species],
                            "conditioning_variable": variable,
                            "bin_index": bin_index,
                            "bin_low": float(low),
                            "bin_high": float(high),
                            "n": int(bin_mask.sum()),
                            "target": target,
                            "observed_mean": float(np.mean(truth)),
                            "sampled_mean": float(np.mean(sample)),
                            "absolute_mean_difference": float(abs(np.mean(sample) - np.mean(truth))),
                            "observed_std": truth_std,
                            "sampled_std": sample_std,
                            "std_ratio": sample_std / truth_std if truth_std > 0 else float("nan"),
                            "wasserstein_1d": _wasserstein_equal_sample(truth, sample),
                        }
                    )
    return rows


def joint_and_physical_metrics(
    split: PreparedSplit,
    feature_scaler: Standardizer,
    target_scaler: Standardizer,
    sampled_targets: np.ndarray,
) -> dict[str, Any]:
    """Check residual correlations and reconstruct physical sampled kinematics.

    Samples are mapped back through

        p_rec     = p_gen     + Delta p,
        theta_rec = theta_gen + Delta theta,

    and must obey p_rec>0 and 0<=theta_rec<=pi. The correlation comparison
    checks whether the joint model preserves Cov(Delta_i,Delta_j).
    """
    observed = target_scaler.inverse(split.targets)
    kinematics = _raw_kinematics(split, feature_scaler)
    correlations: dict[str, Any] = {}
    for species in sorted(SPECIES_LABELS):
        mask = split.raw_species == species
        observed_correlation = np.corrcoef(observed[mask].T)
        sampled_correlation = np.corrcoef(sampled_targets[mask].T)
        correlations[SPECIES_LABELS[species]] = {
            "observed": observed_correlation.tolist(),
            "sampled": sampled_correlation.tolist(),
            "frobenius_difference": float(
                np.linalg.norm(observed_correlation - sampled_correlation)
            ),
        }
    sampled_p = kinematics["gen_p"] + sampled_targets[:, 0]
    sampled_theta = kinematics["gen_theta"] + sampled_targets[:, 1]
    physical = (sampled_p > 0) & (sampled_theta >= 0) & (sampled_theta <= np.pi)
    return {
        "target_correlations_by_species": correlations,
        "physical_sample_fraction": float(np.mean(physical)),
        "sampled_rec_theta_below_33deg_fraction": float(
            np.mean((sampled_theta >= 0) & (sampled_theta < np.deg2rad(33.0)))
        ),
    }


def pid_metrics(
    split: PreparedSplit,
    predicted_pid: np.ndarray,
    pid_probabilities: np.ndarray,
    rec_pid_vocabulary: list[int],
) -> dict[str, Any]:
    class_count = len(rec_pid_vocabulary) + 1
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for observed, predicted in zip(split.rec_pid_index, predicted_pid):
        confusion[int(observed), int(predicted)] += 1
    labels: list[int | str] = [*rec_pid_vocabulary, "OTHER"]
    observed_fractions = np.bincount(split.rec_pid_index, minlength=class_count) / len(split)
    sampled_fractions = np.bincount(predicted_pid, minlength=class_count) / len(split)
    probability_fractions = pid_probabilities.mean(axis=0)
    return {
        "labels": labels,
        "sampled_confusion_counts": confusion.tolist(),
        "observed_class_fractions": observed_fractions.tolist(),
        "sampled_class_fractions": sampled_fractions.tolist(),
        "mean_predicted_probabilities": probability_fractions.tolist(),
        "max_absolute_fraction_difference": float(
            np.max(np.abs(observed_fractions - probability_fractions))
        ),
    }


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(history: list[dict[str, Any]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train"]["total_loss"] for row in history], label="train")
    axes[0].plot(
        epochs,
        [row["validation"]["total_loss"] for row in history],
        label="validation",
    )
    axes[0].set(xlabel="epoch", ylabel="joint loss", title="Training history")
    axes[0].legend()
    axes[1].plot(
        epochs,
        [row["validation"]["pid_accuracy"] for row in history],
        color="tab:green",
    )
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Validation REC-PID accuracy")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_closure(
    split: PreparedSplit,
    target_scaler: Standardizer,
    sampled_targets: np.ndarray,
    path: Path,
) -> None:
    observed = target_scaler.inverse(split.targets)
    figure, axes = plt.subplots(3, 3, figsize=(13, 10))
    for row_index, species in enumerate(sorted(SPECIES_LABELS)):
        mask = split.raw_species == species
        for column_index, target in enumerate(TARGET_COLUMNS):
            axis = axes[row_index, column_index]
            combined = np.concatenate(
                [observed[mask, column_index], sampled_targets[mask, column_index]]
            )
            low, high = np.quantile(combined, [0.005, 0.995])
            bins = np.linspace(low, high, 70)
            axis.hist(
                observed[mask, column_index], bins=bins, density=True, histtype="step", label="full sim"
            )
            axis.hist(
                sampled_targets[mask, column_index],
                bins=bins,
                density=True,
                histtype="step",
                label="MDN sample",
            )
            axis.set_title(f"{SPECIES_LABELS[species]}: {target}")
            if row_index == 0 and column_index == 0:
                axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def evaluate_and_write(
    model: ConditionalMDN,
    splits: dict[str, PreparedSplit],
    feature_scaler: Standardizer,
    target_scaler: Standardizer,
    rec_pid_vocabulary: list[int],
    history: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    training = config["training"]
    seed = int(config["project"]["seed"])
    test_loader: DataLoader = make_loader(
        splits["test"],
        int(training["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=0,
    )
    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        float(training["pid_loss_weight"]),
    )
    sampled_targets, sampled_pid, pid_probabilities = predict_test_sample(
        model,
        splits["test"],
        target_scaler,
        device,
        int(training["batch_size"]),
        seed,
    )
    rows = closure_rows(splits["test"], target_scaler, sampled_targets)
    kinematic_rows = kinematic_closure_rows(
        splits["test"], feature_scaler, target_scaler, sampled_targets
    )
    pid = pid_metrics(splits["test"], sampled_pid, pid_probabilities, rec_pid_vocabulary)
    joint_and_physical = joint_and_physical_metrics(
        splits["test"], feature_scaler, target_scaler, sampled_targets
    )
    write_rows_csv(rows, run_dir / "closure_metrics.csv")
    write_rows_csv(kinematic_rows, run_dir / "kinematic_closure_metrics.csv")
    plot_history(history, run_dir / "training_history.png")
    plot_closure(splits["test"], target_scaler, sampled_targets, run_dir / "residual_closure.png")
    metrics = {
        "test": test_metrics.as_dict(),
        "closure": rows,
        "kinematic_closure": kinematic_rows,
        "pid": pid,
        "joint_and_physical": joint_and_physical,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    return metrics
