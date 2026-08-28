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

from .data import (
    BETA_TARGET_COLUMN,
    PreparedSplit,
    Standardizer,
    generated_beta,
)
from .model import ConditionalMDN, sample_standardized_residuals
from .training import make_loader, run_epoch


SPECIES_LABELS = {-211: "pi-", 211: "pi+", 2212: "proton"}
DEFAULT_PID_MOMENTUM_EDGES_GEV = np.arange(0.0, 10.0, 1.0)


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
    if "delta_phi" in split.target_names:
        phi_index = split.target_names.index("delta_phi")
        physical_targets[:, phi_index] = (
            physical_targets[:, phi_index] + np.pi
        ) % (2.0 * np.pi) - np.pi
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
        for target_index, target in enumerate(split.target_names):
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
                for target_index, target in enumerate(split.target_names):
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
            "target_names": list(split.target_names),
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


def conditional_pid_response_rows(
    generated_species: np.ndarray,
    generated_momentum: np.ndarray,
    observed_pid_index: np.ndarray,
    predicted_probabilities: np.ndarray,
    class_labels: list[int | str],
    momentum_edges_gev: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate conditional PID closure in fixed generated-momentum bins.

    For each generated species ``s`` and momentum interval ``b``, the teacher
    response and Forward-FM response are

        P_COATJAVA(r | s,b) = N(s,b,r) / N(s,b),
        P_FM(r | s,b)       = (1/N(s,b)) sum_i q_theta(r | x_i).

    Here ``q_theta`` is the PID-head softmax vector.  This is a distributional
    closure test: it intentionally does not use argmax classification or
    stochastic categorical draws. Intervals are [low, high), except that the
    final interval includes its upper edge.
    """
    generated_species = np.asarray(generated_species)
    generated_momentum = np.asarray(generated_momentum, dtype=np.float64)
    observed_pid_index = np.asarray(observed_pid_index, dtype=np.int64)
    predicted_probabilities = np.asarray(predicted_probabilities, dtype=np.float64)
    edges = np.asarray(momentum_edges_gev, dtype=np.float64)

    row_count = len(generated_species)
    if not (
        len(generated_momentum) == row_count
        and len(observed_pid_index) == row_count
        and len(predicted_probabilities) == row_count
    ):
        raise ValueError("PID closure inputs must have identical row counts")
    if predicted_probabilities.ndim != 2:
        raise ValueError("predicted_probabilities must have shape (rows, classes)")
    if predicted_probabilities.shape[1] != len(class_labels):
        raise ValueError("class_labels must match the softmax class dimension")
    if len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("momentum_edges_gev must be strictly increasing")
    if not np.all(np.isfinite(predicted_probabilities)):
        raise ValueError("predicted_probabilities contains non-finite values")
    if np.any(observed_pid_index < 0) or np.any(observed_pid_index >= len(class_labels)):
        raise ValueError("observed_pid_index contains an out-of-range class")

    response_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for species in sorted(int(value) for value in np.unique(generated_species)):
        species_mask = generated_species == species
        for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            inclusive_high = bin_index == len(edges) - 2
            momentum_mask = (generated_momentum >= low) & (
                (generated_momentum <= high)
                if inclusive_high
                else (generated_momentum < high)
            )
            mask = species_mask & momentum_mask
            n = int(mask.sum())
            if n == 0:
                continue

            observed_counts = np.bincount(
                observed_pid_index[mask], minlength=len(class_labels)
            )
            observed_fractions = observed_counts / n
            probabilities = predicted_probabilities[mask]
            fm_fractions = probabilities.mean(axis=0)
            observed_se = np.sqrt(observed_fractions * (1.0 - observed_fractions) / n)
            if n > 1:
                fm_mean_se = probabilities.std(axis=0, ddof=1) / np.sqrt(n)
            else:
                fm_mean_se = np.zeros(len(class_labels), dtype=np.float64)
            differences = fm_fractions - observed_fractions

            for class_index, class_label in enumerate(class_labels):
                response_rows.append(
                    {
                        "generated_pid": species,
                        "generated_species": SPECIES_LABELS.get(species, str(species)),
                        "bin_index": bin_index,
                        "p_low_gev": float(low),
                        "p_high_gev": float(high),
                        "upper_edge_inclusive": inclusive_high,
                        "n": n,
                        "reconstructed_pid": class_label,
                        "observed_count": int(observed_counts[class_index]),
                        "coatjava_fraction": float(observed_fractions[class_index]),
                        "coatjava_standard_error": float(observed_se[class_index]),
                        "fm_mean_probability": float(fm_fractions[class_index]),
                        "fm_mean_standard_error": float(fm_mean_se[class_index]),
                        "signed_difference_fm_minus_coatjava": float(differences[class_index]),
                        "absolute_difference": float(abs(differences[class_index])),
                    }
                )

            worst_index = int(np.argmax(np.abs(differences)))
            summary_rows.append(
                {
                    "generated_pid": species,
                    "generated_species": SPECIES_LABELS.get(species, str(species)),
                    "bin_index": bin_index,
                    "p_low_gev": float(low),
                    "p_high_gev": float(high),
                    "upper_edge_inclusive": inclusive_high,
                    "n": n,
                    "total_variation_distance": float(0.5 * np.abs(differences).sum()),
                    "max_absolute_class_difference": float(abs(differences[worst_index])),
                    "worst_reconstructed_pid": class_labels[worst_index],
                    "coatjava_row_sum": float(observed_fractions.sum()),
                    "fm_row_sum": float(fm_fractions.sum()),
                }
            )
    return response_rows, summary_rows


def integrated_correct_pid_response(
    generated_species: np.ndarray,
    observed_pid_index: np.ndarray,
    predicted_probabilities: np.ndarray,
    class_labels: list[int | str],
) -> list[dict[str, Any]]:
    """Return momentum-integrated diagonal PID response for each generated species."""
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    rows: list[dict[str, Any]] = []
    for species in sorted(int(value) for value in np.unique(generated_species)):
        if species not in label_to_index:
            continue
        mask = generated_species == species
        class_index = label_to_index[species]
        rows.append(
            {
                "generated_pid": species,
                "generated_species": SPECIES_LABELS.get(species, str(species)),
                "n": int(mask.sum()),
                "coatjava_correct_fraction": float(
                    np.mean(observed_pid_index[mask] == class_index)
                ),
                "fm_correct_mean_probability": float(
                    predicted_probabilities[mask, class_index].mean()
                ),
            }
        )
    return rows


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
    target_count = len(split.target_names)
    figure, axes = plt.subplots(3, target_count, figsize=(4.3 * target_count, 10), squeeze=False)
    for row_index, species in enumerate(sorted(SPECIES_LABELS)):
        mask = split.raw_species == species
        for column_index, target in enumerate(split.target_names):
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


def plot_conditional_correct_pid_response(
    response_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Plot the diagonal response P(s_rec=s_gen | s_gen,p_gen)."""
    figure, axes = plt.subplots(1, len(SPECIES_LABELS), figsize=(15, 4.5), sharey=True)
    for axis, species in zip(axes, sorted(SPECIES_LABELS)):
        rows = [
            row
            for row in response_rows
            if row["generated_pid"] == species and row["reconstructed_pid"] == species
        ]
        centers = np.asarray(
            [(row["p_low_gev"] + row["p_high_gev"]) / 2.0 for row in rows]
        )
        if rows:
            axis.errorbar(
                centers,
                [row["coatjava_fraction"] for row in rows],
                yerr=[row["coatjava_standard_error"] for row in rows],
                marker="o",
                capsize=2,
                label="COATJAVA fraction",
            )
            axis.errorbar(
                centers,
                [row["fm_mean_probability"] for row in rows],
                yerr=[row["fm_mean_standard_error"] for row in rows],
                marker="s",
                capsize=2,
                label="Forward FM mean probability",
            )
        axis.set(
            xlabel=r"generated momentum $p_{\rm gen}$ [GeV]",
            title=f"generated {SPECIES_LABELS[species]}",
            ylim=(-0.03, 1.03),
        )
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"$P(s_{\rm rec}=s_{\rm gen}\mid s_{\rm gen},p_{\rm gen})$")
    axes[0].legend(fontsize=8)
    figure.suptitle("Conditional correct-PID response closure (fixed 1-GeV bins)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def beta_closure_rows(
    split: PreparedSplit,
    feature_scaler: Standardizer,
    target_scaler: Standardizer,
    sampled_targets: np.ndarray,
    momentum_edges_gev: np.ndarray,
    beta_min_exclusive: float,
    beta_max_inclusive: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare reconstructed beta distributions in fixed p_gen intervals.

    The model learns Delta beta around the relativistic truth reference

        beta_gen = p_gen / sqrt(p_gen^2 + m_s^2),
        beta_rec = beta_gen + Delta beta.

    All rows in a bin are paired only by the common held-out conditioning
    sample; distribution metrics do not treat the random MDN draw as a
    per-particle point prediction.
    """
    if BETA_TARGET_COLUMN not in split.target_names:
        raise ValueError("beta closure requires delta_beta in split.target_names")
    edges = np.asarray(momentum_edges_gev, dtype=np.float64)
    if len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("momentum_edges_gev must be strictly increasing")
    observed_targets = target_scaler.inverse(split.targets)
    kinematics = _raw_kinematics(split, feature_scaler)
    beta_index = split.target_names.index(BETA_TARGET_COLUMN)
    beta_reference = generated_beta(kinematics["gen_p"], split.raw_species)
    observed_beta = beta_reference + observed_targets[:, beta_index]
    sampled_beta = beta_reference + sampled_targets[:, beta_index]

    rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    for species in sorted(SPECIES_LABELS):
        species_mask = split.raw_species == species
        if not np.any(species_mask):
            continue
        observed_species = observed_beta[species_mask]
        sampled_species = sampled_beta[species_mask]
        observed_stats = _distribution_stats(observed_species)
        sampled_stats = _distribution_stats(sampled_species)
        overall_rows.append(
            {
                "generated_pid": species,
                "generated_species": SPECIES_LABELS[species],
                "n": int(species_mask.sum()),
                "wasserstein_1d": _wasserstein_equal_sample(
                    observed_species, sampled_species
                ),
                "absolute_mean_difference": float(
                    abs(sampled_stats["mean"] - observed_stats["mean"])
                ),
                "std_ratio": float(
                    sampled_stats["std"] / observed_stats["std"]
                    if observed_stats["std"] > 0
                    else 0.0
                ),
                "observed_in_training_domain_fraction": float(
                    np.mean(
                        (observed_species > beta_min_exclusive)
                        & (observed_species <= beta_max_inclusive)
                    )
                ),
                "sampled_in_training_domain_fraction": float(
                    np.mean(
                        (sampled_species > beta_min_exclusive)
                        & (sampled_species <= beta_max_inclusive)
                    )
                ),
                **{f"observed_{key}": value for key, value in observed_stats.items()},
                **{f"sampled_{key}": value for key, value in sampled_stats.items()},
            }
        )
        for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            inclusive_high = bin_index == len(edges) - 2
            bin_mask = species_mask & (kinematics["gen_p"] >= low) & (
                (kinematics["gen_p"] <= high)
                if inclusive_high
                else (kinematics["gen_p"] < high)
            )
            n = int(bin_mask.sum())
            if n < 20:
                continue
            observed_values = observed_beta[bin_mask]
            sampled_values = sampled_beta[bin_mask]
            observed_stats = _distribution_stats(observed_values)
            sampled_stats = _distribution_stats(sampled_values)
            rows.append(
                {
                    "generated_pid": species,
                    "generated_species": SPECIES_LABELS[species],
                    "bin_index": bin_index,
                    "p_low_gev": float(low),
                    "p_high_gev": float(high),
                    "upper_edge_inclusive": inclusive_high,
                    "n": n,
                    "observed_mean": observed_stats["mean"],
                    "sampled_mean": sampled_stats["mean"],
                    "absolute_mean_difference": float(
                        abs(sampled_stats["mean"] - observed_stats["mean"])
                    ),
                    "observed_std": observed_stats["std"],
                    "sampled_std": sampled_stats["std"],
                    "std_ratio": float(
                        sampled_stats["std"] / observed_stats["std"]
                        if observed_stats["std"] > 0
                        else 0.0
                    ),
                    "observed_q05": observed_stats["q05"],
                    "sampled_q05": sampled_stats["q05"],
                    "observed_median": observed_stats["median"],
                    "sampled_median": sampled_stats["median"],
                    "observed_q95": observed_stats["q95"],
                    "sampled_q95": sampled_stats["q95"],
                    "wasserstein_1d": _wasserstein_equal_sample(
                        observed_values, sampled_values
                    ),
                    "observed_in_training_domain_fraction": float(
                        np.mean(
                            (observed_values > beta_min_exclusive)
                            & (observed_values <= beta_max_inclusive)
                        )
                    ),
                    "sampled_in_training_domain_fraction": float(
                        np.mean(
                            (sampled_values > beta_min_exclusive)
                            & (sampled_values <= beta_max_inclusive)
                        )
                    ),
                }
            )
    return rows, overall_rows


def plot_beta_response_vs_gen_p(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Plot mean and central 90% beta response versus generated momentum."""
    figure, axes = plt.subplots(1, len(SPECIES_LABELS), figsize=(15, 4.6), sharey=True)
    for axis, species in zip(axes, sorted(SPECIES_LABELS)):
        species_rows = [row for row in rows if row["generated_pid"] == species]
        centers = np.asarray(
            [(row["p_low_gev"] + row["p_high_gev"]) / 2.0 for row in species_rows]
        )
        if species_rows:
            observed_mean = np.asarray([row["observed_mean"] for row in species_rows])
            sampled_mean = np.asarray([row["sampled_mean"] for row in species_rows])
            axis.fill_between(
                centers,
                [row["observed_q05"] for row in species_rows],
                [row["observed_q95"] for row in species_rows],
                color="tab:blue",
                alpha=0.16,
            )
            axis.fill_between(
                centers,
                [row["sampled_q05"] for row in species_rows],
                [row["sampled_q95"] for row in species_rows],
                color="tab:orange",
                alpha=0.16,
            )
            axis.plot(centers, observed_mean, "o-", color="tab:blue", label="COATJAVA")
            axis.plot(centers, sampled_mean, "s-", color="tab:orange", label="Forward FM")
            reference = generated_beta(centers, np.full(len(centers), species))
            axis.plot(centers, reference, "--", color="black", linewidth=1, label=r"$\beta_{gen}$")
        axis.set(
            xlabel=r"generated momentum $p_{\rm gen}$ [GeV]",
            title=f"generated {SPECIES_LABELS[species]}",
            ylim=(0.0, 1.2),
        )
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"reconstructed $\beta=v/c$")
    axes[0].legend(fontsize=8, loc="lower right")
    figure.suptitle(r"Continuous $\beta$ response closure; bands show 5th--95th percentiles")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_beta_vs_reconstructed_p(
    split: PreparedSplit,
    feature_scaler: Standardizer,
    target_scaler: Standardizer,
    sampled_targets: np.ndarray,
    path: Path,
) -> None:
    """Show the observed and jointly sampled (p_rec,beta_rec) PID-response plane."""
    observed_targets = target_scaler.inverse(split.targets)
    kinematics = _raw_kinematics(split, feature_scaler)
    delta_p_index = split.target_names.index("delta_p")
    beta_index = split.target_names.index(BETA_TARGET_COLUMN)
    beta_reference = generated_beta(kinematics["gen_p"], split.raw_species)
    observed_p = kinematics["gen_p"] + observed_targets[:, delta_p_index]
    sampled_p = kinematics["gen_p"] + sampled_targets[:, delta_p_index]
    observed_beta = beta_reference + observed_targets[:, beta_index]
    sampled_beta = beta_reference + sampled_targets[:, beta_index]

    figure, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True, sharey=True)
    momentum_grid = np.linspace(0.05, 9.0, 300)
    reference_masses = {
        "pion": 0.13957039,
        "kaon": 0.493677,
        "proton": 0.93827208816,
    }
    for row_index, species in enumerate(sorted(SPECIES_LABELS)):
        species_mask = split.raw_species == species
        for column_index, (title, momentum, beta) in enumerate(
            (("COATJAVA", observed_p, observed_beta), ("Forward FM", sampled_p, sampled_beta))
        ):
            axis = axes[row_index, column_index]
            finite = species_mask & np.isfinite(momentum) & np.isfinite(beta)
            axis.hexbin(
                momentum[finite],
                beta[finite],
                gridsize=(70, 45),
                extent=(0.0, 9.0, 0.0, 1.2),
                bins="log",
                mincnt=1,
                cmap="viridis",
            )
            for label, mass in reference_masses.items():
                curve = momentum_grid / np.sqrt(momentum_grid**2 + mass**2)
                axis.plot(
                    momentum_grid,
                    curve,
                    linewidth=1,
                    label=label if row_index == 0 and column_index == 0 else None,
                )
            axis.set_title(f"generated {SPECIES_LABELS[species]}: {title}")
            axis.grid(alpha=0.12)
    for axis in axes[-1]:
        axis.set_xlabel(r"reconstructed momentum $p_{\rm rec}$ [GeV]")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"reconstructed $\beta=v/c$")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    figure.suptitle(r"Observed and sampled timing response in the $(p_{rec},\beta_{rec})$ plane")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
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
    pid_labels: list[int | str] = [*rec_pid_vocabulary, "OTHER"]
    configured_edges = config.get("evaluation", {}).get("pid_momentum_edges_gev")
    momentum_edges = (
        np.asarray(configured_edges, dtype=np.float64)
        if configured_edges is not None
        else DEFAULT_PID_MOMENTUM_EDGES_GEV
    )
    generated_momentum = _raw_kinematics(splits["test"], feature_scaler)["gen_p"]
    conditional_pid_rows, conditional_pid_summary = conditional_pid_response_rows(
        splits["test"].raw_species,
        generated_momentum,
        splits["test"].rec_pid_index,
        pid_probabilities,
        pid_labels,
        momentum_edges,
    )
    integrated_correct_pid = integrated_correct_pid_response(
        splits["test"].raw_species,
        splits["test"].rec_pid_index,
        pid_probabilities,
        pid_labels,
    )
    inside_pid_momentum_range = (generated_momentum >= momentum_edges[0]) & (
        generated_momentum <= momentum_edges[-1]
    )
    pid_momentum_coverage = []
    for species in sorted(SPECIES_LABELS):
        species_mask = splits["test"].raw_species == species
        total = int(species_mask.sum())
        inside = int((species_mask & inside_pid_momentum_range).sum())
        pid_momentum_coverage.append(
            {
                "generated_pid": species,
                "generated_species": SPECIES_LABELS[species],
                "test_rows": total,
                "rows_inside_configured_range": inside,
                "coverage_fraction": inside / total,
            }
        )
    joint_and_physical = joint_and_physical_metrics(
        splits["test"], feature_scaler, target_scaler, sampled_targets
    )
    beta_metrics: dict[str, Any] | None = None
    if BETA_TARGET_COLUMN in splits["test"].target_names:
        beta_config = config["data"]["beta_response"]
        configured_beta_edges = config.get("evaluation", {}).get(
            "beta_momentum_edges_gev", momentum_edges
        )
        beta_edges = np.asarray(configured_beta_edges, dtype=np.float64)
        beta_rows, beta_overall = beta_closure_rows(
            splits["test"],
            feature_scaler,
            target_scaler,
            sampled_targets,
            beta_edges,
            float(beta_config["rec_beta_min_exclusive"]),
            float(beta_config["rec_beta_max_inclusive"]),
        )
        beta_metrics = {
            "definition": (
                "beta_rec = p_gen/sqrt(p_gen^2+m_s^2) + sampled_delta_beta"
            ),
            "momentum_edges_gev": beta_edges.tolist(),
            "training_domain": {
                "rec_beta_min_exclusive": float(beta_config["rec_beta_min_exclusive"]),
                "rec_beta_max_inclusive": float(beta_config["rec_beta_max_inclusive"]),
            },
            "overall_by_generated_species": beta_overall,
            "fixed_momentum_bins": beta_rows,
        }
        write_rows_csv(beta_rows, run_dir / "beta_closure_vs_gen_p.csv")
        write_rows_csv(beta_overall, run_dir / "beta_closure_overall.csv")
        plot_beta_response_vs_gen_p(beta_rows, run_dir / "beta_response_vs_gen_p.png")
        plot_beta_vs_reconstructed_p(
            splits["test"],
            feature_scaler,
            target_scaler,
            sampled_targets,
            run_dir / "beta_vs_reconstructed_p.png",
        )
    write_rows_csv(rows, run_dir / "closure_metrics.csv")
    write_rows_csv(kinematic_rows, run_dir / "kinematic_closure_metrics.csv")
    write_rows_csv(conditional_pid_rows, run_dir / "pid_response_fixed_bins.csv")
    write_rows_csv(conditional_pid_summary, run_dir / "pid_bin_closure_summary.csv")
    write_rows_csv(integrated_correct_pid, run_dir / "pid_integrated_correct_id.csv")
    plot_history(history, run_dir / "training_history.png")
    plot_closure(splits["test"], target_scaler, sampled_targets, run_dir / "residual_closure.png")
    plot_conditional_correct_pid_response(
        conditional_pid_rows, run_dir / "pid_correct_response_vs_gen_p.png"
    )
    metrics = {
        "test": test_metrics.as_dict(),
        "closure": rows,
        "kinematic_closure": kinematic_rows,
        "pid": pid,
        "pid_conditional_closure": {
            "definition": "Observed class fractions versus mean PID-head softmax probabilities",
            "momentum_edges_gev": momentum_edges.tolist(),
            "momentum_range_coverage": pid_momentum_coverage,
            "bin_summary": conditional_pid_summary,
            "integrated_correct_id": integrated_correct_pid,
        },
        "joint_and_physical": joint_and_physical,
    }
    if beta_metrics is not None:
        metrics["beta_closure"] = beta_metrics
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    return metrics
