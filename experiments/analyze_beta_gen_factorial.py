#!/usr/bin/env python3
"""Aggregate the matched beta-gen factorial experiment over model seeds."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = tuple(range(20260822, 20260832))
VARIANTS = (
    "A_original",
    "B_beta_input",
    "C_beta_target",
    "D_input_target_pid02",
    "A_original_pid1",
    "B_beta_input_pid1",
    "D_input_target_pid1",
)
VARIANT_LABELS = {
    "A_original": r"A: original, $\lambda=0.2$",
    "B_beta_input": r"B: $\beta_{gen}$ input, $\lambda=0.2$",
    "C_beta_target": r"C: $\Delta\beta$ target, $\lambda=0.2$",
    "D_input_target_pid02": r"D: input + target, $\lambda=0.2$",
    "A_original_pid1": r"A: original, $\lambda=1.0$",
    "B_beta_input_pid1": r"B: $\beta_{gen}$ input, $\lambda=1.0$",
    "D_input_target_pid1": r"D: input + target, $\lambda=1.0$",
}
GENERATED_SPECIES = (-211, 211, 2212)
SPECIES_KEY = {-211: "pi_minus", 211: "pi_plus", 2212: "proton"}
SPECIES_LABEL = {-211: r"$\pi^-$", 211: r"$\pi^+$", 2212: "proton"}
CONTRASTS = (
    ("beta_input_without_target", "A_original", "B_beta_input"),
    ("delta_beta_target_without_input", "A_original", "C_beta_target"),
    ("combined_vs_original", "A_original", "D_input_target_pid02"),
    ("beta_input_with_target", "C_beta_target", "D_input_target_pid02"),
    ("delta_beta_target_with_input", "B_beta_input", "D_input_target_pid02"),
    ("beta_input_pid1", "A_original_pid1", "B_beta_input_pid1"),
    ("delta_beta_target_pid1", "B_beta_input_pid1", "D_input_target_pid1"),
    ("pid_weight_original", "A_original", "A_original_pid1"),
    ("pid_weight_beta_input", "B_beta_input", "B_beta_input_pid1"),
    ("pid_weight_combined", "D_input_target_pid02", "D_input_target_pid1"),
)
T_CRITICAL_95 = {
    1: 12.7062,
    2: 4.3027,
    3: 3.1824,
    4: 2.7764,
    5: 2.5706,
    6: 2.4469,
    7: 2.3646,
    8: 2.3060,
    9: 2.2622,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default=str(REPOSITORY_ROOT / "runs/gpu_beta_gen_factorial"),
    )
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration {path} is not a mapping")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1))


def paired_statistics(control: np.ndarray, treatment: np.ndarray) -> dict[str, Any]:
    """Analyze d_j=M_control,j-M_treatment,j for lower-is-better M."""
    control = np.asarray(control, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    if control.shape != treatment.shape or len(control) < 2:
        raise ValueError("Paired vectors must have the same length and at least two seeds")
    improvement = control - treatment
    n = len(improvement)
    if n - 1 not in T_CRITICAL_95:
        raise ValueError("Supported paired-study size is 2--10 seeds")
    mean = float(improvement.mean())
    sd = float(improvement.std(ddof=1))
    half_width = T_CRITICAL_95[n - 1] * sd / np.sqrt(n)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)))
    null_values = np.abs(np.mean(signs * improvement[None, :], axis=1))
    return {
        "n_pairs": n,
        "mean_improvement": mean,
        "median_improvement": float(np.median(improvement)),
        "improvement_sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "favorable_pairs": int(np.sum(improvement > 0.0)),
        "exact_sign_flip_p": float(
            np.mean(null_values >= abs(mean) - 1.0e-15)
        ),
    }


def collect_run(run_dir: Path, seed: int, variant: str) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    config = load_yaml(run_dir / "resolved_config.yaml")
    configured_epochs = int(config["training"]["epochs"])
    if len(history) != configured_epochs:
        raise ValueError(
            f"Incomplete fixed-budget run at {run_dir}: "
            f"found {len(history)} of {configured_epochs} epochs"
        )
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    checkpoint_selection = checkpoint["checkpoint_selection"]
    correct_rows = read_csv(run_dir / "pid_correct_id_closure_mae.csv")
    tv_rows = read_csv(run_dir / "pid_bin_closure_summary.csv")
    closure_rows = read_csv(run_dir / "closure_metrics.csv")
    row: dict[str, Any] = {
        "seed": seed,
        "variant": variant,
        "test_pid_cross_entropy": float(metrics["test"]["pid_cross_entropy"]),
        "test_pid_accuracy": float(metrics["test"]["pid_accuracy"]),
        "selected_epoch": int(checkpoint["best_epoch"]),
        "min_total_loss_epoch": int(
            checkpoint_selection["candidates"]["total_loss"]["epoch"]
        ),
        "min_pid_cross_entropy_epoch": int(
            checkpoint_selection["candidates"]["pid_cross_entropy"]["epoch"]
        ),
        "feature_count": len(checkpoint["feature_names"]),
        "target_count": len(checkpoint["target_names"]),
        "trained_epochs": len(history),
    }

    correct_by_species = {int(value["generated_pid"]): value for value in correct_rows}
    for species in GENERATED_SPECIES:
        key = SPECIES_KEY[species]
        value = correct_by_species[species]
        row[f"correct_mae_unweighted_{key}"] = float(
            value["correct_id_mae_unweighted"]
        )
        row[f"correct_mae_weighted_{key}"] = float(
            value["correct_id_mae_particle_weighted"]
        )

        species_tv = [value for value in tv_rows if int(value["generated_pid"]) == species]
        tv_values = np.asarray(
            [float(value["total_variation_distance"]) for value in species_tv]
        )
        counts = np.asarray([int(value["n"]) for value in species_tv])
        row[f"weighted_bin_tv_{key}"] = float(np.average(tv_values, weights=counts))

    row["macro_correct_mae_unweighted"] = float(
        np.mean([row[f"correct_mae_unweighted_{SPECIES_KEY[s]}"] for s in GENERATED_SPECIES])
    )
    row["macro_correct_mae_weighted"] = float(
        np.mean([row[f"correct_mae_weighted_{SPECIES_KEY[s]}"] for s in GENERATED_SPECIES])
    )
    row["macro_weighted_bin_tv"] = float(
        np.mean([row[f"weighted_bin_tv_{SPECIES_KEY[s]}"] for s in GENERATED_SPECIES])
    )

    for target in ("delta_p", "delta_theta", "delta_phi"):
        values = [
            float(value["wasserstein_1d"])
            for value in closure_rows
            if value["target"] == target
        ]
        row[f"mean_w1_{target}"] = float(np.mean(values))
    beta_path = run_dir / "beta_closure_overall.csv"
    row["mean_beta_w1"] = (
        float(np.mean([float(value["wasserstein_1d"]) for value in read_csv(beta_path)]))
        if beta_path.is_file()
        else ""
    )
    return row


def aggregate_conditions(per_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "macro_weighted_bin_tv",
        "macro_correct_mae_unweighted",
        "macro_correct_mae_weighted",
        "test_pid_cross_entropy",
        "test_pid_accuracy",
        "mean_w1_delta_p",
        "mean_w1_delta_theta",
        "mean_w1_delta_phi",
    )
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        selected = [row for row in per_run if row["variant"] == variant]
        output: dict[str, Any] = {"variant": variant, "n_seeds": len(selected)}
        for metric in metrics:
            mean, sd = mean_sd(np.asarray([row[metric] for row in selected]))
            output[f"{metric}_mean"] = mean
            output[f"{metric}_sd"] = sd
        beta_values = [float(row["mean_beta_w1"]) for row in selected if row["mean_beta_w1"] != ""]
        output["mean_beta_w1_mean"] = float(np.mean(beta_values)) if beta_values else ""
        output["mean_beta_w1_sd"] = float(np.std(beta_values, ddof=1)) if beta_values else ""
        rows.append(output)
    return rows


def analyze_contrasts(per_run: list[dict[str, Any]], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    metrics = [
        "macro_weighted_bin_tv",
        "macro_correct_mae_unweighted",
        "macro_correct_mae_weighted",
        "test_pid_cross_entropy",
        *[
            f"correct_mae_unweighted_{SPECIES_KEY[species]}"
            for species in GENERATED_SPECIES
        ],
    ]
    indexed = {(int(row["seed"]), row["variant"]): row for row in per_run}
    rows: list[dict[str, Any]] = []
    for contrast, control, treatment in CONTRASTS:
        for metric in metrics:
            left = np.asarray([indexed[(seed, control)][metric] for seed in seeds])
            right = np.asarray([indexed[(seed, treatment)][metric] for seed in seeds])
            left_mean, left_sd = mean_sd(left)
            right_mean, right_sd = mean_sd(right)
            rows.append(
                {
                    "contrast": contrast,
                    "control": control,
                    "treatment": treatment,
                    "metric": metric,
                    "control_mean": left_mean,
                    "control_sd": left_sd,
                    "treatment_mean": right_mean,
                    "treatment_sd": right_sd,
                    **paired_statistics(left, right),
                }
            )
    for metric in metrics:
        effect_without_target = np.asarray(
            [
                indexed[(seed, "A_original")][metric]
                - indexed[(seed, "B_beta_input")][metric]
                for seed in seeds
            ]
        )
        effect_with_target = np.asarray(
            [
                indexed[(seed, "C_beta_target")][metric]
                - indexed[(seed, "D_input_target_pid02")][metric]
                for seed in seeds
            ]
        )
        rows.append(
            {
                "contrast": "factorial_interaction_beta_input_by_beta_target",
                "control": "input_effect_with_target",
                "treatment": "input_effect_without_target",
                "metric": metric,
                "control_mean": float(effect_with_target.mean()),
                "control_sd": float(effect_with_target.std(ddof=1)),
                "treatment_mean": float(effect_without_target.mean()),
                "treatment_sd": float(effect_without_target.std(ddof=1)),
                # Positive means beta_gen helps more when delta_beta is a target.
                **paired_statistics(effect_with_target, effect_without_target),
            }
        )
    return rows


def plot_correct_id_curves(
    run_root: Path, seeds: tuple[int, ...], output_path: Path
) -> None:
    records: dict[tuple[str, int, int], dict[str, list[float]]] = {}
    for seed in seeds:
        for variant in VARIANTS:
            rows = read_csv(run_root / f"seed_{seed}" / variant / "pid_response_fixed_bins.csv")
            for row in rows:
                species = int(row["generated_pid"])
                if row["reconstructed_pid"] != str(species):
                    continue
                key = (variant, species, int(row["bin_index"]))
                values = records.setdefault(
                    key,
                    {"low": [], "high": [], "coatjava": [], "fm": []},
                )
                values["low"].append(float(row["p_low_gev"]))
                values["high"].append(float(row["p_high_gev"]))
                values["coatjava"].append(float(row["coatjava_fraction"]))
                values["fm"].append(float(row["fm_mean_probability"]))

    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(VARIANTS)))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for axis, species in zip(axes, GENERATED_SPECIES):
        bin_indices = sorted(
            key[2] for key in records if key[0] == VARIANTS[0] and key[1] == species
        )
        centers = np.asarray(
            [
                0.5
                * (
                    np.mean(records[(VARIANTS[0], species, index)]["low"])
                    + np.mean(records[(VARIANTS[0], species, index)]["high"])
                )
                for index in bin_indices
            ]
        )
        coatjava = np.asarray(
            [np.mean(records[(VARIANTS[0], species, index)]["coatjava"]) for index in bin_indices]
        )
        axis.plot(centers, coatjava, "ko--", linewidth=1.8, label="COATJAVA")
        for color, variant in zip(colors, VARIANTS):
            matrix = np.asarray(
                [records[(variant, species, index)]["fm"] for index in bin_indices]
            )
            mean = matrix.mean(axis=1)
            half_width = T_CRITICAL_95[len(seeds) - 1] * matrix.std(axis=1, ddof=1) / np.sqrt(len(seeds))
            axis.plot(centers, mean, marker="o", linewidth=1.4, color=color, label=VARIANT_LABELS[variant])
            axis.fill_between(centers, mean - half_width, mean + half_width, color=color, alpha=0.12)
        axis.set_title(f"generated {SPECIES_LABEL[species]}")
        axis.set_xlabel(r"$p_{gen}$ [GeV]")
        axis.grid(alpha=0.25)
        axis.set_ylim(0.0, 1.03)
    axes[0].set_ylabel(r"$P(s_{rec}=s_{gen}\mid s_{gen},p_{gen})$")
    axes[-1].legend(fontsize=7, loc="best")
    figure.suptitle("Correct-ID PID closure: ten matched seeds; bands are 95% seed intervals")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_reference_style_mae_bars(
    per_run: list[dict[str, Any]], output_path: Path
) -> None:
    """Show the Email-3 A/B/D sequence independently at both PID weights."""
    rows = (
        (
            r"$\lambda_{PID}=0.2$",
            ("A_original", "B_beta_input", "D_input_target_pid02"),
        ),
        (
            r"$\lambda_{PID}=1.0$",
            ("A_original_pid1", "B_beta_input_pid1", "D_input_target_pid1"),
        ),
    )
    labels = ("A: original", r"B: $\beta_{gen}$ input", r"D: input + $\Delta\beta$ target")
    colors = ("tab:blue", "tab:orange", "tab:green")
    x = np.arange(len(GENERATED_SPECIES))
    width = 0.24
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
    for axis, (title, variants) in zip(axes, rows):
        for index, (variant, label, color) in enumerate(zip(variants, labels, colors)):
            means = []
            standard_deviations = []
            for species in GENERATED_SPECIES:
                metric = f"correct_mae_unweighted_{SPECIES_KEY[species]}"
                values = np.asarray(
                    [100.0 * row[metric] for row in per_run if row["variant"] == variant]
                )
                means.append(values.mean())
                standard_deviations.append(values.std(ddof=1))
            axis.bar(
                x + (index - 1) * width,
                means,
                width,
                yerr=standard_deviations,
                capsize=3,
                color=color,
                label=label,
            )
        axis.set_xticks(x, (r"$\pi^-$", r"$\pi^+$", "proton"))
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Correct-ID closure MAE [percentage points]")
    axes[1].legend(fontsize=8)
    figure.suptitle("Email-3 beta ablation: ten paired seeds")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_reference_style_pid_rows(
    run_root: Path, seeds: tuple[int, ...], output_path: Path
) -> None:
    """Plot the A/B/D closure sequence in one row per PID-loss weight."""
    row_variants = (
        ("A_original", "B_beta_input", "D_input_target_pid02"),
        ("A_original_pid1", "B_beta_input_pid1", "D_input_target_pid1"),
    )
    variants = tuple(dict.fromkeys(variant for row in row_variants for variant in row))
    records: dict[tuple[str, int, int], dict[str, list[float]]] = {}
    for seed in seeds:
        for variant in variants:
            rows = read_csv(
                run_root / f"seed_{seed}" / variant / "pid_response_fixed_bins.csv"
            )
            for row in rows:
                species = int(row["generated_pid"])
                if row["reconstructed_pid"] != str(species):
                    continue
                key = (variant, species, int(row["bin_index"]))
                values = records.setdefault(
                    key,
                    {"coatjava": [], "fm": [], "low": [], "high": []},
                )
                values["coatjava"].append(float(row["coatjava_fraction"]))
                values["fm"].append(float(row["fm_mean_probability"]))
                values["low"].append(float(row["p_low_gev"]))
                values["high"].append(float(row["p_high_gev"]))

    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharey=True)
    for column, species in enumerate(GENERATED_SPECIES):
        bins = sorted(
            key[2]
            for key in records
            if key[0] == "A_original" and key[1] == species
        )
        x = np.arange(len(bins))
        bin_labels = [
            f"{np.mean(records[('A_original', species, b)]['low']):g}–"
            f"{np.mean(records[('A_original', species, b)]['high']):g}"
            for b in bins
        ]
        coatjava = np.asarray(
            [
                np.mean(records[("A_original", species, b)]["coatjava"])
                for b in bins
            ]
        )
        for row_index, axis in enumerate(axes[:, column]):
            axis.plot(x, coatjava, "o-", color="black", label="COATJAVA")
            styles = {
                "A_original": ("tab:blue", "x", "FM A: original"),
                "B_beta_input": ("tab:orange", "s", r"FM B: $\beta_{gen}$ input"),
                "D_input_target_pid02": (
                    "tab:green", "^", r"FM D: input + $\Delta\beta$ target",
                ),
                "A_original_pid1": ("tab:blue", "x", "FM A: original"),
                "B_beta_input_pid1": (
                    "tab:orange", "s", r"FM B: $\beta_{gen}$ input",
                ),
                "D_input_target_pid1": (
                    "tab:green", "^", r"FM D: input + $\Delta\beta$ target",
                ),
            }
            for variant in row_variants[row_index]:
                matrix = np.asarray(
                    [records[(variant, species, b)]["fm"] for b in bins]
                )
                mean = matrix.mean(axis=1)
                half_width = (
                    T_CRITICAL_95[len(seeds) - 1]
                    * matrix.std(axis=1, ddof=1)
                    / np.sqrt(len(seeds))
                )
                color, marker, label = styles[variant]
                axis.plot(x, mean, marker=marker, color=color, label=label)
                axis.fill_between(
                    x,
                    mean - half_width,
                    mean + half_width,
                    color=color,
                    alpha=0.15,
                )
            axis.set_xticks(x, bin_labels, rotation=35)
            axis.set_ylim(0.0, 1.03)
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel(
                    "$\\lambda_{PID}=0.2$\nCorrect-ID response"
                    if row_index == 0 else "$\\lambda_{PID}=1.0$\nCorrect-ID response"
                )
            if row_index == 1:
                axis.set_xlabel(r"Generated momentum $p_{gen}$ [GeV]")
        axes[0, column].set_title(f"Generated {SPECIES_LABEL[species]}")
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[1, 0].legend(fontsize=8, loc="best")
    figure.suptitle(
        "Email-3 PID closure — equal 1-GeV bins, ten-seed means"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_seed_metrics(per_run: list[dict[str, Any]], output_path: Path) -> None:
    indexed = {(int(row["seed"]), row["variant"]): row for row in per_run}
    seeds = sorted({int(row["seed"]) for row in per_run})
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    specifications = (
        ("macro_weighted_bin_tv", "Macro weighted-bin TV"),
        ("macro_correct_mae_unweighted", "Macro correct-ID MAE"),
    )
    x = np.arange(len(VARIANTS))
    for axis, (metric, title) in zip(axes, specifications):
        for seed in seeds:
            values = [indexed[(seed, variant)][metric] for variant in VARIANTS]
            axis.plot(x, values, color="0.75", linewidth=0.8, alpha=0.75)
            axis.scatter(x, values, color="0.55", s=12, alpha=0.75)
        matrix = np.asarray(
            [[indexed[(seed, variant)][metric] for variant in VARIANTS] for seed in seeds]
        )
        means = matrix.mean(axis=0)
        half_width = T_CRITICAL_95[len(seeds) - 1] * matrix.std(axis=0, ddof=1) / np.sqrt(len(seeds))
        axis.errorbar(x, means, yerr=half_width, fmt="ko", capsize=4, label="mean ± 95% CI")
        axis.set_xticks(
            x,
            ["A02", "B02", "C02", "D02", "A10", "B10", "D10"],
            rotation=25,
        )
        axis.set_title(title)
        axis.set_ylabel("lower is better")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("PID closure across matched model seeds")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_beta_response_curves(
    run_root: Path, seeds: tuple[int, ...], output_path: Path
) -> None:
    """Compare the continuous beta response for the three target-beta models."""
    variants = (
        "C_beta_target",
        "D_input_target_pid02",
        "D_input_target_pid1",
    )
    records: dict[tuple[str, int, int], dict[str, list[float]]] = {}
    for seed in seeds:
        for variant in variants:
            rows = read_csv(
                run_root / f"seed_{seed}" / variant / "beta_closure_vs_gen_p.csv"
            )
            for row in rows:
                key = (variant, int(row["generated_pid"]), int(row["bin_index"]))
                values = records.setdefault(
                    key,
                    {"low": [], "high": [], "observed": [], "sampled": []},
                )
                values["low"].append(float(row["p_low_gev"]))
                values["high"].append(float(row["p_high_gev"]))
                values["observed"].append(float(row["observed_mean"]))
                values["sampled"].append(float(row["sampled_mean"]))

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(variants)))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, species in zip(axes, GENERATED_SPECIES):
        bins = sorted(
            key[2] for key in records if key[0] == variants[0] and key[1] == species
        )
        centers = np.asarray(
            [
                0.5
                * (
                    np.mean(records[(variants[0], species, index)]["low"])
                    + np.mean(records[(variants[0], species, index)]["high"])
                )
                for index in bins
            ]
        )
        observed = np.asarray(
            [np.mean(records[(variants[0], species, index)]["observed"]) for index in bins]
        )
        axis.plot(centers, observed, "ko--", linewidth=1.8, label="COATJAVA")
        for color, variant in zip(colors, variants):
            matrix = np.asarray(
                [records[(variant, species, index)]["sampled"] for index in bins]
            )
            mean = matrix.mean(axis=1)
            half_width = (
                T_CRITICAL_95[len(seeds) - 1]
                * matrix.std(axis=1, ddof=1)
                / np.sqrt(len(seeds))
            )
            axis.plot(
                centers,
                mean,
                marker="o",
                linewidth=1.4,
                color=color,
                label=VARIANT_LABELS[variant],
            )
            axis.fill_between(
                centers,
                mean - half_width,
                mean + half_width,
                color=color,
                alpha=0.15,
            )
        axis.set_title(f"generated {SPECIES_LABEL[species]}")
        axis.set_xlabel(r"$p_{gen}$ [GeV]")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"mean reconstructed $\beta$")
    axes[-1].legend(fontsize=7, loc="best")
    figure.suptitle("Continuous beta-response closure: 95% intervals across seeds")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_pid_total_variation(
    run_root: Path, seeds: tuple[int, ...], output_path: Path
) -> None:
    records: dict[tuple[str, int, int], dict[str, list[float]]] = {}
    for seed in seeds:
        for variant in VARIANTS:
            rows = read_csv(
                run_root / f"seed_{seed}" / variant / "pid_bin_closure_summary.csv"
            )
            for row in rows:
                key = (variant, int(row["generated_pid"]), int(row["bin_index"]))
                values = records.setdefault(
                    key, {"low": [], "high": [], "tv": []}
                )
                values["low"].append(float(row["p_low_gev"]))
                values["high"].append(float(row["p_high_gev"]))
                values["tv"].append(float(row["total_variation_distance"]))

    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(VARIANTS)))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, species in zip(axes, GENERATED_SPECIES):
        bins = sorted(
            key[2] for key in records if key[0] == VARIANTS[0] and key[1] == species
        )
        centers = np.asarray(
            [
                0.5
                * (
                    np.mean(records[(VARIANTS[0], species, index)]["low"])
                    + np.mean(records[(VARIANTS[0], species, index)]["high"])
                )
                for index in bins
            ]
        )
        for color, variant in zip(colors, VARIANTS):
            matrix = np.asarray(
                [records[(variant, species, index)]["tv"] for index in bins]
            )
            mean = matrix.mean(axis=1)
            half_width = (
                T_CRITICAL_95[len(seeds) - 1]
                * matrix.std(axis=1, ddof=1)
                / np.sqrt(len(seeds))
            )
            axis.plot(
                centers,
                mean,
                marker="o",
                linewidth=1.4,
                color=color,
                label=VARIANT_LABELS[variant],
            )
            axis.fill_between(
                centers,
                mean - half_width,
                mean + half_width,
                color=color,
                alpha=0.12,
            )
        axis.set_title(f"generated {SPECIES_LABEL[species]}")
        axis.set_xlabel(r"$p_{gen}$ [GeV]")
        axis.grid(alpha=0.25)
        axis.set_ylim(bottom=0.0)
    axes[0].set_ylabel("PID-response total variation")
    axes[-1].legend(fontsize=7, loc="best")
    figure.suptitle("Full PID-distribution closure versus generated momentum")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_pid_migration_channels(
    run_root: Path, seeds: tuple[int, ...], output_path: Path
) -> None:
    channels = (
        (211, 2212, r"$\pi^+\rightarrow p$"),
        (2212, 211, r"$p\rightarrow\pi^+$"),
        (211, 321, r"$\pi^+\rightarrow K^+$"),
        (2212, 321, r"$p\rightarrow K^+$"),
    )
    records: dict[tuple[str, int, int, int], dict[str, list[float]]] = {}
    for seed in seeds:
        for variant in VARIANTS:
            rows = read_csv(
                run_root / f"seed_{seed}" / variant / "pid_response_fixed_bins.csv"
            )
            for row in rows:
                generated = int(row["generated_pid"])
                reconstructed = row["reconstructed_pid"]
                if not reconstructed.lstrip("-").isdigit():
                    continue
                key = (
                    variant,
                    generated,
                    int(reconstructed),
                    int(row["bin_index"]),
                )
                values = records.setdefault(
                    key,
                    {"low": [], "high": [], "coatjava": [], "fm": []},
                )
                values["low"].append(float(row["p_low_gev"]))
                values["high"].append(float(row["p_high_gev"]))
                values["coatjava"].append(float(row["coatjava_fraction"]))
                values["fm"].append(float(row["fm_mean_probability"]))

    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(VARIANTS)))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, (generated, reconstructed, title) in zip(axes.flat, channels):
        bins = sorted(
            key[3]
            for key in records
            if key[:3] == (VARIANTS[0], generated, reconstructed)
        )
        centers = np.asarray(
            [
                0.5
                * (
                    np.mean(records[(VARIANTS[0], generated, reconstructed, index)]["low"])
                    + np.mean(records[(VARIANTS[0], generated, reconstructed, index)]["high"])
                )
                for index in bins
            ]
        )
        observed = np.asarray(
            [
                np.mean(records[(VARIANTS[0], generated, reconstructed, index)]["coatjava"])
                for index in bins
            ]
        )
        axis.plot(centers, observed, "ko--", linewidth=1.8, label="COATJAVA")
        for color, variant in zip(colors, VARIANTS):
            matrix = np.asarray(
                [records[(variant, generated, reconstructed, index)]["fm"] for index in bins]
            )
            mean = matrix.mean(axis=1)
            half_width = (
                T_CRITICAL_95[len(seeds) - 1]
                * matrix.std(axis=1, ddof=1)
                / np.sqrt(len(seeds))
            )
            axis.plot(centers, mean, marker="o", linewidth=1.3, color=color, label=VARIANT_LABELS[variant])
            axis.fill_between(centers, mean - half_width, mean + half_width, color=color, alpha=0.12)
        axis.set_title(title)
        axis.set_xlabel(r"$p_{gen}$ [GeV]")
        axis.set_ylabel("response probability")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=7, loc="best")
    figure.suptitle("Selected PID migration-channel closure")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_report(
    output_path: Path,
    aggregate: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    per_run: list[dict[str, Any]],
) -> None:
    aggregate_by_variant = {row["variant"]: row for row in aggregate}
    contrast_rows = [
        row
        for row in contrasts
        if row["metric"] in ("macro_weighted_bin_tv", "macro_correct_mae_unweighted")
    ]
    lines = [
        "# Physics-informed beta-gen factorial study",
        "",
        "All conditions use the same beta-valid teacher rows and event-disjoint split. "
        "The locked test checkpoint minimizes validation PID cross-entropy.",
        "",
        "## Condition summary",
        "",
        "| Condition | Macro TV | Correct-ID MAE | Test PID CE | Test PID accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = aggregate_by_variant[variant]
        lines.append(
            f"| {variant} | {row['macro_weighted_bin_tv_mean']:.5f} ± {row['macro_weighted_bin_tv_sd']:.5f} "
            f"| {row['macro_correct_mae_unweighted_mean']:.5f} ± {row['macro_correct_mae_unweighted_sd']:.5f} "
            f"| {row['test_pid_cross_entropy_mean']:.5f} ± {row['test_pid_cross_entropy_sd']:.5f} "
            f"| {row['test_pid_accuracy_mean']:.4f} ± {row['test_pid_accuracy_sd']:.4f} |"
        )

    contrast_by_key = {
        (row["contrast"], row["metric"]): row for row in contrasts
    }
    input_pid02 = contrast_by_key[
        ("beta_input_without_target", "macro_weighted_bin_tv")
    ]
    input_pid1 = contrast_by_key[
        ("beta_input_pid1", "macro_weighted_bin_tv")
    ]
    target_pid02 = contrast_by_key[
        ("delta_beta_target_with_input", "macro_weighted_bin_tv")
    ]
    target_pid1 = contrast_by_key[
        ("delta_beta_target_pid1", "macro_weighted_bin_tv")
    ]
    pid_weight_effects = {
        variant: contrast_by_key[(contrast, "macro_weighted_bin_tv")]
        for variant, contrast in (
            ("A", "pid_weight_original"),
            ("B", "pid_weight_beta_input"),
            ("D", "pid_weight_combined"),
        )
    }
    best_variant = min(
        aggregate,
        key=lambda row: float(row["macro_weighted_bin_tv_mean"]),
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The sequential Email-3 contrasts are $A\\to B$ (add "
            "$\\beta_{\\rm gen}$ input) and $B\\to D$ (also learn "
            "$\\Delta\\beta$). For macro TV:",
            "",
            f"- $A\\to B$, $\\lambda=0.2$: {input_pid02['mean_improvement']:+.5f} "
            f"([{input_pid02['ci95_low']:+.5f}, {input_pid02['ci95_high']:+.5f}]);",
            f"- $B\\to D$, $\\lambda=0.2$: {target_pid02['mean_improvement']:+.5f} "
            f"([{target_pid02['ci95_low']:+.5f}, {target_pid02['ci95_high']:+.5f}]);",
            f"- $A\\to B$, $\\lambda=1.0$: {input_pid1['mean_improvement']:+.5f} "
            f"([{input_pid1['ci95_low']:+.5f}, {input_pid1['ci95_high']:+.5f}]);",
            f"- $B\\to D$, $\\lambda=1.0$: {target_pid1['mean_improvement']:+.5f} "
            f"([{target_pid1['ci95_low']:+.5f}, {target_pid1['ci95_high']:+.5f}]).",
            "",
            "The paired effect of increasing $\\lambda_{\\rm PID}:0.2\\to1.0$ is:",
            "",
            *[
                f"- {variant}: {row['mean_improvement']:+.5f} "
                f"([{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}]), "
                f"{row['favorable_pairs']}/{row['n_pairs']} favorable pairs."
                for variant, row in pid_weight_effects.items()
            ],
            "",
            f"The smallest mean macro TV is {best_variant['variant']} at "
            f"{best_variant['macro_weighted_bin_tv_mean']:.5f}. Claims about either "
            "beta coordinate use paired intervals above; test-set ranking is descriptive, "
            "not a checkpoint-selection rule.",
        ]
    )
    lines.extend(
        [
            "",
            "## Paired primary contrasts",
            "",
            "Positive improvement favors the treatment because both metrics are lower-is-better.",
            "",
            "| Contrast | Metric | Mean improvement [95% CI] | Median | Favorable pairs | Exact p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in contrast_rows:
        lines.append(
            f"| {row['contrast']} | {row['metric']} | {row['mean_improvement']:.5f} "
            f"[{row['ci95_low']:.5f}, {row['ci95_high']:.5f}] | "
            f"{row['median_improvement']:.5f} | {row['favorable_pairs']}/{row['n_pairs']} "
            f"| {row['exact_sign_flip_p']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Dr. Joo-aligned correct-ID MAE",
            "",
            "Values are unweighted momentum-bin MAE in percent, averaged over seeds.",
            "",
            "| Species | A02 | B02 | C02 diagnostic | D02 | A10 | B10 | D10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for species in GENERATED_SPECIES:
        metric = f"correct_mae_unweighted_{SPECIES_KEY[species]}"
        cells = []
        for variant in VARIANTS:
            values = np.asarray(
                [row[metric] for row in per_run if row["variant"] == variant]
            )
            cells.append(f"{100.0 * values.mean():.2f} ± {100.0 * values.std(ddof=1):.2f}%")
        lines.append(f"| {SPECIES_LABEL[species]} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Continuous beta response",
            "",
            "| Condition | Mean beta W1 across species |",
            "|---|---:|",
        ]
    )
    for variant in ("C_beta_target", "D_input_target_pid02", "D_input_target_pid1"):
        row = aggregate_by_variant[variant]
        lines.append(
            f"| {variant} | {row['mean_beta_w1_mean']:.5f} ± {row['mean_beta_w1_sd']:.5f} |"
        )

    lines.extend(
        [
            "",
            "![Momentum-dependent correct-ID closure](pid_correct_id_vs_gen_p_factorial.png)",
            "",
            "![Collaborator-style correct-ID MAE comparison](pid_closure_mae_comparison_our_10seed.png)",
            "",
            "![Collaborator-style momentum comparison](pid_closure_beta_comparison_our_10seed.png)",
            "",
            "![Seed-to-seed PID closure](pid_closure_across_conditions.png)",
            "",
            "![Full PID-distribution closure](pid_total_variation_vs_gen_p_factorial.png)",
            "",
            "![Selected PID migration channels](pid_migration_channels_factorial.png)",
            "",
            "![Continuous beta-response closure](beta_response_vs_gen_p_factorial.png)",
            "",
            "Machine-readable results: `per_run_metrics.csv`, `condition_summary.csv`, "
            "and `paired_contrasts.csv`.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run = [
        collect_run(run_root / f"seed_{seed}" / variant, seed, variant)
        for seed in seeds
        for variant in VARIANTS
    ]
    aggregate = aggregate_conditions(per_run)
    contrasts = analyze_contrasts(per_run, seeds)
    write_csv(output_dir / "per_run_metrics.csv", per_run)
    write_csv(output_dir / "condition_summary.csv", aggregate)
    write_csv(output_dir / "paired_contrasts.csv", contrasts)
    plot_correct_id_curves(
        run_root, seeds, output_dir / "pid_correct_id_vs_gen_p_factorial.png"
    )
    plot_reference_style_mae_bars(
        per_run, output_dir / "pid_closure_mae_comparison_our_10seed.png"
    )
    plot_reference_style_pid_rows(
        run_root, seeds, output_dir / "pid_closure_beta_comparison_our_10seed.png"
    )
    plot_seed_metrics(per_run, output_dir / "pid_closure_across_conditions.png")
    plot_pid_total_variation(
        run_root, seeds, output_dir / "pid_total_variation_vs_gen_p_factorial.png"
    )
    plot_pid_migration_channels(
        run_root, seeds, output_dir / "pid_migration_channels_factorial.png"
    )
    plot_beta_response_curves(
        run_root, seeds, output_dir / "beta_response_vs_gen_p_factorial.png"
    )
    write_report(
        output_dir / "BETA_GEN_FACTORIAL_REPORT.md", aggregate, contrasts, per_run
    )
    print(f"wrote factorial summary to {output_dir}")


if __name__ == "__main__":
    main()
