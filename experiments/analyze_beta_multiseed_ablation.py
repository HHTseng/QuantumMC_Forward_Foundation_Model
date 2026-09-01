#!/usr/bin/env python3
"""Aggregate and visualize the controlled ten-pair beta auxiliary-task study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = tuple(range(20260822, 20260832))
VARIANTS = ("no_beta", "joint_beta")
GENERATED_SPECIES = (-211, 211, 2212)
SPECIES_KEY = {-211: "pi_minus", 211: "pi_plus", 2212: "proton"}
SPECIES_TEXT = {-211: "pi-", 211: "pi+", 2212: "proton"}
SPECIES_MATH = {-211: r"$\pi^-$", 211: r"$\pi^+$", 2212: "proton"}
T_CRITICAL_95_DF9 = 2.2621571627409915


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default=str(REPOSITORY_ROOT / "runs/gpu_beta_multiseed_ablation"),
    )
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def mean_std(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def paired_statistics(
    no_beta_values: np.ndarray,
    joint_beta_values: np.ndarray,
    lower_is_better: bool,
) -> dict[str, float | int]:
    """Return paired improvement, t interval, exact sign-flip p, and Cohen dz."""
    no_beta_values = np.asarray(no_beta_values, dtype=np.float64)
    joint_beta_values = np.asarray(joint_beta_values, dtype=np.float64)
    if no_beta_values.shape != joint_beta_values.shape or len(no_beta_values) != 10:
        raise ValueError("Paired statistics require two aligned vectors of length ten")
    improvement = (
        no_beta_values - joint_beta_values
        if lower_is_better
        else joint_beta_values - no_beta_values
    )
    n = len(improvement)
    mean_improvement = float(np.mean(improvement))
    sd_improvement = float(np.std(improvement, ddof=1))
    half_width = T_CRITICAL_95_DF9 * sd_improvement / np.sqrt(n)
    # Exact paired randomization under exchangeability: all 2^10 sign flips.
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)))
    null_statistics = np.abs(np.mean(signs * improvement[None, :], axis=1))
    exact_p = float(
        np.mean(null_statistics >= abs(mean_improvement) - 1.0e-15)
    )
    return {
        "mean_paired_improvement": mean_improvement,
        "median_paired_improvement": float(np.median(improvement)),
        "paired_improvement_sd": sd_improvement,
        "paired_improvement_ci95_low": mean_improvement - half_width,
        "paired_improvement_ci95_high": mean_improvement + half_width,
        "exact_two_sided_sign_flip_p": exact_p,
        "paired_cohen_dz": (
            mean_improvement / sd_improvement
            if sd_improvement > 0.0
            else float("inf")
        ),
        "joint_beta_better_pairs": int(np.sum(improvement > 0.0)),
        "no_beta_better_pairs": int(np.sum(improvement < 0.0)),
        "tied_pairs": int(np.sum(improvement == 0.0)),
    }


def weighted_species_tv(summary_rows: list[dict[str, Any]], species: int) -> float:
    selected = [row for row in summary_rows if int(row["generated_pid"]) == species]
    weights = np.asarray([row["n"] for row in selected], dtype=np.float64)
    values = np.asarray(
        [row["total_variation_distance"] for row in selected], dtype=np.float64
    )
    return float(np.average(values, weights=weights))


def run_metrics(run_dir: Path, seed: int, variant: str) -> tuple[dict[str, Any], dict[str, Any]]:
    required = (
        "metrics.json",
        "data_audit.json",
        "history.json",
        "model.pt",
        "pid_response_fixed_bins.csv",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete run {run_dir}: {missing}")
    metrics = read_json(run_dir / "metrics.json")
    audit = read_json(run_dir / "data_audit.json")
    history = read_json(run_dir / "history.json")
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    if int(checkpoint["seed"]) != seed:
        raise AssertionError(f"Checkpoint seed mismatch in {run_dir}")
    expected_dim = 3 if variant == "no_beta" else 4
    if int(checkpoint["architecture"]["target_dim"]) != expected_dim:
        raise AssertionError(f"Unexpected target dimension in {run_dir}")
    bin_summary = metrics["pid_conditional_closure"]["bin_summary"]
    integrated = {
        int(row["generated_pid"]): row
        for row in metrics["pid_conditional_closure"]["integrated_correct_id"]
    }
    best_epoch = int(checkpoint["best_epoch"])
    selected_history = next(
        item for item in history if int(item["epoch"]) == best_epoch
    )
    best_pid_history = min(
        history, key=lambda item: float(item["validation"]["pid_cross_entropy"])
    )
    row: dict[str, Any] = {
        "seed": seed,
        "variant": variant,
        "best_epoch": best_epoch,
        "selected_validation_pid_cross_entropy": float(
            selected_history["validation"]["pid_cross_entropy"]
        ),
        "selected_validation_pid_accuracy": float(
            selected_history["validation"]["pid_accuracy"]
        ),
        "best_validation_pid_epoch": int(best_pid_history["epoch"]),
        "best_validation_pid_cross_entropy": float(
            best_pid_history["validation"]["pid_cross_entropy"]
        ),
        "best_validation_pid_accuracy": float(
            best_pid_history["validation"]["pid_accuracy"]
        ),
        "target_dim": expected_dim,
        "parameter_count": int(
            sum(value.numel() for value in checkpoint["model_state"].values())
        ),
        "test_rows": int(sum(audit["sampled_counts"]["test"].values())),
        "test_pid_cross_entropy": float(metrics["test"]["pid_cross_entropy"]),
        "test_pid_accuracy": float(metrics["test"]["pid_accuracy"]),
        "worst_bin_tv": float(
            max(item["total_variation_distance"] for item in bin_summary)
        ),
        "checkpoint_sha256": sha256_file(run_dir / "model.pt"),
        "dataset_metadata_sha256": audit["dataset_metadata_sha256"],
        "selection_sha256": sha256_text(audit["selection_sql"]),
        "data_split_seed": int(audit["data_split_seed"]),
        "data_order_seed": int(audit["data_order_seed"]),
    }
    weighted_tvs = []
    correct_errors = []
    for species in GENERATED_SPECIES:
        species_key = SPECIES_KEY[species]
        tv = weighted_species_tv(bin_summary, species)
        integrated_row = integrated[species]
        teacher = float(integrated_row["coatjava_correct_fraction"])
        prediction = float(integrated_row["fm_correct_mean_probability"])
        error = abs(prediction - teacher)
        row[f"weighted_bin_tv_{species_key}"] = tv
        row[f"correct_id_teacher_{species_key}"] = teacher
        row[f"correct_id_probability_{species_key}"] = prediction
        row[f"correct_id_abs_error_{species_key}"] = error
        weighted_tvs.append(tv)
        correct_errors.append(error)
    row["macro_weighted_bin_tv"] = float(np.mean(weighted_tvs))
    row["macro_correct_id_mae"] = float(np.mean(correct_errors))
    provenance = {
        "audit": audit,
        "checkpoint": checkpoint,
        "response_rows": read_csv(run_dir / "pid_response_fixed_bins.csv"),
    }
    return row, provenance


def validate_all_runs(
    rows: list[dict[str, Any]], provenance: dict[tuple[int, str], dict[str, Any]]
) -> dict[str, Any]:
    dataset_hashes = {row["dataset_metadata_sha256"] for row in rows}
    selection_hashes = {row["selection_sha256"] for row in rows}
    test_counts = {row["test_rows"] for row in rows}
    split_seeds = {row["data_split_seed"] for row in rows}
    order_seeds = {row["data_order_seed"] for row in rows}
    if not all(len(values) == 1 for values in (dataset_hashes, selection_hashes, test_counts, split_seeds, order_seeds)):
        raise AssertionError("Dataset, selection, or test population differs across runs")
    reference_teacher: dict[tuple[str, str, str], tuple[str, str]] | None = None
    for key, item in provenance.items():
        teacher = {
            (
                row["generated_pid"],
                row["bin_index"],
                row["reconstructed_pid"],
            ): (row["n"], row["coatjava_fraction"])
            for row in item["response_rows"]
        }
        if reference_teacher is None:
            reference_teacher = teacher
        elif teacher != reference_teacher:
            raise AssertionError(f"Teacher fixed-bin responses differ for run {key}")
    for seed in sorted({int(row["seed"]) for row in rows}):
        pair = [row for row in rows if int(row["seed"]) == seed]
        if {row["variant"] for row in pair} != set(VARIANTS):
            raise AssertionError(f"Seed {seed} does not have exactly one model pair")
    return {
        "dataset_metadata_sha256": next(iter(dataset_hashes)),
        "selection_sha256": next(iter(selection_hashes)),
        "test_rows": next(iter(test_counts)),
        "data_split_seed": next(iter(split_seeds)),
        "data_order_seed": next(iter(order_seeds)),
        "teacher_fixed_bin_rows_identical": True,
        "complete_pair_count": 10,
    }


METRICS = (
    ("macro_weighted_bin_tv", "Macro weighted-bin TV", True, "primary"),
    ("macro_correct_id_mae", "Macro correct-ID MAE", True, "primary"),
    ("worst_bin_tv", "Worst fixed-bin TV", True, "secondary"),
    ("test_pid_cross_entropy", "Test PID cross entropy", True, "secondary"),
    ("test_pid_accuracy", "Test top-1 PID accuracy", False, "secondary"),
    ("weighted_bin_tv_pi_minus", "pi- weighted-bin TV", True, "species"),
    ("weighted_bin_tv_pi_plus", "pi+ weighted-bin TV", True, "species"),
    ("weighted_bin_tv_proton", "proton weighted-bin TV", True, "species"),
    ("correct_id_abs_error_pi_minus", "pi- correct-ID absolute error", True, "species"),
    ("correct_id_abs_error_pi_plus", "pi+ correct-ID absolute error", True, "species"),
    ("correct_id_abs_error_proton", "proton correct-ID absolute error", True, "species"),
)


def aggregate_rows(run_rows: list[dict[str, Any]], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    by_key = {(int(row["seed"]), row["variant"]): row for row in run_rows}
    aggregate: list[dict[str, Any]] = []
    for metric, label, lower_is_better, family in METRICS:
        control = np.asarray([by_key[(seed, "no_beta")][metric] for seed in seeds])
        treatment = np.asarray([by_key[(seed, "joint_beta")][metric] for seed in seeds])
        control_mean, control_std = mean_std(control)
        treatment_mean, treatment_std = mean_std(treatment)
        statistics = paired_statistics(control, treatment, lower_is_better)
        aggregate.append(
            {
                "metric": metric,
                "label": label,
                "family": family,
                "lower_is_better": lower_is_better,
                "pair_count": len(seeds),
                "no_beta_mean": control_mean,
                "no_beta_sd": control_std,
                "joint_beta_mean": treatment_mean,
                "joint_beta_sd": treatment_std,
                "relative_mean_improvement_percent": (
                    100.0
                    * float(statistics["mean_paired_improvement"])
                    / control_mean
                    if control_mean != 0.0
                    else 0.0
                ),
                **statistics,
            }
        )
    return aggregate


def paired_wide_rows(
    run_rows: list[dict[str, Any]], seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    by_key = {(int(row["seed"]), row["variant"]): row for row in run_rows}
    output: list[dict[str, Any]] = []
    for seed in seeds:
        control = by_key[(seed, "no_beta")]
        treatment = by_key[(seed, "joint_beta")]
        row: dict[str, Any] = {
            "seed": seed,
            "no_beta_best_epoch": control["best_epoch"],
            "joint_beta_best_epoch": treatment["best_epoch"],
        }
        for metric, _, lower_is_better, _ in METRICS:
            row[f"no_beta_{metric}"] = control[metric]
            row[f"joint_beta_{metric}"] = treatment[metric]
            row[f"joint_beta_improvement_{metric}"] = (
                control[metric] - treatment[metric]
                if lower_is_better
                else treatment[metric] - control[metric]
            )
        output.append(row)
    return output


def correct_id_bin_rows(
    provenance: dict[tuple[int, str], dict[str, Any]], seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for species in GENERATED_SPECIES:
        available_bins = sorted(
            {
                int(item["bin_index"])
                for item in provenance[(seeds[0], "no_beta")]["response_rows"]
                if int(item["generated_pid"]) == species
                and item["reconstructed_pid"] == str(species)
            }
        )
        for bin_index in available_bins:
            values: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
            teacher_value: float | None = None
            n_value: int | None = None
            low_value: float | None = None
            high_value: float | None = None
            for seed in seeds:
                for variant in VARIANTS:
                    row = next(
                        item
                        for item in provenance[(seed, variant)]["response_rows"]
                        if int(item["generated_pid"]) == species
                        and int(item["bin_index"]) == bin_index
                        and item["reconstructed_pid"] == str(species)
                    )
                    values[variant].append(float(row["fm_mean_probability"]))
                    current_teacher = float(row["coatjava_fraction"])
                    if teacher_value is None:
                        teacher_value = current_teacher
                        n_value = int(row["n"])
                        low_value = float(row["p_low_gev"])
                        high_value = float(row["p_high_gev"])
                    elif current_teacher != teacher_value:
                        raise AssertionError("Fixed test teacher unexpectedly varies")
            control = np.asarray(values["no_beta"])
            treatment = np.asarray(values["joint_beta"])
            control_mean, control_sd = mean_std(control)
            treatment_mean, treatment_sd = mean_std(treatment)
            output.append(
                {
                    "generated_pid": species,
                    "generated_species": SPECIES_TEXT[species],
                    "bin_index": bin_index,
                    "p_low_gev": low_value,
                    "p_high_gev": high_value,
                    "n": n_value,
                    "coatjava_fraction": teacher_value,
                    "no_beta_mean_probability": control_mean,
                    "no_beta_sd_across_seeds": control_sd,
                    "joint_beta_mean_probability": treatment_mean,
                    "joint_beta_sd_across_seeds": treatment_sd,
                    **paired_statistics(
                        np.abs(control - teacher_value),
                        np.abs(treatment - teacher_value),
                        lower_is_better=True,
                    ),
                }
            )
    return output


def ci95_half_width(values: np.ndarray) -> float:
    return float(T_CRITICAL_95_DF9 * np.std(values, ddof=1) / np.sqrt(len(values)))


def plot_correct_id_multiseed(
    rows: list[dict[str, Any]], path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.7), sharey=True)
    for axis, species in zip(axes, GENERATED_SPECIES):
        selected = [row for row in rows if int(row["generated_pid"]) == species]
        x = np.asarray(
            [0.5 * (row["p_low_gev"] + row["p_high_gev"]) for row in selected]
        )
        teacher = np.asarray([row["coatjava_fraction"] for row in selected])
        axis.plot(x, teacher, "o-", color="tab:blue", label="COATJAVA")
        for key, color, marker, label in (
            ("no_beta", "tab:orange", "s", r"FM: no $\beta$ target"),
            ("joint_beta", "tab:green", "^", r"FM: joint $\Delta\beta$ target"),
        ):
            mean = np.asarray([row[f"{key}_mean_probability"] for row in selected])
            sd = np.asarray([row[f"{key}_sd_across_seeds"] for row in selected])
            half_width = T_CRITICAL_95_DF9 * sd / np.sqrt(10.0)
            axis.plot(x, mean, marker=marker, color=color, label=label)
            axis.fill_between(
                x, mean - half_width, mean + half_width, color=color, alpha=0.14
            )
        axis.set(
            title=f"generated {SPECIES_MATH[species]}",
            xlabel=r"generated momentum $p_{\mathrm{gen}}$ [GeV]",
            xlim=(0.0, 9.0),
            ylim=(0.0, 1.05),
            xticks=np.arange(0.5, 9.0, 1.0),
            xticklabels=[f"{index}–{index + 1}" for index in range(9)],
        )
        axis.tick_params(axis="x", labelrotation=45)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Correct reconstructed-PID response")
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Ten-paired-seed direct correct-ID closure\n"
        "lines are seed means; bands are 95% t intervals across training seeds"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_paired_panels(
    run_rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
    metrics: list[tuple[str, str]],
    shape: tuple[int, int],
    path: Path,
    title: str,
) -> None:
    by_key = {(int(row["seed"]), row["variant"]): row for row in run_rows}
    figure, axes = plt.subplots(*shape, figsize=(6.0 * shape[1], 5.0 * shape[0]))
    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    for axis, (metric, label) in zip(flat_axes, metrics):
        control = np.asarray([by_key[(seed, "no_beta")][metric] for seed in seeds])
        treatment = np.asarray(
            [by_key[(seed, "joint_beta")][metric] for seed in seeds]
        )
        for index, seed in enumerate(seeds):
            axis.plot(
                [0, 1],
                [control[index], treatment[index]],
                color="0.68",
                linewidth=1.0,
                alpha=0.85,
            )
        axis.scatter(np.zeros(10), control, color="tab:orange", marker="s", zorder=3)
        axis.scatter(np.ones(10), treatment, color="tab:green", marker="^", zorder=3)
        axis.plot(
            [0, 1],
            [control.mean(), treatment.mean()],
            color="black",
            marker="o",
            linewidth=3,
            markersize=8,
            label="mean",
        )
        axis.set(
            title=label,
            xticks=(0, 1),
            xticklabels=(r"no $\beta$ target", r"joint $\Delta\beta$ target"),
            xlim=(-0.25, 1.25),
        )
        axis.grid(axis="y", alpha=0.25)
    for axis in flat_axes[len(metrics) :]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def aggregate_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["metric"]: row for row in rows}


def format_mean_sd(row: dict[str, Any], prefix: str) -> str:
    return f"{row[f'{prefix}_mean']:.6f} ± {row[f'{prefix}_sd']:.6f}"


def format_ci(row: dict[str, Any]) -> str:
    return (
        f"{row['mean_paired_improvement']:.6f} "
        f"[{row['paired_improvement_ci95_low']:.6f}, "
        f"{row['paired_improvement_ci95_high']:.6f}]"
    )


def build_report(
    aggregate: list[dict[str, Any]],
    validation: dict[str, Any],
    run_rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
) -> str:
    lookup = aggregate_lookup(aggregate)
    lines = [
        "# Ten-paired-seed beta auxiliary-task ablation",
        "",
        "## Confirmatory design",
        "",
        f"Twenty full-data models were trained as ten matched pairs with model/training seeds `{seeds[0]}` through `{seeds[-1]}`. Every run uses the same {validation['test_rows']:,}-particle beta-valid held-out population, event split seed `{validation['data_split_seed']}`, query-order seed `{validation['data_order_seed']}`, shared architecture, optimizer, PID loss weight, batch size, and early-stopping policy.",
        "",
        "Within a seed pair, component-specific initialization streams make the species embedding, shared backbone, mixture-weight head, and direct PID head exactly identical at epoch zero. The treatment adds only the fourth continuous target",
        "",
        "$$",
        "\\Delta\\beta=\\beta_{\\mathrm{rec}}-",
        "\\frac{p_{\\mathrm{gen}}}{\\sqrt{p_{\\mathrm{gen}}^2+m_s^2}}.",
        "$$",
        "",
        "Training order was alternated between variants across pairs. All teacher fixed-bin fractions, selected row counts, selection SQL, and dataset fingerprints were verified identical.",
        "",
        "## Primary paired outcomes",
        "",
        "Positive paired improvement means that the joint-$\\Delta\\beta$ model closes better. Intervals are two-sided 95% Student-$t$ intervals over the ten paired improvements. The $p$ value is an exact two-sided paired sign-flip randomization test over all $2^{10}=1024$ sign assignments.",
        "",
        "| Metric | No-beta mean ± SD | Joint-$\\Delta\\beta$ mean ± SD | Paired improvement [95% CI] | Better pairs | Exact $p$ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in ("macro_weighted_bin_tv", "macro_correct_id_mae"):
        row = lookup[metric]
        lines.append(
            f"| {row['label']} | {format_mean_sd(row, 'no_beta')} | "
            f"{format_mean_sd(row, 'joint_beta')} | {format_ci(row)} | "
            f"{row['joint_beta_better_pairs']}/10 | "
            f"{row['exact_two_sided_sign_flip_p']:.6f} |"
        )
    macro_tv = lookup["macro_weighted_bin_tv"]
    macro_correct = lookup["macro_correct_id_mae"]
    lines.extend(
        [
            "",
            f"For macro TV, the mean improvement is {macro_tv['mean_paired_improvement']:.6f}, but the median is {macro_tv['median_paired_improvement']:.6f} and joint beta is better in only {macro_tv['joint_beta_better_pairs']}/10 pairs. For correct-ID MAE, the mean improvement is {macro_correct['mean_paired_improvement']:.6f}, the median is {macro_correct['median_paired_improvement']:.6f}, and joint beta is better in {macro_correct['joint_beta_better_pairs']}/10 pairs. Both 95% intervals include zero, and neither exact test rejects a no-effect explanation.",
            "",
            "Therefore this controlled study does **not** reproduce the earlier large single-checkpoint PID improvement as a reliable auxiliary-task effect. The means reflect a mixture of occasional rescued and degraded optimization runs, not a uniform shift across seeds.",
            "",
            "The macro weighted-bin total-variation endpoint is",
            "",
            "$$",
            "\\overline{\\mathrm{TV}}_{\\mathrm{macro}}=",
            "\\frac{1}{3}\\sum_s",
            "\\frac{\\sum_b N_{s,b}\\,\\mathrm{TV}(s,b)}{\\sum_b N_{s,b}},",
            "\\qquad",
            "\\mathrm{TV}(s,b)=\\frac{1}{2}\\sum_r",
            "\\left|P_{\\mathrm{FM}}(r\\mid s,b)-P_{\\mathrm{CJ}}(r\\mid s,b)\\right|.",
            "$$",
            "",
            "## Closure by generated species",
            "",
            "| Generated species | Metric | No-beta mean ± SD | Joint-$\\Delta\\beta$ mean ± SD | Paired improvement [95% CI] | Better pairs | Exact $p$ |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for species in GENERATED_SPECIES:
        for prefix, metric_label in (
            ("weighted_bin_tv", "weighted-bin TV"),
            ("correct_id_abs_error", "integrated correct-ID abs. error"),
        ):
            row = lookup[f"{prefix}_{SPECIES_KEY[species]}"]
            lines.append(
                f"| {SPECIES_MATH[species]} | {metric_label} | "
                f"{format_mean_sd(row, 'no_beta')} | "
                f"{format_mean_sd(row, 'joint_beta')} | {format_ci(row)} | "
                f"{row['joint_beta_better_pairs']}/10 | "
                f"{row['exact_two_sided_sign_flip_p']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Ten-seed correct-ID closure versus generated momentum](pid_correct_id_vs_gen_p_multiseed.png)",
            "",
            "![Paired weighted-bin total-variation results](paired_weighted_bin_tv.png)",
            "",
            "![Paired correct-ID absolute errors](paired_correct_id_error.png)",
            "",
            "## Secondary diagnostics",
            "",
            "| Metric | No-beta mean ± SD | Joint-$\\Delta\\beta$ mean ± SD | Paired improvement [95% CI] | Exact $p$ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in ("worst_bin_tv", "test_pid_cross_entropy", "test_pid_accuracy"):
        row = lookup[metric]
        lines.append(
            f"| {row['label']} | {format_mean_sd(row, 'no_beta')} | "
            f"{format_mean_sd(row, 'joint_beta')} | {format_ci(row)} | "
            f"{row['exact_two_sided_sign_flip_p']:.6f} |"
        )
    epochs = {
        variant: np.asarray(
            [row["best_epoch"] for row in run_rows if row["variant"] == variant]
        )
        for variant in VARIANTS
    }
    parameters = {
        variant: {row["parameter_count"] for row in run_rows if row["variant"] == variant}
        for variant in VARIANTS
    }
    worst_control = max(
        (row for row in run_rows if row["variant"] == "no_beta"),
        key=lambda row: row["macro_weighted_bin_tv"],
    )
    worst_treatment = max(
        (row for row in run_rows if row["variant"] == "joint_beta"),
        key=lambda row: row["macro_weighted_bin_tv"],
    )
    lines.extend(
        [
            "",
            "## Training summary and interpretation",
            "",
            f"The no-beta models have {next(iter(parameters['no_beta'])):,} parameters and selected epochs {epochs['no_beta'].min()}–{epochs['no_beta'].max()}; joint-$\\Delta\\beta$ models have {next(iter(parameters['joint_beta'])):,} parameters and selected epochs {epochs['joint_beta'].min()}–{epochs['joint_beta'].max()}.",
            "",
            "The current early-stopping rule minimizes the combined validation response loss, not PID closure alone. This matters for the observed outliers. "
            f"The worst no-beta PID-closure run (seed `{worst_control['seed']}`) selected epoch {worst_control['best_epoch']}, where validation PID accuracy was {worst_control['selected_validation_pid_accuracy']:.4f}; its lowest validation PID cross entropy occurred at epoch {worst_control['best_validation_pid_epoch']}, where PID accuracy was {worst_control['best_validation_pid_accuracy']:.4f}. "
            f"The worst joint-beta run (seed `{worst_treatment['seed']}`) likewise selected epoch {worst_treatment['best_epoch']} instead of its PID-cross-entropy optimum at epoch {worst_treatment['best_validation_pid_epoch']}. This makes optimization and checkpoint selection a plausible source of the large paired swings; the study does not isolate a purely physical representation benefit from $\\Delta\\beta$.",
            "",
            "The paired statistics quantify sensitivity to model initialization and shuffled training order on one fixed data split. They do not quantify uncertainty from new simulated datasets, alternative detector conditions, hyperparameter choices, or a changed event split. The ten seeds are independent training replicates, while particles within a held-out event are not treated as independent training replicates.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python experiments/run_beta_multiseed_ablation.py --device cuda:0",
            "python experiments/analyze_beta_multiseed_ablation.py",
            "```",
            "",
            "Machine-readable per-run, paired, aggregate, fixed-bin, provenance, and checkpoint-hash tables are stored beside this report.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if seeds != DEFAULT_SEEDS:
        if len(seeds) != 10 or len(set(seeds)) != 10:
            raise ValueError("Analysis requires ten distinct paired seeds")
    run_root = Path(args.run_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else run_root / "summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = sorted(run_root.glob("experiment_manifest*.json"))
    if not manifest_paths:
        raise FileNotFoundError("No experiment manifests were found")
    experiment_manifests = [read_json(path) for path in manifest_paths]
    complete_manifests = [
        manifest
        for manifest in experiment_manifests
        if manifest.get("status") == "complete"
    ]
    if not complete_manifests:
        raise RuntimeError("No completed experiment worker manifests were found")
    manifested_seeds = {
        int(pair["seed"])
        for manifest in complete_manifests
        for pair in manifest["pairs"]
    }
    if manifested_seeds != set(seeds):
        raise AssertionError("Completed worker manifests do not cover all analysis seeds")

    run_rows: list[dict[str, Any]] = []
    provenance: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in seeds:
        for variant in VARIANTS:
            run_dir = run_root / f"seed_{seed}" / variant
            row, item = run_metrics(run_dir, seed, variant)
            run_rows.append(row)
            provenance[(seed, variant)] = item
    validation = validate_all_runs(run_rows, provenance)
    aggregate = aggregate_rows(run_rows, seeds)
    paired_rows = paired_wide_rows(run_rows, seeds)
    correct_rows = correct_id_bin_rows(provenance, seeds)

    portable_runs = [
        {
            key: value
            for key, value in row.items()
            if key not in {"checkpoint_sha256"}
        }
        | {"checkpoint_sha256": row["checkpoint_sha256"]}
        for row in run_rows
    ]
    write_csv(output_dir / "per_run_metrics.csv", portable_runs)
    write_csv(output_dir / "paired_seed_metrics.csv", paired_rows)
    write_csv(output_dir / "aggregate_paired_statistics.csv", aggregate)
    write_csv(output_dir / "pid_correct_id_vs_gen_p_multiseed.csv", correct_rows)
    write_json(
        output_dir / "multiseed_ablation_metadata.json",
        {
            "definition": "Ten paired model/training seeds on one fixed beta-valid data split",
            "paired_seeds": list(seeds),
            "variant_names": list(VARIANTS),
            "statistics": {
                "confidence_interval": "two-sided Student-t interval over paired improvements, df=9",
                "randomization_test": "exact two-sided paired sign-flip test over 2^10 assignments",
                "effect_size": "paired Cohen dz = mean paired improvement / SD paired improvement",
                "multiplicity": "No multiplicity correction; two macro metrics designated primary, remaining metrics diagnostic",
            },
            "validation": validation,
            "source_manifests": [
                {
                    "name": path.name,
                    "status": manifest.get("status", "unknown"),
                    "included": manifest.get("status") == "complete",
                }
                for path, manifest in zip(manifest_paths, experiment_manifests)
            ],
            "source_manifest_policy": (
                "Only completed worker manifests contribute seeds; incomplete launch "
                "attempts are retained in provenance and excluded."
            ),
        },
    )
    plot_correct_id_multiseed(
        correct_rows, output_dir / "pid_correct_id_vs_gen_p_multiseed.png"
    )
    plot_paired_panels(
        run_rows,
        seeds,
        [
            ("macro_weighted_bin_tv", "Macro weighted-bin TV"),
            ("weighted_bin_tv_pi_plus", r"Generated $\pi^+$ weighted-bin TV"),
            ("weighted_bin_tv_proton", "Generated proton weighted-bin TV"),
            ("weighted_bin_tv_pi_minus", r"Generated $\pi^-$ weighted-bin TV"),
        ],
        (2, 2),
        output_dir / "paired_weighted_bin_tv.png",
        "Ten paired seeds: full direct-PID distribution closure",
    )
    plot_paired_panels(
        run_rows,
        seeds,
        [
            (
                "correct_id_abs_error_pi_minus",
                r"Generated $\pi^-$ correct-ID absolute error",
            ),
            (
                "correct_id_abs_error_pi_plus",
                r"Generated $\pi^+$ correct-ID absolute error",
            ),
            (
                "correct_id_abs_error_proton",
                "Generated proton correct-ID absolute error",
            ),
        ],
        (1, 3),
        output_dir / "paired_correct_id_error.png",
        "Ten paired seeds: momentum-integrated correct-ID closure",
    )
    report = build_report(aggregate, validation, run_rows, seeds)
    (output_dir / "MULTISEED_BETA_ABLATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print("MULTISEED_ANALYSIS_DONE")
    print(f"output_dir={output_dir}")
    for metric in ("macro_weighted_bin_tv", "macro_correct_id_mae"):
        row = aggregate_lookup(aggregate)[metric]
        print(
            f"{metric}: no_beta={row['no_beta_mean']:.6f} "
            f"joint_beta={row['joint_beta_mean']:.6f} "
            f"improvement={row['mean_paired_improvement']:.6f} "
            f"ci=[{row['paired_improvement_ci95_low']:.6f},"
            f"{row['paired_improvement_ci95_high']:.6f}] "
            f"p={row['exact_two_sided_sign_flip_p']:.6f}"
        )


if __name__ == "__main__":
    main()
