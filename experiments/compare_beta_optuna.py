#!/usr/bin/env python3
"""Compare same-seed D10 and validation-selected Optuna checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SPECIES = ((-211, r"$\pi^-$"), (211, r"$\pi^+$"), (2212, "proton"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def weighted_pid_tv(path: Path) -> float:
    rows = read_csv(path / "pid_bin_closure_summary.csv")
    counts = np.asarray([int(row["n"]) for row in rows])
    values = np.asarray([float(row["total_variation_distance"]) for row in rows])
    return float(np.average(values, weights=counts))


def macro_beta_w1(path: Path) -> float:
    rows = read_csv(path / "beta_closure_overall.csv")
    return float(np.mean([float(row["wasserstein_1d"]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="runs/gpu_beta_gen_factorial/seed_20260822/D_input_target_pid1",
    )
    parser.add_argument("--tuned", default="runs/beta_optuna_best")
    parser.add_argument("--output-dir", default="runs/beta_optuna_analysis")
    args = parser.parse_args()
    baseline, tuned = Path(args.baseline), Path(args.tuned)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, float | str]] = []
    for name, path in (("D10 default", baseline), ("Optuna", tuned)):
        metrics = json.loads((path / "metrics.json").read_text())
        test = metrics["test"]
        summary.append(
            {
                "model": name,
                "pid_tv": weighted_pid_tv(path),
                "pid_cross_entropy": float(test["pid_cross_entropy"]),
                "pid_accuracy": float(test["pid_accuracy"]),
                "residual_nll": float(test["residual_nll"]),
                "beta_w1_macro": macro_beta_w1(path),
            }
        )
    with (output / "test_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    pid = {name: read_csv(path / "pid_response_fixed_bins.csv") for name, path in (("D10 default", baseline), ("Optuna", tuned))}
    beta = {name: read_csv(path / "beta_closure_vs_gen_p.csv") for name, path in (("D10 default", baseline), ("Optuna", tuned))}
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), sharex="col")
    for column, (species, label) in enumerate(SPECIES):
        axis = axes[0, column]
        for model, style in (("D10 default", "--s"), ("Optuna", "-^")):
            rows = [
                row
                for row in pid[model]
                if int(row["generated_pid"]) == species
                and row["reconstructed_pid"] == str(species)
            ]
            x = [(float(row["p_low_gev"]) + float(row["p_high_gev"])) / 2 for row in rows]
            if model == "D10 default":
                axis.plot(x, [float(row["coatjava_fraction"]) for row in rows], "o-", color="black", label="COATJAVA")
            axis.plot(x, [float(row["fm_mean_probability"]) for row in rows], style, label=model)
        axis.set_title(f"generated {label}")
        axis.grid(alpha=0.25)
        axis.set_ylim(0, 1.02)

        axis = axes[1, column]
        for model, style in (("D10 default", "--s"), ("Optuna", "-^")):
            rows = [row for row in beta[model] if int(row["generated_pid"]) == species]
            x = [(float(row["p_low_gev"]) + float(row["p_high_gev"])) / 2 for row in rows]
            if model == "D10 default":
                axis.plot(x, [float(row["observed_mean"]) for row in rows], "o-", color="black", label="COATJAVA")
            axis.plot(x, [float(row["sampled_mean"]) for row in rows], style, label=model)
        axis.grid(alpha=0.25)
        axis.set_xlabel(r"$p_{\rm gen}$ [GeV]")
        axis.set_ylim(0.5, 1.02)
    axes[0, 0].set_ylabel("correct-PID probability")
    axes[1, 0].set_ylabel(r"mean reconstructed $\beta$")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    figure.suptitle("Same-seed D10: default versus validation-selected Optuna recipe")
    figure.tight_layout()
    figure.savefig(output / "beta_optuna_test_closure.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
