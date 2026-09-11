#!/usr/bin/env python3
"""Compare baseline and validation-selected trigger-efficiency checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIABLES = (
    ("gen_p", r"$p_e$ [GeV]"),
    ("gen_theta", r"$\theta_e$ [deg]"),
    ("gen_phi", r"$\phi_e$ [deg]"),
    ("gen_vz", r"$v_{z,e}$ [cm]"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="runs/trigger_electron_efficiency")
    parser.add_argument("--tuned", default="runs/trigger_optuna_best")
    parser.add_argument("--output-dir", default="runs/trigger_optuna_analysis")
    args = parser.parse_args()
    baseline, tuned = Path(args.baseline), Path(args.tuned)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaded = {
        name: json.loads((path / "metrics.json").read_text())
        for name, path in (("baseline", baseline), ("Optuna", tuned))
    }
    fields = (
        "binary_cross_entropy",
        "brier_score",
        "expected_calibration_error",
        "roc_auc",
        "threshold_accuracy",
        "signed_integrated_difference",
    )
    rows = []
    for name in ("baseline", "Optuna"):
        test = loaded[name]["test"]
        row = {"model": name, **{field: test[field] for field in fields}}
        for variable in ("gen_p", "gen_theta", "gen_phi", "gen_vz", "gen_p_theta"):
            row[f"{variable}_weighted_mae"] = loaded[name]["closure"][variable][
                "particle_weighted_mean_absolute_error"
            ]
        rows.append(row)
    with (output / "test_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for axis, (variable, xlabel) in zip(axes.flat, VARIABLES):
        by_model = {
            name: read_csv(path / f"efficiency_vs_{variable}.csv")
            for name, path in (("baseline", baseline), ("Optuna", tuned))
        }
        observed = by_model["baseline"]
        centers = [
            (float(row["low"]) + float(row["high"])) / 2 for row in observed
        ]
        axis.errorbar(
            centers,
            [float(row["observed_efficiency"]) for row in observed],
            yerr=[float(row["observed_binomial_standard_error"]) for row in observed],
            color="black",
            marker="o",
            capsize=2,
            label="full simulation",
        )
        for name, style in (("baseline", "--s"), ("Optuna", "-^")):
            values = by_model[name]
            axis.plot(
                centers,
                [float(row["fm_mean_probability"]) for row in values],
                style,
                label=name,
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(r"$P(T=1\mid x_e)$")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Trigger-efficiency closure: default versus Optuna")
    figure.tight_layout()
    figure.savefig(output / "trigger_optuna_test_closure.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
