#!/usr/bin/env python3
"""Compare finished training runs on the untouched held-out test split.

The Optuna search only ever saw the training and validation splits.  This
script reads the artifacts that ``train.py`` already writes for each run and
produces the side-by-side tables and figures used in the README:

* headline held-out metrics (residual NLL, PID cross entropy, PID accuracy,
  marginal PID discrepancy, physical sampled fraction, parameter count);
* moment closure of the three residual targets, per generated species, as a
  bias term ``|E[Delta]_model-E[Delta]_obs|/Std[Delta]_obs`` and a width term
  ``Std[Delta]_model/Std[Delta]_obs``;
* conditional PID response closure against the COATJAVA teacher in the fixed
  1 GeV generated-momentum bins, both correct-identification curves and the
  total-variation distance over the full reconstructed-class distribution;
* validation learning curves.

Usage:

    python experiments/compare_final_models.py \
        --run baseline=runs/optuna_baseline_repro \
        --run tuned=runs/optuna_best \
        --output-dir runs/optuna_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from forwardfm_step1.evaluation import SPECIES_LABELS

TARGETS = ("delta_p", "delta_theta", "delta_phi")
TARGET_TITLES = {
    "delta_p": r"$\Delta p$ [GeV]",
    "delta_theta": r"$\Delta\theta$ [rad]",
    "delta_phi": r"$\Delta\phi$ [rad]",
}
SPECIES_TITLES = {"pi-": r"$\pi^-$", "pi+": r"$\pi^+$", "proton": "proton"}
PALETTE = ("#2b6cb0", "#c53030", "#2f855a", "#6b46c1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeatable; the first entry is treated as the reference run",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


class Run:
    def __init__(self, label: str, path: Path) -> None:
        self.label = label
        self.path = path
        self.metrics = json.loads((path / "metrics.json").read_text())
        self.history = json.loads((path / "history.json").read_text())
        self.config = json.loads(
            json.dumps(_load_yaml(path / "resolved_config.yaml"))
        )
        self.pid_bins = _read_csv(path / "pid_response_fixed_bins.csv")

    @property
    def parameter_count(self) -> int | None:
        card = self.path / "MODEL_CARD.md"
        if not card.exists():
            return None
        for line in card.read_text().splitlines():
            if line.startswith("- Trainable parameters:"):
                return int(line.split(":")[1].strip().replace(",", ""))
        return None

    @property
    def best_epoch(self) -> int | None:
        card = self.path / "MODEL_CARD.md"
        if not card.exists():
            return None
        for line in card.read_text().splitlines():
            if line.startswith("- Best validation epoch:"):
                return int(line.split(":")[1].strip())
        return None


def _load_yaml(path: Path) -> Any:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def weighted_mean_tv(run: Run) -> tuple[float, float]:
    summary = run.metrics["pid_conditional_closure"]["bin_summary"]
    weights = np.array([row["n"] for row in summary], dtype=np.float64)
    values = np.array(
        [row["total_variation_distance"] for row in summary], dtype=np.float64
    )
    return float(np.average(values, weights=weights)), float(values.max())


def moment_rows(run: Run) -> list[dict[str, Any]]:
    rows = []
    for row in run.metrics["closure"]:
        observed_std = float(row["observed_std"])
        rows.append(
            {
                "run": run.label,
                "species": row["species"],
                "target": row["target"],
                "n": row["n"],
                "observed_mean": row["observed_mean"],
                "sampled_mean": row["sampled_mean"],
                "bias_in_sigma": abs(
                    float(row["sampled_mean"]) - float(row["observed_mean"])
                )
                / observed_std,
                "observed_std": observed_std,
                "sampled_std": row["sampled_std"],
                "std_ratio": row["std_ratio"],
                "wasserstein_1d": row["wasserstein_1d"],
            }
        )
    return rows


def moment_error(run: Run) -> float:
    rows = moment_rows(run)
    return float(
        np.mean([row["bias_in_sigma"] + abs(float(row["std_ratio"]) - 1.0) for row in rows])
    )


def headline_table(runs: list[Run]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        test = run.metrics["test"]
        tv_mean, tv_max = weighted_mean_tv(run)
        model = run.config["model"]
        training = run.config["training"]
        rows.append(
            {
                "run": run.label,
                "run_dir": str(run.path),
                "hidden_width": model["hidden_width"],
                "hidden_layers": model["hidden_layers"],
                "pid_embedding_dim": model["pid_embedding_dim"],
                "mixture_components": model["mixture_components"],
                "dropout": model["dropout"],
                "batch_size": training["batch_size"],
                "learning_rate": training["learning_rate"],
                "weight_decay": training["weight_decay"],
                "pid_loss_weight": training["pid_loss_weight"],
                "lr_schedule": training.get("lr_schedule", "none"),
                "epoch_budget": training["epochs"],
                "epochs_run": len(run.history),
                "best_epoch": run.best_epoch,
                "trainable_parameters": run.parameter_count,
                "test_residual_nll": test["residual_nll"],
                "test_pid_cross_entropy": test["pid_cross_entropy"],
                "test_joint_nll": test["residual_nll"] + test["pid_cross_entropy"],
                "test_pid_accuracy": test["pid_accuracy"],
                "pid_max_marginal_discrepancy": run.metrics["pid"][
                    "max_absolute_fraction_difference"
                ],
                "pid_weighted_mean_tv": tv_mean,
                "pid_max_bin_tv": tv_max,
                "moment_closure_error": moment_error(run),
                "physical_sample_fraction": run.metrics["joint_and_physical"][
                    "physical_sample_fraction"
                ],
                "test_examples_per_second": test["examples_per_second"],
            }
        )
    return rows


def correct_identification_rows(run: Run) -> dict[str, dict[str, list[float]]]:
    """Diagonal reconstructed-class response versus generated momentum bin."""
    out: dict[str, dict[str, list[float]]] = {}
    for species_pid, label in SPECIES_LABELS.items():
        selected = [
            row
            for row in run.pid_bins
            if int(row["generated_pid"]) == species_pid
            and row["reconstructed_pid"] == str(species_pid)
        ]
        selected.sort(key=lambda row: int(row["bin_index"]))
        out[label] = {
            "centers": [
                0.5 * (float(row["p_low_gev"]) + float(row["p_high_gev"]))
                for row in selected
            ],
            "coatjava": [float(row["coatjava_fraction"]) for row in selected],
            "coatjava_se": [float(row["coatjava_standard_error"]) for row in selected],
            "model": [float(row["fm_mean_probability"]) for row in selected],
            "n": [int(row["n"]) for row in selected],
        }
    return out


def plot_headline(runs: list[Run], path: Path) -> None:
    panels = [
        ("test_residual_nll", "residual NLL", True),
        ("test_pid_cross_entropy", "PID cross entropy", True),
        ("test_pid_accuracy", "PID top-1 accuracy", False),
        ("pid_weighted_mean_tv", "PID weighted mean TV", True),
        ("moment_closure_error", "moment closure error", True),
        ("physical_sample_fraction", r"physical $(p,\theta)$ fraction", False),
    ]
    table = {row["run"]: row for row in headline_table(runs)}
    figure, axes_grid = plt.subplots(2, 3, figsize=(14.0, 7.2))
    labels = [run.label for run in runs]
    for axes, (key, title, lower_better) in zip(axes_grid.ravel(), panels):
        values = [table[label][key] for label in labels]
        bars = axes.bar(labels, values, color=PALETTE[: len(labels)], width=0.55)
        for bar, value in zip(bars, values):
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4g}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
        axes.set_title(f"{title}  ({'lower' if lower_better else 'higher'} is better)", fontsize=10)
        axes.grid(alpha=0.25, axis="y")
        axes.tick_params(axis="x", labelsize=9)
        span = max(values) - min(values)
        if span > 0:
            axes.set_ylim(min(values) - 0.6 * span, max(values) + 0.45 * span)
    figure.suptitle("Held-out test metrics on the untouched test split", y=0.99)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_moment_closure(runs: list[Run], path: Path) -> None:
    species = list(SPECIES_LABELS.values())
    figure, axes_grid = plt.subplots(2, len(runs), figsize=(4.9 * len(runs), 6.6), squeeze=False)
    for column, run in enumerate(runs):
        lookup = {(row["species"], row["target"]): row for row in moment_rows(run)}
        width = np.array(
            [[float(lookup[(s, t)]["std_ratio"]) for t in TARGETS] for s in species]
        )
        bias = np.array(
            [[float(lookup[(s, t)]["bias_in_sigma"]) for t in TARGETS] for s in species]
        )
        for row_index, (values, title, cmap, limits) in enumerate(
            (
                (width, "sampled / observed width", "RdBu_r", (0.9, 1.1)),
                (bias, r"$|$bias$|$ in units of observed $\sigma$", "magma_r", (0.0, 0.05)),
            )
        ):
            axes = axes_grid[row_index][column]
            image = axes.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1])
            axes.set_xticks(range(len(TARGETS)), [TARGET_TITLES[t] for t in TARGETS])
            axes.set_yticks(range(len(species)), [SPECIES_TITLES[s] for s in species])
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    axes.text(
                        j, i, f"{values[i, j]:.4f}", ha="center", va="center", fontsize=9
                    )
            axes.set_title(f"{run.label}\n{title}", fontsize=10)
            figure.colorbar(image, ax=axes, shrink=0.8)
    figure.suptitle("Residual moment closure: perfect agreement is 1.0000 and 0.0000", y=1.0)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_pid_correct_identification(runs: list[Run], path: Path) -> None:
    species = list(SPECIES_LABELS.values())
    figure, axes_grid = plt.subplots(1, len(species), figsize=(5.2 * len(species), 4.5))
    reference = correct_identification_rows(runs[0])
    for axes, label in zip(np.atleast_1d(axes_grid), species):
        teacher = reference[label]
        axes.errorbar(
            teacher["centers"],
            teacher["coatjava"],
            yerr=teacher["coatjava_se"],
            fmt="ko",
            ms=5,
            capsize=3,
            label="COATJAVA (full simulation)",
            zorder=5,
        )
        for color, run in zip(PALETTE, runs):
            curve = correct_identification_rows(run)[label]
            axes.plot(
                curve["centers"], curve["model"], "-o", ms=4, color=color, label=run.label
            )
        axes.set_title(f"generated {SPECIES_TITLES[label]}")
        axes.set_xlabel(r"generated $p$ [GeV]")
        axes.set_ylabel(r"$P(\hat{s}=s_{\mathrm{gen}}\mid s_{\mathrm{gen}},p)$")
        axes.grid(alpha=0.3)
        axes.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Correct-identification response in fixed 1 GeV generated-momentum bins", y=1.0
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_pid_total_variation(runs: list[Run], path: Path) -> None:
    species = list(SPECIES_LABELS.values())
    figure, axes_grid = plt.subplots(1, len(species), figsize=(5.2 * len(species), 4.3))
    for axes, label in zip(np.atleast_1d(axes_grid), species):
        for color, run in zip(PALETTE, runs):
            summary = [
                row
                for row in run.metrics["pid_conditional_closure"]["bin_summary"]
                if row["generated_species"] == label
            ]
            summary.sort(key=lambda row: row["bin_index"])
            centers = [
                0.5 * (row["p_low_gev"] + row["p_high_gev"]) for row in summary
            ]
            axes.plot(
                centers,
                [row["total_variation_distance"] for row in summary],
                "-o",
                ms=4,
                color=color,
                label=run.label,
            )
        axes.set_title(f"generated {SPECIES_TITLES[label]}")
        axes.set_xlabel(r"generated $p$ [GeV]")
        axes.set_ylabel(r"$\mathrm{TV}(s,b)$")
        axes.grid(alpha=0.3)
        axes.legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Total-variation distance over the full reconstructed-class distribution "
        "(lower is better)",
        y=1.0,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_learning_curves(runs: list[Run], path: Path) -> None:
    figure, axes_grid = plt.subplots(1, 3, figsize=(15.0, 4.4))
    keys = (
        ("residual_nll", "validation residual NLL"),
        ("pid_cross_entropy", "validation PID cross entropy"),
        ("pid_accuracy", "validation PID top-1 accuracy"),
    )
    for axes, (key, title) in zip(axes_grid, keys):
        for color, run in zip(PALETTE, runs):
            epochs = [entry["epoch"] for entry in run.history]
            axes.plot(
                epochs,
                [entry["validation"][key] for entry in run.history],
                color=color,
                label=run.label,
            )
            if run.best_epoch:
                index = run.best_epoch - 1
                axes.plot(
                    run.best_epoch,
                    run.history[index]["validation"][key],
                    "*",
                    ms=14,
                    color=color,
                )
        axes.set_xlabel("epoch")
        axes.set_ylabel(title)
        axes.grid(alpha=0.3)
        axes.legend(frameon=False, fontsize=9)
    figure.suptitle("Validation trajectories; the star marks the selected checkpoint", y=1.0)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    runs: list[Run] = []
    for entry in args.run:
        label, _, raw_path = entry.partition("=")
        runs.append(Run(label, Path(raw_path)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    headline = headline_table(runs)
    write_csv(headline, output_dir / "final_headline_metrics.csv")
    write_csv(
        [row for run in runs for row in moment_rows(run)],
        output_dir / "final_moment_closure.csv",
    )
    tv_rows = []
    for run in runs:
        for row in run.metrics["pid_conditional_closure"]["bin_summary"]:
            tv_rows.append({"run": run.label, **row})
    write_csv(tv_rows, output_dir / "final_pid_bin_total_variation.csv")
    correct_rows = []
    for run in runs:
        for label, curve in correct_identification_rows(run).items():
            for index, center in enumerate(curve["centers"]):
                correct_rows.append(
                    {
                        "run": run.label,
                        "generated_species": label,
                        "p_center_gev": center,
                        "n": curve["n"][index],
                        "coatjava_fraction": curve["coatjava"][index],
                        "fm_mean_probability": curve["model"][index],
                        "absolute_difference": abs(
                            curve["model"][index] - curve["coatjava"][index]
                        ),
                    }
                )
    write_csv(correct_rows, output_dir / "final_pid_correct_identification.csv")

    plot_headline(runs, output_dir / "final_headline_metrics.png")
    plot_moment_closure(runs, output_dir / "final_moment_closure.png")
    plot_pid_correct_identification(runs, output_dir / "final_pid_correct_identification.png")
    plot_pid_total_variation(runs, output_dir / "final_pid_total_variation.png")
    plot_learning_curves(runs, output_dir / "final_learning_curves.png")

    reference, *others = headline
    deltas = []
    for row in others:
        delta = {"run": row["run"], "versus": reference["run"]}
        for key in (
            "test_residual_nll",
            "test_pid_cross_entropy",
            "test_joint_nll",
            "test_pid_accuracy",
            "pid_max_marginal_discrepancy",
            "pid_weighted_mean_tv",
            "pid_max_bin_tv",
            "moment_closure_error",
            "physical_sample_fraction",
        ):
            delta[f"{key}_delta"] = row[key] - reference[key]
            if reference[key]:
                delta[f"{key}_relative"] = (row[key] - reference[key]) / abs(reference[key])
        deltas.append(delta)
    write_csv(deltas, output_dir / "final_deltas.csv")
    print(json.dumps({"headline": headline, "deltas": deltas}, indent=2)[:4000])


if __name__ == "__main__":
    main()
