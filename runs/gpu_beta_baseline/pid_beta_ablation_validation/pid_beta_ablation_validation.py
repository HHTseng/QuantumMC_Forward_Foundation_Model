#!/usr/bin/env python3
"""Compare direct PID closure with and without the joint Delta-beta target.

Both published checkpoints are evaluated on the beta baseline's exact held-out
particles.  The teacher response and each model response in generated-species
and generated-momentum bin b are

    P_CJ(r | s,b) = N(s,b,r) / N(s,b),
    P_FM(r | s,b) = (1/N(s,b)) sum_i softmax(z_i)_r.

The comparison therefore tests distributional PID closure, not top-1 accuracy.
It reproduces the fixed-bin correct-ID and composite layouts used for Dr.
Joo's first-email validation while adding both checkpoints to every panel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from forwardfm_step1.config import load_config  # noqa: E402
from forwardfm_step1.data import Standardizer, load_all_splits, selection_sql  # noqa: E402
from forwardfm_step1.evaluation import (  # noqa: E402
    _raw_kinematics,
    conditional_pid_response_rows,
    integrated_correct_pid_response,
)
from forwardfm_step1.model import ConditionalMDN  # noqa: E402


EXPECTED_ORIGINAL_SHA256 = (
    "22dde8fe78c5bec337e5014be46e4c8037673015bc88d4fbe812c05bebcffe11"
)
EXPECTED_BETA_SHA256 = (
    "31e2c65ac417081123c87edf3fc7d874e618739b8b7cdd061c2ef3f92a102078"
)
EXPECTED_DATASET_SHA256 = (
    "6a7245cb0ec4125610b9dcd8c1635d70a7773eeb2b29d146dd80d5f149eb43ab"
)
EXPECTED_TEST_ROWS = 158_482
GENERATED_SPECIES = (-211, 211, 2212)
SPECIES_TEXT = {-211: "pi-", 211: "pi+", 2212: "proton"}
SPECIES_MATH = {-211: r"$\pi^-$", 211: r"$\pi^+$", 2212: "proton"}
PID_TEXT = {
    -2212: "anti-proton",
    -321: "K-",
    -211: "pi-",
    -11: "e+",
    11: "e-",
    22: "gamma",
    45: "PID 45",
    211: "pi+",
    321: "K+",
    2112: "neutron",
    2212: "proton",
    "OTHER": "OTHER",
}
MODEL_STYLE = {
    "coatjava": {"label": "COATJAVA", "color": "tab:blue", "marker": "o"},
    "original": {"label": r"FM: no $\beta$ target", "color": "tab:orange", "marker": "s"},
    "beta": {"label": r"FM: joint $\Delta\beta$ target", "color": "tab:green", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same-test direct-PID closure comparison for two checkpoints."
    )
    parser.add_argument(
        "--original-checkpoint",
        default=str(REPOSITORY_ROOT / "runs/tara_gpu_full/model.pt"),
    )
    parser.add_argument(
        "--beta-checkpoint",
        default=str(REPOSITORY_ROOT / "runs/gpu_beta_baseline/model.pt"),
    )
    parser.add_argument(
        "--beta-config",
        default=str(REPOSITORY_ROOT / "configs/gpu_beta_baseline.yaml"),
        help="Defines the common beta-valid held-out population and momentum bins.",
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def display_path(path: Path) -> str:
    """Prefer a repository-relative artifact name in portable metadata."""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def infer_pid_probabilities(
    checkpoint: dict[str, Any],
    raw_features: np.ndarray,
    species_index: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Evaluate one PID head after applying its own training feature scaler."""
    scaler = Standardizer.from_dict(checkpoint["feature_scaler"])
    features = scaler.transform(raw_features)
    model = ConditionalMDN(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            output = model(
                torch.from_numpy(features[start:stop]).to(device),
                torch.from_numpy(species_index[start:stop]).to(device),
            )
            batches.append(torch.softmax(output.pid_logits, dim=-1).cpu().numpy())
    probabilities = np.concatenate(batches, axis=0)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("PID softmax rows do not sum to one")
    return probabilities


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["generated_pid"]),
        int(row["bin_index"]),
        row.get("reconstructed_pid"),
    )


def merge_response_rows(
    original_rows: list[dict[str, Any]], beta_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    beta_by_key = {row_key(row): row for row in beta_rows}
    merged: list[dict[str, Any]] = []
    for original in original_rows:
        beta = beta_by_key[row_key(original)]
        if original["n"] != beta["n"] or not np.isclose(
            original["coatjava_fraction"], beta["coatjava_fraction"], atol=1e-14
        ):
            raise AssertionError("Teacher rows differ between checkpoint evaluations")
        merged.append(
            {
                "generated_pid": original["generated_pid"],
                "generated_species": original["generated_species"],
                "bin_index": original["bin_index"],
                "p_low_gev": original["p_low_gev"],
                "p_high_gev": original["p_high_gev"],
                "upper_edge_inclusive": original["upper_edge_inclusive"],
                "n": original["n"],
                "reconstructed_pid": original["reconstructed_pid"],
                "reconstructed_class": PID_TEXT.get(
                    original["reconstructed_pid"], str(original["reconstructed_pid"])
                ),
                "coatjava_fraction": original["coatjava_fraction"],
                "coatjava_standard_error": original["coatjava_standard_error"],
                "original_fm_mean_probability": original["fm_mean_probability"],
                "original_fm_mean_standard_error": original[
                    "fm_mean_standard_error"
                ],
                "original_signed_error": original[
                    "signed_difference_fm_minus_coatjava"
                ],
                "original_absolute_error": original["absolute_difference"],
                "beta_fm_mean_probability": beta["fm_mean_probability"],
                "beta_fm_mean_standard_error": beta["fm_mean_standard_error"],
                "beta_signed_error": beta["signed_difference_fm_minus_coatjava"],
                "beta_absolute_error": beta["absolute_difference"],
                "absolute_error_reduction": original["absolute_difference"]
                - beta["absolute_difference"],
            }
        )
    return merged


def merge_summary_rows(
    original_rows: list[dict[str, Any]], beta_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    beta_by_key = {
        (int(row["generated_pid"]), int(row["bin_index"])): row for row in beta_rows
    }
    merged: list[dict[str, Any]] = []
    for original in original_rows:
        beta = beta_by_key[(int(original["generated_pid"]), int(original["bin_index"]))]
        if original["n"] != beta["n"]:
            raise AssertionError("Momentum-bin populations differ")
        merged.append(
            {
                "generated_pid": original["generated_pid"],
                "generated_species": original["generated_species"],
                "bin_index": original["bin_index"],
                "p_low_gev": original["p_low_gev"],
                "p_high_gev": original["p_high_gev"],
                "upper_edge_inclusive": original["upper_edge_inclusive"],
                "n": original["n"],
                "original_total_variation": original["total_variation_distance"],
                "beta_total_variation": beta["total_variation_distance"],
                "total_variation_reduction": original["total_variation_distance"]
                - beta["total_variation_distance"],
                "original_max_channel_error": original[
                    "max_absolute_class_difference"
                ],
                "original_worst_reconstructed_pid": original[
                    "worst_reconstructed_pid"
                ],
                "beta_max_channel_error": beta["max_absolute_class_difference"],
                "beta_worst_reconstructed_pid": beta["worst_reconstructed_pid"],
            }
        )
    return merged


def integrated_distribution(
    generated_species: np.ndarray,
    observed_pid_index: np.ndarray,
    probabilities: np.ndarray,
    class_count: int,
    species: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    mask = generated_species == species
    observed = np.bincount(observed_pid_index[mask], minlength=class_count) / mask.sum()
    predicted = probabilities[mask].mean(axis=0)
    total_variation = float(0.5 * np.abs(predicted - observed).sum())
    return observed, predicted, total_variation


def build_integrated_rows(
    generated_species: np.ndarray,
    observed_pid_index: np.ndarray,
    original_probabilities: np.ndarray,
    beta_probabilities: np.ndarray,
    class_labels: list[int | str],
) -> list[dict[str, Any]]:
    original_correct = {
        int(row["generated_pid"]): row
        for row in integrated_correct_pid_response(
            generated_species,
            observed_pid_index,
            original_probabilities,
            class_labels,
        )
    }
    beta_correct = {
        int(row["generated_pid"]): row
        for row in integrated_correct_pid_response(
            generated_species,
            observed_pid_index,
            beta_probabilities,
            class_labels,
        )
    }
    rows: list[dict[str, Any]] = []
    for species in GENERATED_SPECIES:
        original = original_correct[species]
        beta = beta_correct[species]
        observed_correct = float(original["coatjava_correct_fraction"])
        original_value = float(original["fm_correct_mean_probability"])
        beta_value = float(beta["fm_correct_mean_probability"])
        original_error = abs(original_value - observed_correct)
        beta_error = abs(beta_value - observed_correct)
        _, _, original_tv = integrated_distribution(
            generated_species,
            observed_pid_index,
            original_probabilities,
            len(class_labels),
            species,
        )
        _, _, beta_tv = integrated_distribution(
            generated_species,
            observed_pid_index,
            beta_probabilities,
            len(class_labels),
            species,
        )
        rows.append(
            {
                "generated_pid": species,
                "generated_species": SPECIES_TEXT[species],
                "n": int(original["n"]),
                "coatjava_correct_fraction": observed_correct,
                "original_fm_correct_probability": original_value,
                "beta_fm_correct_probability": beta_value,
                "original_signed_correct_id_error": original_value - observed_correct,
                "beta_signed_correct_id_error": beta_value - observed_correct,
                "original_absolute_correct_id_error": original_error,
                "beta_absolute_correct_id_error": beta_error,
                "absolute_correct_id_error_reduction": original_error - beta_error,
                "relative_absolute_error_reduction_percent": 100.0
                * (original_error - beta_error)
                / original_error,
                "original_integrated_total_variation": original_tv,
                "beta_integrated_total_variation": beta_tv,
                "integrated_total_variation_reduction": original_tv - beta_tv,
            }
        )
    return rows


def build_species_summary(
    bin_rows: list[dict[str, Any]], integrated_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    integrated_by_species = {
        int(row["generated_pid"]): row for row in integrated_rows
    }
    rows: list[dict[str, Any]] = []
    for species in GENERATED_SPECIES:
        selected = [row for row in bin_rows if int(row["generated_pid"]) == species]
        weights = np.asarray([row["n"] for row in selected], dtype=np.float64)
        original_tv = np.asarray(
            [row["original_total_variation"] for row in selected]
        )
        beta_tv = np.asarray([row["beta_total_variation"] for row in selected])
        integrated = integrated_by_species[species]
        original_worst = selected[int(np.argmax(original_tv))]
        beta_worst = selected[int(np.argmax(beta_tv))]
        rows.append(
            {
                "generated_pid": species,
                "generated_species": SPECIES_TEXT[species],
                "n": integrated["n"],
                "original_weighted_mean_bin_total_variation": float(
                    np.average(original_tv, weights=weights)
                ),
                "beta_weighted_mean_bin_total_variation": float(
                    np.average(beta_tv, weights=weights)
                ),
                "weighted_mean_bin_total_variation_reduction": float(
                    np.average(original_tv - beta_tv, weights=weights)
                ),
                "original_max_bin_total_variation": original_worst[
                    "original_total_variation"
                ],
                "original_max_bin_momentum_gev": (
                    f"{original_worst['p_low_gev']:.0f}-{original_worst['p_high_gev']:.0f}"
                ),
                "beta_max_bin_total_variation": beta_worst["beta_total_variation"],
                "beta_max_bin_momentum_gev": (
                    f"{beta_worst['p_low_gev']:.0f}-{beta_worst['p_high_gev']:.0f}"
                ),
                "original_integrated_total_variation": integrated[
                    "original_integrated_total_variation"
                ],
                "beta_integrated_total_variation": integrated[
                    "beta_integrated_total_variation"
                ],
            }
        )
    return rows


def build_low_momentum_rows(
    generated_species: np.ndarray,
    generated_momentum: np.ndarray,
    observed_pid_index: np.ndarray,
    original_probabilities: np.ndarray,
    beta_probabilities: np.ndarray,
    class_labels: list[int | str],
) -> list[dict[str, Any]]:
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    explicit_classes: tuple[int | str, ...] = (211, 2212, 321, "other")
    rows: list[dict[str, Any]] = []
    for species in (211, 2212):
        mask = (
            (generated_species == species)
            & (generated_momentum >= 0.0)
            & (generated_momentum < 1.0)
        )
        for rec_pid in explicit_classes:
            if rec_pid == "other":
                explicit_indices = [label_to_index[value] for value in (211, 2212, 321)]
                coatjava = 1.0 - sum(
                    np.mean(observed_pid_index[mask] == index)
                    for index in explicit_indices
                )
                original = 1.0 - original_probabilities[mask][
                    :, explicit_indices
                ].sum(axis=1).mean()
                beta = 1.0 - beta_probabilities[mask][:, explicit_indices].sum(
                    axis=1
                ).mean()
            else:
                index = label_to_index[rec_pid]
                coatjava = np.mean(observed_pid_index[mask] == index)
                original = original_probabilities[mask, index].mean()
                beta = beta_probabilities[mask, index].mean()
            rows.append(
                {
                    "generated_pid": species,
                    "generated_species": SPECIES_TEXT[species],
                    "p_low_gev": 0.0,
                    "p_high_gev": 1.0,
                    "n": int(mask.sum()),
                    "reconstructed_pid": rec_pid,
                    "reconstructed_class": PID_TEXT.get(rec_pid, str(rec_pid)),
                    "coatjava_fraction": float(coatjava),
                    "original_fm_mean_probability": float(original),
                    "beta_fm_mean_probability": float(beta),
                    "original_signed_error": float(original - coatjava),
                    "beta_signed_error": float(beta - coatjava),
                }
            )
    return rows


def channel_rows(
    rows: list[dict[str, Any]], species: int, rec_pid: int
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if int(row["generated_pid"]) == species
        and row["reconstructed_pid"] == rec_pid
    ]


def draw_channel(
    axis: Any,
    rows: list[dict[str, Any]],
    title: str,
    ylabel: str | None = None,
    annotate: bool = False,
) -> None:
    x = np.asarray(
        [0.5 * (row["p_low_gev"] + row["p_high_gev"]) for row in rows]
    )
    series = (
        (
            "coatjava",
            "coatjava_fraction",
            "coatjava_standard_error",
        ),
        (
            "original",
            "original_fm_mean_probability",
            "original_fm_mean_standard_error",
        ),
        (
            "beta",
            "beta_fm_mean_probability",
            "beta_fm_mean_standard_error",
        ),
    )
    for series_index, (name, value_key, error_key) in enumerate(series):
        style = MODEL_STYLE[name]
        values = np.asarray([row[value_key] for row in rows])
        errors = np.asarray([row[error_key] for row in rows])
        axis.plot(
            x,
            values,
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            linewidth=1.8,
            markersize=5,
        )
        axis.fill_between(
            x,
            values - 1.96 * errors,
            values + 1.96 * errors,
            color=style["color"],
            alpha=0.07,
        )
        if annotate:
            offset = (0.040, -0.055, 0.0)[series_index]
            for x_value, value in zip(x, values):
                axis.text(
                    x_value,
                    value + offset,
                    f"{value:.2f}",
                    color=style["color"],
                    ha="center",
                    fontsize=6.4,
                )
    axis.set(
        title=title,
        xlabel=r"generated momentum $p_{\mathrm{gen}}$ [GeV]",
        ylabel=ylabel,
        xlim=(0.0, 9.0),
        ylim=(0.0, 1.05),
        xticks=np.arange(0.5, 9.0, 1.0),
        xticklabels=[f"{index}–{index + 1}" for index in range(9)],
    )
    axis.tick_params(axis="x", labelrotation=45)
    axis.grid(alpha=0.25)


def plot_correct_id(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.7), sharey=True)
    for axis, species in zip(axes, GENERATED_SPECIES):
        draw_channel(
            axis,
            channel_rows(rows, species, species),
            f"generated {SPECIES_MATH[species]}",
            "Correct reconstructed-PID response" if species == -211 else None,
        )
    axes[0].legend(fontsize=9)
    figure.suptitle(
        r"Direct correct-ID closure: no $\beta$ target vs joint $\Delta\beta$ target"
        "\nSame 158,482 beta-valid held-out particles; fixed 1-GeV bins"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def matrix_from_rows(
    rows: list[dict[str, Any]], value_key: str
) -> np.ndarray:
    matrix = np.zeros((2, 4), dtype=np.float64)
    generated = (211, 2212)
    reconstructed: tuple[int | str, ...] = (211, 2212, 321, "other")
    for row_index, species in enumerate(generated):
        for column_index, rec_pid in enumerate(reconstructed):
            match = next(
                row
                for row in rows
                if int(row["generated_pid"]) == species
                and row["reconstructed_pid"] == rec_pid
            )
            matrix[row_index, column_index] = match[value_key]
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError(f"Collapsed matrix {value_key} rows do not sum to one")
    return matrix


def plot_matrix(
    axis: Any, matrix: np.ndarray, title: str, show_xlabel: bool = True
) -> Any:
    image = axis.imshow(matrix, cmap="Blues", norm=Normalize(0.0, 1.0), aspect="auto")
    axis.set(
        title=title,
        xticks=np.arange(4),
        xticklabels=[r"$\pi^+$", "proton", r"$K^+$", "other"],
        yticks=np.arange(2),
        yticklabels=[r"generated $\pi^+$", "generated proton"],
        xlabel="reconstructed PID class" if show_xlabel else None,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=8.5,
            )
    return image


def draw_total_variation(axis: Any, rows: list[dict[str, Any]]) -> None:
    species_colors = {-211: "tab:blue", 211: "tab:orange", 2212: "tab:green"}
    for species in GENERATED_SPECIES:
        selected = [row for row in rows if int(row["generated_pid"]) == species]
        x = [0.5 * (row["p_low_gev"] + row["p_high_gev"]) for row in selected]
        axis.plot(
            x,
            [row["original_total_variation"] for row in selected],
            marker="s",
            linestyle="--",
            color=species_colors[species],
            alpha=0.65,
            label=f"{SPECIES_TEXT[species]}: no beta target",
        )
        axis.plot(
            x,
            [row["beta_total_variation"] for row in selected],
            marker="^",
            linestyle="-",
            color=species_colors[species],
            label=f"{SPECIES_TEXT[species]}: joint Delta-beta",
        )
    axis.set(
        xlabel=r"generated momentum $p_{\mathrm{gen}}$ [GeV]",
        ylabel="Total-variation distance",
        xlim=(0.0, 9.0),
        ylim=(0.0, 0.52),
    )
    axis.grid(alpha=0.25)


def plot_total_variation(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 6.5))
    draw_total_variation(axis, rows)
    axis.legend(ncol=2, fontsize=8)
    axis.set_title(
        "Full conditional PID-distribution closure\n"
        r"$\mathrm{TV}=\frac{1}{2}\sum_r|P_{\mathrm{FM}}(r)-P_{\mathrm{CJ}}(r)|$"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_composite(
    response_rows: list[dict[str, Any]],
    bin_rows: list[dict[str, Any]],
    integrated_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    figure = plt.figure(figsize=(23, 16.5))
    outer = figure.add_gridspec(
        3,
        2,
        width_ratios=(2.3, 1.0),
        height_ratios=(1.05, 1.55, 0.72),
        hspace=0.37,
        wspace=0.17,
    )
    correct_grid = outer[0, 0].subgridspec(1, 3, wspace=0.20)
    correct_axes = [figure.add_subplot(correct_grid[0, index]) for index in range(3)]
    for axis, species in zip(correct_axes, GENERATED_SPECIES):
        draw_channel(
            axis,
            channel_rows(response_rows, species, species),
            f"generated {SPECIES_MATH[species]}",
            "Correct-ID response" if species == -211 else None,
            annotate=False,
        )
    correct_axes[0].legend(fontsize=7)
    correct_axes[0].text(
        -0.18,
        1.18,
        "1. Correct-ID response vs generated momentum",
        transform=correct_axes[0].transAxes,
        fontsize=13,
        color="navy",
        fontweight="bold",
    )

    bar_axis = figure.add_subplot(outer[0, 1])
    x = np.arange(3)
    width = 0.25
    bar_series = (
        (
            "coatjava_correct_fraction",
            "COATJAVA",
            "tab:blue",
            -width,
        ),
        (
            "original_fm_correct_probability",
            r"FM: no $\beta$ target",
            "tab:orange",
            0.0,
        ),
        (
            "beta_fm_correct_probability",
            r"FM: joint $\Delta\beta$",
            "tab:green",
            width,
        ),
    )
    for key, label, color, offset in bar_series:
        bars = bar_axis.bar(
            x + offset,
            [row[key] for row in integrated_rows],
            width,
            label=label,
            color=color,
        )
        bar_axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    bar_axis.set(
        title="2. Momentum-integrated correct-ID response",
        ylabel="Correct-ID fraction / probability",
        xticks=x,
        xticklabels=[r"$\pi^-$", r"$\pi^+$", "proton"],
        ylim=(0.0, 1.08),
    )
    bar_axis.legend(fontsize=8)
    bar_axis.grid(axis="y", alpha=0.20)

    migration_grid = outer[1, 0].subgridspec(2, 3, hspace=0.43, wspace=0.22)
    channels = [
        (211, 211, r"$\pi^+\rightarrow\pi^+$ (correct)"),
        (211, 2212, r"$\pi^+\rightarrow p$"),
        (211, 321, r"$\pi^+\rightarrow K^+$"),
        (2212, 2212, r"$p\rightarrow p$ (correct)"),
        (2212, 211, r"$p\rightarrow\pi^+$"),
        (2212, 321, r"$p\rightarrow K^+$"),
    ]
    migration_axes = []
    for index, (species, rec_pid, title) in enumerate(channels):
        axis = figure.add_subplot(migration_grid[index // 3, index % 3])
        migration_axes.append(axis)
        draw_channel(
            axis,
            channel_rows(response_rows, species, rec_pid),
            title,
            "Response probability" if index % 3 == 0 else None,
        )
    migration_axes[0].text(
        -0.18,
        1.22,
        "3. Key migration channels vs generated momentum",
        transform=migration_axes[0].transAxes,
        fontsize=13,
        color="navy",
        fontweight="bold",
    )
    migration_axes[0].legend(fontsize=6.5, loc="upper right")

    matrix_grid = outer[1, 1].subgridspec(3, 1, hspace=0.34)
    matrix_axes = [figure.add_subplot(matrix_grid[index, 0]) for index in range(3)]
    matrices = (
        ("coatjava_fraction", "COATJAVA, 0–1 GeV"),
        ("original_fm_mean_probability", r"FM without $\beta$ target, 0–1 GeV"),
        ("beta_fm_mean_probability", r"FM with joint $\Delta\beta$ target, 0–1 GeV"),
    )
    image = None
    for index, (axis, (key, title)) in enumerate(zip(matrix_axes, matrices)):
        image = plot_matrix(
            axis,
            matrix_from_rows(matrix_rows, key),
            title,
            show_xlabel=index == len(matrix_axes) - 1,
        )
    matrix_axes[0].text(
        -0.08,
        1.31,
        "4. Low-momentum positive-hadron response",
        transform=matrix_axes[0].transAxes,
        fontsize=13,
        color="purple",
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=matrix_axes, fraction=0.035, pad=0.03)
    colorbar.set_label("response probability")

    tv_axis = figure.add_subplot(outer[2, 0])
    draw_total_variation(tv_axis, bin_rows)
    tv_axis.legend(ncol=3, fontsize=6.8, loc="upper right")
    tv_axis.set_title("5. Full reconstructed-PID distribution discrepancy")

    note_axis = figure.add_subplot(outer[2, 1])
    note_axis.axis("off")
    note_axis.text(
        0.0,
        0.98,
        "Interpretation boundary",
        fontsize=13,
        fontweight="bold",
        color="navy",
        va="top",
    )
    note_axis.text(
        0.0,
        0.82,
        "• Both checkpoints are evaluated on the same beta-valid test rows.\n"
        "• Mean softmax probabilities—not argmax accuracy—define FM response.\n"
        "• Joint Delta-beta training is associated with markedly better pi+ and proton closure.\n"
        "• This is not a pure beta-task ablation: training selections and selected epochs also differ.\n"
        "• A same-selection, multi-seed retraining study is required for causal attribution.",
        fontsize=10.5,
        va="top",
        linespacing=1.45,
    )
    figure.suptitle(
        r"Direct PID closure with and without joint $\Delta\beta$ prediction"
        "\nSame beta-valid held-out particles; fixed 1-GeV bins",
        fontsize=20,
        fontweight="bold",
        y=0.987,
    )
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def report_markdown(
    integrated_rows: list[dict[str, Any]],
    species_rows: list[dict[str, Any]],
    bin_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    original_hash: str,
    beta_hash: str,
    original_checkpoint: dict[str, Any],
    beta_checkpoint: dict[str, Any],
) -> str:
    lines = [
        "# Same-test PID closure: with and without joint beta prediction",
        "",
        "## Definition and scope",
        "",
        "For generated species $s$, reconstructed class $r$, and generated-momentum bin $b$:",
        "",
        "$$",
        "P_{\\mathrm{CJ}}(r\\mid s,b)=\\frac{N(s,b,r)}{N(s,b)},",
        "\\qquad",
        "P_{\\mathrm{FM}}(r\\mid s,b)=\\frac{1}{N(s,b)}\\sum_{i\\in(s,b)}q_\\theta(r\\mid x_i).",
        "$$",
        "",
        "Here $q_\\theta$ is the direct PID-head softmax. No argmax labels or sampled PID draws are used. Full-distribution closure is summarized by",
        "",
        "$$",
        "\\mathrm{TV}(s,b)=\\frac{1}{2}\\sum_r",
        "\\left|P_{\\mathrm{FM}}(r\\mid s,b)-P_{\\mathrm{CJ}}(r\\mid s,b)\\right|.",
        "$$",
        "",
        f"Both checkpoints are evaluated on the same {EXPECTED_TEST_ROWS:,} beta-valid held-out particles and fixed 1-GeV bins. The no-beta checkpoint (epoch {original_checkpoint['best_epoch']}) has SHA-256 `{original_hash}`; the joint-$\\Delta\\beta$ checkpoint (epoch {beta_checkpoint['best_epoch']}) has SHA-256 `{beta_hash}`.",
        "",
        "## Dr. Joo-style figures",
        "",
        "![Correct-ID closure with and without the beta target](pid_correct_id_with_without_beta.png)",
        "",
        "![Composite PID closure with and without the beta target](pid_composite_with_without_beta.png)",
        "",
        "![Total-variation closure with and without the beta target](pid_total_variation_with_without_beta.png)",
        "",
        "## Momentum-integrated closure",
        "",
        "| Generated species | N | COATJAVA correct | No-beta FM | Joint-$\\Delta\\beta$ FM | No-beta abs. error | Joint-$\\Delta\\beta$ abs. error | Error reduction | No-beta TV | Joint-$\\Delta\\beta$ TV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in integrated_rows:
        lines.append(
            f"| {row['generated_species']} | {row['n']:,} | "
            f"{row['coatjava_correct_fraction']:.6f} | "
            f"{row['original_fm_correct_probability']:.6f} | "
            f"{row['beta_fm_correct_probability']:.6f} | "
            f"{row['original_absolute_correct_id_error']:.6f} | "
            f"{row['beta_absolute_correct_id_error']:.6f} | "
            f"{row['absolute_correct_id_error_reduction']:.6f} | "
            f"{row['original_integrated_total_variation']:.6f} | "
            f"{row['beta_integrated_total_variation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Fixed-bin full-distribution closure",
            "",
            "The weighted mean uses the particle count in each momentum bin. The maximum identifies the worst single fixed bin for each checkpoint.",
            "",
            "| Generated species | No-beta weighted TV | Joint-$\\Delta\\beta$ weighted TV | Reduction | No-beta maximum TV (bin GeV) | Joint-$\\Delta\\beta$ maximum TV (bin GeV) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in species_rows:
        lines.append(
            f"| {row['generated_species']} | "
            f"{row['original_weighted_mean_bin_total_variation']:.6f} | "
            f"{row['beta_weighted_mean_bin_total_variation']:.6f} | "
            f"{row['weighted_mean_bin_total_variation_reduction']:.6f} | "
            f"{row['original_max_bin_total_variation']:.6f} ({row['original_max_bin_momentum_gev']}) | "
            f"{row['beta_max_bin_total_variation']:.6f} ({row['beta_max_bin_momentum_gev']}) |"
        )
    largest = sorted(
        bin_rows,
        key=lambda row: float(row["original_total_variation"]),
        reverse=True,
    )[:6]
    lines.extend(
        [
            "",
            "### Six bins with the largest no-beta discrepancy",
            "",
            "| Generated species | Momentum [GeV] | N | No-beta TV | Joint-$\\Delta\\beta$ TV | Reduction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in largest:
        lines.append(
            f"| {row['generated_species']} | {row['p_low_gev']:.0f}–{row['p_high_gev']:.0f} | "
            f"{row['n']:,} | {row['original_total_variation']:.6f} | "
            f"{row['beta_total_variation']:.6f} | "
            f"{row['total_variation_reduction']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Low-momentum positive-hadron response (0–1 GeV)",
            "",
            "| Generated | Reconstructed | COATJAVA | No-beta FM | Joint-$\\Delta\\beta$ FM |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in matrix_rows:
        lines.append(
            f"| {row['generated_species']} | {row['reconstructed_class']} | "
            f"{row['coatjava_fraction']:.6f} | "
            f"{row['original_fm_mean_probability']:.6f} | "
            f"{row['beta_fm_mean_probability']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The `other` column sums every reconstructed class except $\\pi^+$, proton, and $K^+$.",
            "",
            "## Interpretation boundary",
            "",
            "On this common test population, the joint-$\\Delta\\beta$ checkpoint is associated with substantially smaller direct PID closure errors for generated $\\pi^+$ and protons, including the low-momentum cross-migration channels emphasized by Dr. Joo. The $\\pi^-$ correct-ID closure also improves when integrated, although its worst low-momentum full-distribution bin remains visibly imperfect.",
            "",
            "This comparison is controlled at evaluation time, not training time. The checkpoints share the architecture width/depth, PID loss weight, dataset fingerprint, event-split seed, and evaluation rows, but the no-beta model was trained with the older `rec_beta > -99` selection and three continuous targets; the beta model used $0<\\beta_{\\mathrm{rec}}\\leq1.2$ and four continuous targets. They also selected different early-stopping epochs. Consequently, the plots establish an association, not that the auxiliary beta task alone caused the improvement. A same-selection no-beta retraining and multiple seeds are still required.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python runs/gpu_beta_baseline/pid_beta_ablation_validation/pid_beta_ablation_validation.py",
            "```",
            "",
            "Machine-readable tables and metadata are stored beside this report.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    original_path = Path(args.original_checkpoint).resolve()
    beta_path = Path(args.beta_checkpoint).resolve()
    config_path = Path(args.beta_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_hash = file_sha256(original_path)
    beta_hash = file_sha256(beta_path)
    if original_hash != EXPECTED_ORIGINAL_SHA256:
        raise AssertionError("Unexpected original checkpoint SHA-256")
    if beta_hash != EXPECTED_BETA_SHA256:
        raise AssertionError("Unexpected beta checkpoint SHA-256")
    original_checkpoint = torch.load(
        original_path, map_location="cpu", weights_only=False
    )
    beta_checkpoint = torch.load(beta_path, map_location="cpu", weights_only=False)
    for checkpoint in (original_checkpoint, beta_checkpoint):
        if checkpoint["dataset_metadata_sha256"] != EXPECTED_DATASET_SHA256:
            raise AssertionError("A checkpoint records a different dataset fingerprint")

    config = load_config(config_path)
    if selection_sql(config) != beta_checkpoint["selection_sql"]:
        raise AssertionError("Beta config selection differs from beta checkpoint")
    splits, beta_feature_scaler, _, vocabulary, audit = load_all_splits(config)
    test = splits["test"]
    if len(test) != EXPECTED_TEST_ROWS:
        raise AssertionError(f"Unexpected beta-valid test row count: {len(test)}")
    if list(original_checkpoint["rec_pid_vocabulary"]) != list(vocabulary):
        raise AssertionError("Original checkpoint PID vocabulary differs")
    if list(beta_checkpoint["rec_pid_vocabulary"]) != list(vocabulary):
        raise AssertionError("Beta checkpoint PID vocabulary differs")

    device = choose_device(args.device)
    raw_features = beta_feature_scaler.inverse(test.continuous)
    batch_size = int(config["training"]["batch_size"])
    original_probabilities = infer_pid_probabilities(
        original_checkpoint,
        raw_features,
        test.species_index,
        device,
        batch_size,
    )
    beta_probabilities = infer_pid_probabilities(
        beta_checkpoint,
        raw_features,
        test.species_index,
        device,
        batch_size,
    )
    generated_momentum = _raw_kinematics(test, beta_feature_scaler)["gen_p"]
    class_labels: list[int | str] = [*vocabulary, "OTHER"]
    momentum_edges = np.asarray(
        config["evaluation"]["pid_momentum_edges_gev"], dtype=np.float64
    )
    outside_bins = int(
        np.sum(
            (generated_momentum < momentum_edges[0])
            | (generated_momentum > momentum_edges[-1])
        )
    )
    if outside_bins:
        raise AssertionError(f"{outside_bins} test particles are outside PID bins")

    original_response, original_summary = conditional_pid_response_rows(
        test.raw_species,
        generated_momentum,
        test.rec_pid_index,
        original_probabilities,
        class_labels,
        momentum_edges,
    )
    beta_response, beta_summary = conditional_pid_response_rows(
        test.raw_species,
        generated_momentum,
        test.rec_pid_index,
        beta_probabilities,
        class_labels,
        momentum_edges,
    )
    response_rows = merge_response_rows(original_response, beta_response)
    bin_rows = merge_summary_rows(original_summary, beta_summary)
    integrated_rows = build_integrated_rows(
        test.raw_species,
        test.rec_pid_index,
        original_probabilities,
        beta_probabilities,
        class_labels,
    )
    species_rows = build_species_summary(bin_rows, integrated_rows)
    matrix_rows = build_low_momentum_rows(
        test.raw_species,
        generated_momentum,
        test.rec_pid_index,
        original_probabilities,
        beta_probabilities,
        class_labels,
    )

    write_csv(output_dir / "pid_response_fixed_bins_comparison.csv", response_rows)
    write_csv(output_dir / "pid_bin_closure_comparison.csv", bin_rows)
    write_csv(output_dir / "pid_integrated_correct_id_comparison.csv", integrated_rows)
    write_csv(output_dir / "pid_species_closure_summary.csv", species_rows)
    write_csv(output_dir / "pid_low_momentum_matrices_comparison.csv", matrix_rows)
    plot_correct_id(response_rows, output_dir / "pid_correct_id_with_without_beta.png")
    plot_total_variation(
        bin_rows, output_dir / "pid_total_variation_with_without_beta.png"
    )
    plot_composite(
        response_rows,
        bin_rows,
        integrated_rows,
        matrix_rows,
        output_dir / "pid_composite_with_without_beta.png",
    )

    metadata = {
        "definition": (
            "Both direct PID heads are evaluated on the beta baseline's exact "
            "held-out rows; teacher responses are empirical fractions and model "
            "responses are mean softmax probabilities."
        ),
        "test_rows": len(test),
        "split": "event-disjoint beta-valid held-out test",
        "bin_edges_gev": momentum_edges.tolist(),
        "generated_species": list(GENERATED_SPECIES),
        "reconstructed_pid_classes": class_labels,
        "dataset_metadata_sha256": EXPECTED_DATASET_SHA256,
        "loaded_dataset_metadata_sha256": audit["dataset_metadata_sha256"],
        "device": str(device),
        "original_checkpoint": {
            "path": display_path(original_path),
            "sha256": original_hash,
            "best_epoch": int(original_checkpoint["best_epoch"]),
            "target_dim": int(original_checkpoint["architecture"]["target_dim"]),
            "selection_sql": original_checkpoint["selection_sql"],
        },
        "beta_checkpoint": {
            "path": display_path(beta_path),
            "sha256": beta_hash,
            "best_epoch": int(beta_checkpoint["best_epoch"]),
            "target_dim": int(beta_checkpoint["architecture"]["target_dim"]),
            "selection_sql": beta_checkpoint["selection_sql"],
        },
        "causal_attribution": False,
        "caveat": (
            "Same-test comparison only. Training selections, continuous target "
            "dimensions, and selected early-stopping epochs differ."
        ),
    }
    if metadata["loaded_dataset_metadata_sha256"] != EXPECTED_DATASET_SHA256:
        raise AssertionError("Loaded dataset fingerprint differs")
    write_json(output_dir / "pid_beta_ablation_metadata.json", metadata)
    report = report_markdown(
        integrated_rows,
        species_rows,
        bin_rows,
        matrix_rows,
        original_hash,
        beta_hash,
        original_checkpoint,
        beta_checkpoint,
    )
    (output_dir / "PID_BETA_ABLATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )

    print("Same-test PID beta-ablation validation complete")
    print(f"device={device}")
    print(f"test_rows={len(test)}")
    print(f"original_checkpoint_sha256={original_hash}")
    print(f"beta_checkpoint_sha256={beta_hash}")
    for row in integrated_rows:
        print(
            f"{row['generated_species']}: COATJAVA={row['coatjava_correct_fraction']:.6f} "
            f"no_beta={row['original_fm_correct_probability']:.6f} "
            f"joint_delta_beta={row['beta_fm_correct_probability']:.6f}"
        )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
