#!/usr/bin/env python3
"""Reproduce Dr. Joo's first-email PID response closure figures.

The frozen H100 checkpoint is evaluated on the exact held-out test split. For
each generated species and fixed 1-GeV momentum bin, this script compares the
empirical COATJAVA reconstructed-PID fractions with the mean Forward-FM
softmax probabilities for the same particles. No argmax counts, stochastic
PID draws, optimization, or checkpoint changes enter the primary closure.
"""

from __future__ import annotations

import argparse
import csv
import glob
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
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from forwardfm_step1.data import (  # noqa: E402
    Standardizer,
    _load_frame,
    assert_schema,
    connect,
    prepare_split,
    selection_sql,
)
from forwardfm_step1.model import ConditionalMDN  # noqa: E402
from forwardfm_step1.training import make_loader  # noqa: E402


EXPECTED_CHECKPOINT_SHA256 = (
    "22dde8fe78c5bec337e5014be46e4c8037673015bc88d4fbe812c05bebcffe11"
)
EXPECTED_DATASET_SHA256 = (
    "6a7245cb0ec4125610b9dcd8c1635d70a7773eeb2b29d146dd80d5f149eb43ab"
)
EXPECTED_TEST_ROWS = 158_985
EXPECTED_BEST_EPOCH = 16
GENERATED_SPECIES = (-211, 211, 2212)
FIXED_P_EDGES_GEV = np.arange(0.0, 10.0, 1.0)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-bin reconstructed-PID response closure for Email 1."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(REPOSITORY_ROOT / "runs/tara_gpu_full/model.pt"),
    )
    parser.add_argument(
        "--config",
        default=str(REPOSITORY_ROOT / "configs/gpu_full.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_metadata_sha256(parquet_glob: str) -> tuple[str, list[dict[str, Any]]]:
    """Reproduce build_audit's filename-and-byte-size dataset fingerprint."""
    paths = sorted(glob.glob(parquet_glob))
    if not paths:
        raise FileNotFoundError(f"No Parquet files match {parquet_glob!r}")
    records = [
        {"name": Path(path).name, "bytes": Path(path).stat().st_size}
        for path in paths
    ]
    payload = json.dumps(records, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), records


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
        raise ValueError(f"Cannot write empty table to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def infer_pid_probabilities(
    model: ConditionalMDN,
    split: Any,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    loader = make_loader(
        split,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        seed=int(checkpoint["seed"]),
        num_workers=0,
    )
    batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for continuous, species_index, _, _ in loader:
            output = model(continuous.to(device), species_index.to(device))
            batches.append(torch.softmax(output.pid_logits, dim=-1).cpu().numpy())
    probabilities = np.concatenate(batches, axis=0)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("PID softmax rows do not sum to one")
    return probabilities


def calculate_fixed_bin_response(
    generated_species: np.ndarray,
    generated_p: np.ndarray,
    observed_pid_index: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[int | str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    response_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for species in GENERATED_SPECIES:
        for bin_index, (low, high) in enumerate(
            zip(FIXED_P_EDGES_GEV[:-1], FIXED_P_EDGES_GEV[1:])
        ):
            last_bin = bin_index == len(FIXED_P_EDGES_GEV) - 2
            mask = (
                (generated_species == species)
                & (generated_p >= low)
                & ((generated_p <= high) if last_bin else (generated_p < high))
            )
            indices = np.flatnonzero(mask)
            if not len(indices):
                continue
            observed = observed_pid_index[indices]
            q = probabilities[indices]
            observed_distribution = (
                np.bincount(observed, minlength=len(class_labels)) / len(indices)
            )
            predicted_distribution = q.mean(axis=0)
            difference = predicted_distribution - observed_distribution
            worst_index = int(np.abs(difference).argmax())
            summary_rows.append(
                {
                    "generated_pid": species,
                    "generated_species": SPECIES_TEXT[species],
                    "bin_index": bin_index,
                    "p_low_gev": float(low),
                    "p_high_gev": float(high),
                    "p_center_gev": float(0.5 * (low + high)),
                    "n": len(indices),
                    "total_variation": float(0.5 * np.abs(difference).sum()),
                    "maximum_channel_error": float(abs(difference[worst_index])),
                    "worst_reconstructed_pid": class_labels[worst_index],
                    "coatjava_row_sum": float(observed_distribution.sum()),
                    "fm_row_sum": float(predicted_distribution.sum()),
                }
            )
            for class_index, rec_pid in enumerate(class_labels):
                observed_fraction = float(observed_distribution[class_index])
                predicted_probability = float(predicted_distribution[class_index])
                observed_se = float(
                    np.sqrt(
                        observed_fraction * (1.0 - observed_fraction) / len(indices)
                    )
                )
                fm_se = float(
                    np.std(q[:, class_index], ddof=1) / np.sqrt(len(indices))
                    if len(indices) > 1
                    else 0.0
                )
                response_rows.append(
                    {
                        "generated_pid": species,
                        "generated_species": SPECIES_TEXT[species],
                        "bin_index": bin_index,
                        "p_low_gev": float(low),
                        "p_high_gev": float(high),
                        "p_center_gev": float(0.5 * (low + high)),
                        "n": len(indices),
                        "reconstructed_pid": rec_pid,
                        "reconstructed_class": PID_TEXT.get(rec_pid, str(rec_pid)),
                        "coatjava_fraction": observed_fraction,
                        "coatjava_standard_error": observed_se,
                        "fm_mean_probability": predicted_probability,
                        "fm_standard_error": fm_se,
                        "fm_minus_coatjava": predicted_probability - observed_fraction,
                        "absolute_difference": abs(
                            predicted_probability - observed_fraction
                        ),
                    }
                )
    return response_rows, summary_rows


def calculate_integrated_response(
    generated_species: np.ndarray,
    observed_pid_index: np.ndarray,
    probabilities: np.ndarray,
    class_to_index: dict[int | str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for species in GENERATED_SPECIES:
        indices = np.flatnonzero(generated_species == species)
        class_index = class_to_index[species]
        coatjava = float(np.mean(observed_pid_index[indices] == class_index))
        fm = float(np.mean(probabilities[indices, class_index]))
        rows.append(
            {
                "generated_pid": species,
                "generated_species": SPECIES_TEXT[species],
                "n": len(indices),
                "coatjava_correct_fraction": coatjava,
                "fm_correct_probability": fm,
                "fm_minus_coatjava": fm - coatjava,
            }
        )
    return rows


def calculate_low_momentum_matrices(
    generated_species: np.ndarray,
    generated_p: np.ndarray,
    observed_pid_index: np.ndarray,
    probabilities: np.ndarray,
    class_to_index: dict[int | str, int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    generated_rows = (211, 2212)
    explicit_classes = (211, 2212, 321)
    coatjava = np.zeros((len(generated_rows), 4), dtype=np.float64)
    fm = np.zeros_like(coatjava)
    rows: list[dict[str, Any]] = []
    for row_index, species in enumerate(generated_rows):
        indices = np.flatnonzero(
            (generated_species == species)
            & (generated_p >= 0.0)
            & (generated_p < 1.0)
        )
        for column_index, rec_pid in enumerate(explicit_classes):
            class_index = class_to_index[rec_pid]
            coatjava[row_index, column_index] = np.mean(
                observed_pid_index[indices] == class_index
            )
            fm[row_index, column_index] = np.mean(probabilities[indices, class_index])
        coatjava[row_index, 3] = 1.0 - coatjava[row_index, :3].sum()
        fm[row_index, 3] = 1.0 - fm[row_index, :3].sum()
        for column_index, rec_pid in enumerate((*explicit_classes, "other")):
            rows.append(
                {
                    "generated_pid": species,
                    "generated_species": SPECIES_TEXT[species],
                    "p_low_gev": 0.0,
                    "p_high_gev": 1.0,
                    "n": len(indices),
                    "reconstructed_pid": rec_pid,
                    "coatjava_fraction": float(coatjava[row_index, column_index]),
                    "fm_mean_probability": float(fm[row_index, column_index]),
                    "fm_minus_coatjava": float(
                        fm[row_index, column_index]
                        - coatjava[row_index, column_index]
                    ),
                }
            )
    if not np.allclose(coatjava.sum(axis=1), 1.0):
        raise AssertionError("Collapsed COATJAVA matrix rows do not sum to one")
    if not np.allclose(fm.sum(axis=1), 1.0):
        raise AssertionError("Collapsed FM matrix rows do not sum to one")
    return coatjava, fm, rows


def channel_rows(
    response_rows: list[dict[str, Any]], species: int, rec_pid: int
) -> list[dict[str, Any]]:
    return [
        row
        for row in response_rows
        if row["generated_pid"] == species
        and row["reconstructed_pid"] == rec_pid
    ]


def draw_channel(
    axis: Any,
    rows: list[dict[str, Any]],
    title: str,
    ylabel: str | None = None,
    annotate: bool = False,
) -> None:
    x = np.asarray([row["p_center_gev"] for row in rows])
    coatjava = np.asarray([row["coatjava_fraction"] for row in rows])
    fm = np.asarray([row["fm_mean_probability"] for row in rows])
    coatjava_se = np.asarray([row["coatjava_standard_error"] for row in rows])
    fm_se = np.asarray([row["fm_standard_error"] for row in rows])
    axis.plot(x, coatjava, "o-", color="tab:blue", label="COATJAVA")
    axis.plot(x, fm, "s-", color="tab:orange", label="Forward FM")
    axis.fill_between(
        x,
        coatjava - 1.96 * coatjava_se,
        coatjava + 1.96 * coatjava_se,
        color="tab:blue",
        alpha=0.10,
    )
    axis.fill_between(
        x,
        fm - 1.96 * fm_se,
        fm + 1.96 * fm_se,
        color="tab:orange",
        alpha=0.10,
    )
    if annotate:
        for x_value, value in zip(x, coatjava):
            axis.text(x_value, value + 0.035, f"{value:.2f}", color="tab:blue", ha="center", fontsize=7)
        for x_value, value in zip(x, fm):
            axis.text(x_value, value - 0.055, f"{value:.2f}", color="tab:orange", ha="center", fontsize=7)
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


def plot_correct_id(
    response_rows: list[dict[str, Any]], output_path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for axis, species in zip(axes, GENERATED_SPECIES):
        draw_channel(
            axis,
            channel_rows(response_rows, species, species),
            f"generated {SPECIES_MATH[species]}",
            "Correct reconstructed-PID fraction" if species == -211 else None,
        )
    axes[0].legend()
    figure.suptitle(
        "Correct reconstructed-PID response vs generated momentum\n"
        "fixed 1-GeV bins; COATJAVA fractions vs mean Forward-FM softmax probabilities"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_matrix(axis: Any, matrix: np.ndarray, title: str) -> None:
    image = axis.imshow(matrix, cmap="Blues", norm=Normalize(0.0, 1.0), aspect="auto")
    axis.set(
        title=title,
        xticks=np.arange(4),
        xticklabels=[r"$\pi^+$", "proton", r"$K^+$", "other"],
        yticks=np.arange(2),
        yticklabels=[r"generated $\pi^+$", "generated proton"],
        xlabel="reconstructed PID class",
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
                fontsize=9,
            )
    return image


def plot_composite(
    response_rows: list[dict[str, Any]],
    integrated_rows: list[dict[str, Any]],
    coatjava_matrix: np.ndarray,
    fm_matrix: np.ndarray,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(22, 15))
    outer = figure.add_gridspec(
        3,
        2,
        width_ratios=(2.25, 1.0),
        height_ratios=(1.05, 1.55, 0.55),
        hspace=0.34,
        wspace=0.16,
    )
    correct_grid = outer[0, 0].subgridspec(1, 3, wspace=0.20)
    correct_axes = [figure.add_subplot(correct_grid[0, index]) for index in range(3)]
    for axis, species in zip(correct_axes, GENERATED_SPECIES):
        draw_channel(
            axis,
            channel_rows(response_rows, species, species),
            f"generated {SPECIES_MATH[species]}",
            "Correct-ID response" if species == -211 else None,
            annotate=True,
        )
    correct_axes[0].legend(fontsize=8)
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
    width = 0.34
    coatjava_integrated = [row["coatjava_correct_fraction"] for row in integrated_rows]
    fm_integrated = [row["fm_correct_probability"] for row in integrated_rows]
    left = bar_axis.bar(x - width / 2, coatjava_integrated, width, label="COATJAVA", color="tab:blue")
    right = bar_axis.bar(x + width / 2, fm_integrated, width, label="Forward FM", color="tab:orange")
    bar_axis.bar_label(left, fmt="%.3f", padding=3)
    bar_axis.bar_label(right, fmt="%.3f", padding=3)
    bar_axis.set(
        title="2. Momentum-integrated correct-ID response",
        ylabel="Correct-ID fraction / probability",
        xticks=x,
        xticklabels=[r"$\pi^-$", r"$\pi^+$", "proton"],
        ylim=(0.0, 1.05),
    )
    bar_axis.legend()
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
    migration_axes[0].legend(fontsize=7, loc="upper right")

    matrix_grid = outer[1, 1].subgridspec(2, 1, hspace=0.38)
    matrix_axes = [figure.add_subplot(matrix_grid[index, 0]) for index in range(2)]
    plot_matrix(matrix_axes[0], coatjava_matrix, "COATJAVA, 0–1 GeV")
    image = plot_matrix(matrix_axes[1], fm_matrix, "Forward FM, 0–1 GeV")
    matrix_axes[0].text(
        -0.08,
        1.28,
        "4. Low-momentum positive-hadron response",
        transform=matrix_axes[0].transAxes,
        fontsize=13,
        color="purple",
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=matrix_axes, fraction=0.035, pad=0.03)
    colorbar.set_label("response probability")

    note_axis = figure.add_subplot(outer[2, :])
    note_axis.axis("off")
    note_axis.text(
        0.01,
        0.92,
        "Interpretation",
        fontsize=13,
        fontweight="bold",
        color="navy",
        va="top",
    )
    note_axis.text(
        0.01,
        0.73,
        "• The pi- conditional PID response closes well.\n"
        "• The FM underpredicts correct pi+ and proton response at low/intermediate momentum.\n"
        "• The largest low-momentum discrepancy is enhanced pi+ <-> proton migration.\n"
        "• K+ migration grows with momentum but does not explain the low-momentum deficit.\n"
        "• Matrices show only physical generated pi+ and proton rows; reconstructed classes are collapsed into an 'other' column.",
        fontsize=11,
        va="top",
        linespacing=1.45,
    )
    figure.suptitle(
        "PID response closure vs generated momentum\n"
        "Exact H100 checkpoint; held-out test split; fixed 1-GeV bins",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_total_variation(summary_rows: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    colors = {-211: "tab:blue", 211: "tab:orange", 2212: "tab:green"}
    for species in GENERATED_SPECIES:
        rows = [row for row in summary_rows if row["generated_pid"] == species]
        axis.plot(
            [row["p_center_gev"] for row in rows],
            [row["total_variation"] for row in rows],
            marker="o",
            color=colors[species],
            label=SPECIES_MATH[species],
        )
    axis.set(
        title="Full reconstructed-PID distribution discrepancy vs generated momentum",
        xlabel=r"generated momentum $p_{\mathrm{gen}}$ [GeV]",
        ylabel="Total-variation distance",
        xlim=(0.0, 9.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.25)
    axis.legend(title="generated species")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError(
            f"Checkpoint SHA-256 mismatch: {checkpoint_hash} != "
            f"{EXPECTED_CHECKPOINT_SHA256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["best_epoch"] != EXPECTED_BEST_EPOCH:
        raise AssertionError("Unexpected selected checkpoint epoch")
    if checkpoint["dataset_metadata_sha256"] != EXPECTED_DATASET_SHA256:
        raise AssertionError("Checkpoint records a different dataset fingerprint")

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if int(config["project"]["seed"]) != int(checkpoint["seed"]):
        raise AssertionError("Config seed does not match checkpoint seed")
    if selection_sql(config) != checkpoint["selection_sql"]:
        raise AssertionError("Config selection SQL does not match checkpoint")
    loaded_dataset_hash, dataset_records = dataset_metadata_sha256(
        config["data"]["parquet_glob"]
    )
    if loaded_dataset_hash != EXPECTED_DATASET_SHA256:
        raise AssertionError(
            f"Loaded dataset SHA-256 mismatch: {loaded_dataset_hash} != "
            f"{EXPECTED_DATASET_SHA256}"
        )

    connection = connect(config["data"]["parquet_glob"])
    assert_schema(connection)
    test_frame = _load_frame(connection, "test", config)
    connection.close()
    feature_scaler = Standardizer.from_dict(checkpoint["feature_scaler"])
    target_scaler = Standardizer.from_dict(checkpoint["target_scaler"])
    vocabulary = list(checkpoint["rec_pid_vocabulary"])
    test_split = prepare_split(
        test_frame,
        "test",
        feature_scaler,
        target_scaler,
        vocabulary,
    )
    if len(test_split) != EXPECTED_TEST_ROWS:
        raise AssertionError(f"Unexpected test row count: {len(test_split)}")
    frame_species = test_frame["gen_pid"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame_species, test_split.raw_species):
        raise AssertionError("Frame and prepared split row order are misaligned")

    device = choose_device(args.device)
    model = ConditionalMDN(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    probabilities = infer_pid_probabilities(
        model, test_split, config, checkpoint, device
    )
    if len(probabilities) != len(test_frame):
        raise AssertionError("Inference output is not aligned with test rows")

    observed_pid_index = test_split.rec_pid_index.astype(np.int64)
    generated_species = test_split.raw_species.astype(np.int64)
    generated_p = test_frame["gen_p"].to_numpy(dtype=np.float64)
    class_labels: list[int | str] = [*vocabulary, "OTHER"]
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    outside_bins = int(np.sum((generated_p < 0.0) | (generated_p > 9.0)))
    if outside_bins:
        raise AssertionError(f"{outside_bins} test rows lie outside fixed 0–9 GeV bins")

    response_rows, summary_rows = calculate_fixed_bin_response(
        generated_species,
        generated_p,
        observed_pid_index,
        probabilities,
        class_labels,
    )
    integrated_rows = calculate_integrated_response(
        generated_species,
        observed_pid_index,
        probabilities,
        class_to_index,
    )
    coatjava_matrix, fm_matrix, matrix_rows = calculate_low_momentum_matrices(
        generated_species,
        generated_p,
        observed_pid_index,
        probabilities,
        class_to_index,
    )

    write_csv(output_dir / "pid_response_fixed_bins.csv", response_rows)
    write_csv(output_dir / "pid_bin_closure_summary.csv", summary_rows)
    write_csv(output_dir / "pid_integrated_correct_id.csv", integrated_rows)
    write_csv(output_dir / "pid_low_momentum_matrices.csv", matrix_rows)
    plot_correct_id(
        response_rows,
        output_dir / "pid_truth_rate_vs_gen_p_reproduced.png",
    )
    plot_composite(
        response_rows,
        integrated_rows,
        coatjava_matrix,
        fm_matrix,
        output_dir / "PID_Truth_vs_momentum_reproduced.png",
    )
    plot_total_variation(
        summary_rows,
        output_dir / "pid_total_variation_vs_gen_p.png",
    )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "dataset_metadata_sha256": loaded_dataset_hash,
        "dataset_file_count": len(dataset_records),
        "test_rows": len(test_split),
        "split": "event-disjoint held-out test",
        "selection_sql": checkpoint["selection_sql"],
        "generated_species": list(GENERATED_SPECIES),
        "reconstructed_pid_classes": class_labels,
        "binning": "fixed equal-width",
        "bin_edges_gev": FIXED_P_EDGES_GEV.tolist(),
        "forward_fm_statistic": "mean softmax probability",
        "weights_changed": False,
        "device": str(device),
    }
    write_json(output_dir / "pid_validation_metadata.json", metadata)

    reference_integrated = {
        -211: (0.521, 0.520),
        211: (0.546, 0.424),
        2212: (0.809, 0.575),
    }
    comparison_rows = []
    for row in integrated_rows:
        species = int(row["generated_pid"])
        ref_coatjava, ref_fm = reference_integrated[species]
        comparison_rows.append(
            {
                "generated_pid": species,
                "computed_coatjava": row["coatjava_correct_fraction"],
                "attachment_coatjava_rounded": ref_coatjava,
                "computed_minus_attachment_coatjava": row["coatjava_correct_fraction"]
                - ref_coatjava,
                "computed_fm": row["fm_correct_probability"],
                "attachment_fm_rounded": ref_fm,
                "computed_minus_attachment_fm": row["fm_correct_probability"]
                - ref_fm,
            }
        )
    write_csv(output_dir / "attachment_integrated_comparison.csv", comparison_rows)

    worst_bins = sorted(
        summary_rows,
        key=lambda row: float(row["total_variation"]),
        reverse=True,
    )[:5]
    report_lines = [
        "# First-email PID validation reproduction",
        "",
        f"- Checkpoint SHA-256: `{checkpoint_hash}`",
        f"- Dataset metadata SHA-256: `{loaded_dataset_hash}`",
        f"- Best epoch: {checkpoint['best_epoch']}",
        f"- Held-out test rows: {len(test_split):,}",
        "- Momentum bins: fixed 1-GeV intervals from 0 to 9 GeV",
        "- Forward-FM statistic: mean softmax probability",
        "- Model weights changed: no",
        "",
        "## Momentum-integrated correct-ID response",
        "",
        "| Generated species | COATJAVA | Forward FM | FM - COATJAVA |",
        "|---|---:|---:|---:|",
    ]
    for row in integrated_rows:
        report_lines.append(
            f"| {row['generated_species']} | "
            f"{float(row['coatjava_correct_fraction']):.6f} | "
            f"{float(row['fm_correct_probability']):.6f} | "
            f"{float(row['fm_minus_coatjava']):+.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## Largest fixed-bin distribution discrepancies",
            "",
            "| Generated species | Momentum [GeV] | N | TV distance | Worst reconstructed PID | Max channel error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in worst_bins:
        report_lines.append(
            f"| {row['generated_species']} | "
            f"{float(row['p_low_gev']):.0f}–{float(row['p_high_gev']):.0f} | "
            f"{row['n']} | {float(row['total_variation']):.6f} | "
            f"{row['worst_reconstructed_pid']} | "
            f"{float(row['maximum_channel_error']):.6f} |"
        )
    report_lines.extend(
        [
            "",
            "The low-momentum matrices use only physical generated pi+ and proton rows. The `other` column is the sum of all reconstructed PID classes other than pi+, proton, and K+.",
            "",
        ]
    )
    (output_dir / "EMAIL1_VALIDATION_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("First-email PID validation complete")
    print(f"device={device}")
    print(f"test_rows={len(test_split)}")
    print(f"checkpoint_sha256={checkpoint_hash}")
    print(f"dataset_sha256={loaded_dataset_hash}")
    for row in integrated_rows:
        print(
            f"integrated {row['generated_species']}: "
            f"COATJAVA={float(row['coatjava_correct_fraction']):.6f} "
            f"FM={float(row['fm_correct_probability']):.6f}"
        )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
