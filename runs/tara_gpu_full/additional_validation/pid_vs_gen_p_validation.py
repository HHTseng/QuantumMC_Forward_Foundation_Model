#!/usr/bin/env python3
"""Held-out reconstructed-PID validation versus generated momentum."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Allow direct execution from runs/tara_gpu_full/additional_validation after a
# fresh clone, without requiring callers to set PYTHONPATH manually.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from forwardfm_step1.data import (
    Standardizer,
    _load_frame,
    assert_schema,
    connect,
    prepare_split,
)
from forwardfm_step1.model import ConditionalMDN
from forwardfm_step1.training import make_loader


GENERATED_LABELS = {-211: r"$\pi^-$", 211: r"$\pi^+$", 2212: "proton"}
PID_LABELS = {
    -2212: r"$\bar{p}$",
    -321: r"$K^-$",
    -211: r"$\pi^-$",
    -11: r"$e^+$",
    11: r"$e^-$",
    22: r"$\gamma$",
    45: "nucleus",
    211: r"$\pi^+$",
    321: r"$K^+$",
    2112: "neutron",
    2212: "proton",
    "OTHER": "other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_scaler = Standardizer.from_dict(checkpoint["feature_scaler"])
    target_scaler = Standardizer.from_dict(checkpoint["target_scaler"])
    vocabulary = list(checkpoint["rec_pid_vocabulary"])

    connection = connect(config["data"]["parquet_glob"])
    assert_schema(connection)
    frame = _load_frame(connection, "test", config)
    connection.close()
    split = prepare_split(
        frame,
        "test",
        feature_scaler,
        target_scaler,
        vocabulary,
    )

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    model = ConditionalMDN(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    probabilities_batches: list[np.ndarray] = []
    loader = make_loader(
        split,
        int(config["training"]["batch_size"]),
        shuffle=False,
        seed=int(checkpoint["seed"]),
        num_workers=0,
    )
    with torch.no_grad():
        for continuous, species_index, _, _ in loader:
            output = model(continuous.to(device), species_index.to(device))
            probabilities_batches.append(
                torch.softmax(output.pid_logits, dim=-1).cpu().numpy()
            )
    probabilities = np.concatenate(probabilities_batches)
    observed = split.rec_pid_index.astype(np.int64)
    predicted = probabilities.argmax(axis=1)
    generated_p = frame["gen_p"].to_numpy(dtype=np.float64)
    generated_species = split.raw_species
    class_labels: list[int | str] = [*vocabulary, "OTHER"]

    performance_rows: list[dict[str, object]] = []
    response_rows: list[dict[str, object]] = []
    bin_records: dict[int, list[tuple[float, float, float, np.ndarray]]] = {}

    for species in sorted(GENERATED_LABELS):
        species_mask = generated_species == species
        edges = np.unique(
            np.quantile(generated_p[species_mask], np.linspace(0.0, 1.0, args.bins + 1))
        )
        records: list[tuple[float, float, float, np.ndarray]] = []
        for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            final_bin = bin_index == len(edges) - 2
            mask = species_mask & (generated_p >= low) & (
                (generated_p <= high) if final_bin else (generated_p < high)
            )
            indices = np.flatnonzero(mask)
            if not len(indices):
                continue
            center = float(np.median(generated_p[indices]))
            y = observed[indices]
            p = probabilities[indices]
            yhat = predicted[indices]
            accuracy = float(np.mean(yhat == y))
            cross_entropy = float(np.mean(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))
            one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y]
            brier = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
            confidence = float(np.mean(np.max(p, axis=1)))
            standard_error = float(np.sqrt(accuracy * (1.0 - accuracy) / len(indices)))
            performance_rows.append(
                {
                    "generated_pid": species,
                    "generated_species": GENERATED_LABELS[species].replace("$", ""),
                    "bin_index": bin_index,
                    "p_low_gev": float(low),
                    "p_high_gev": float(high),
                    "p_median_gev": center,
                    "n": len(indices),
                    "top1_accuracy": accuracy,
                    "accuracy_standard_error": standard_error,
                    "cross_entropy": cross_entropy,
                    "multiclass_brier_score": brier,
                    "mean_top1_confidence": confidence,
                    "confidence_minus_accuracy": confidence - accuracy,
                }
            )
            observed_fractions = np.bincount(y, minlength=len(class_labels)) / len(indices)
            predicted_fractions = p.mean(axis=0)
            for class_index, class_label in enumerate(class_labels):
                response_rows.append(
                    {
                        "generated_pid": species,
                        "generated_species": GENERATED_LABELS[species].replace("$", ""),
                        "bin_index": bin_index,
                        "p_low_gev": float(low),
                        "p_high_gev": float(high),
                        "p_median_gev": center,
                        "n": len(indices),
                        "reconstructed_pid": class_label,
                        "observed_fraction": float(observed_fractions[class_index]),
                        "mean_predicted_probability": float(predicted_fractions[class_index]),
                        "predicted_minus_observed": float(
                            predicted_fractions[class_index] - observed_fractions[class_index]
                        ),
                    }
                )
            records.append((float(low), float(high), center, indices))
        bin_records[species] = records

    performance_csv = output_dir / "pid_performance_vs_gen_p.csv"
    response_csv = output_dir / "pid_response_vs_gen_p.csv"
    write_csv(performance_csv, performance_rows)
    write_csv(response_csv, response_rows)

    colors = {-211: "tab:blue", 211: "tab:orange", 2212: "tab:green"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    metric_specs = [
        ("top1_accuracy", "Top-1 accuracy", (0.0, 1.0)),
        ("cross_entropy", "Cross-entropy", None),
        ("multiclass_brier_score", "Multiclass Brier score", None),
        ("mean_top1_confidence", "Mean top-1 confidence", (0.0, 1.0)),
    ]
    for axis, (field, ylabel, ylim) in zip(axes.flat, metric_specs):
        for species in sorted(GENERATED_LABELS):
            rows = [row for row in performance_rows if row["generated_pid"] == species]
            x = np.asarray([row["p_median_gev"] for row in rows])
            y = np.asarray([row[field] for row in rows])
            if field == "top1_accuracy":
                err = 1.96 * np.asarray([row["accuracy_standard_error"] for row in rows])
                axis.fill_between(x, y - err, y + err, color=colors[species], alpha=0.15)
            axis.plot(x, y, marker="o", color=colors[species], label=GENERATED_LABELS[species])
        axis.set(xlabel=r"generated momentum $p_{\mathrm{gen}}$ [GeV]", ylabel=ylabel)
        axis.grid(alpha=0.25)
        if ylim:
            axis.set_ylim(*ylim)
    axes[0, 0].legend(title="generated species")
    figure.suptitle("Held-out reconstructed-PID performance vs generated momentum\n"
                    "species-wise equal-population bins; shaded band is 95% binomial CI")
    figure.tight_layout()
    performance_png = output_dir / "pid_performance_vs_gen_p.png"
    figure.savefig(performance_png, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=False)
    for axis, species in zip(axes, sorted(GENERATED_LABELS)):
        species_indices = np.flatnonzero(generated_species == species)
        overall = np.bincount(observed[species_indices], minlength=len(class_labels))
        top_classes = np.argsort(overall)[::-1][:4]
        centers = np.asarray([record[2] for record in bin_records[species]])
        for rank, class_index in enumerate(top_classes):
            observed_values: list[float] = []
            predicted_values: list[float] = []
            for _, _, _, indices in bin_records[species]:
                observed_values.append(float(np.mean(observed[indices] == class_index)))
                predicted_values.append(float(np.mean(probabilities[indices, class_index])))
            color = plt.cm.tab10(rank)
            label = PID_LABELS.get(class_labels[class_index], str(class_labels[class_index]))
            axis.plot(centers, observed_values, marker="o", color=color, label=f"{label} observed")
            axis.plot(centers, predicted_values, linestyle="--", color=color, label=f"{label} predicted")
        axis.set(
            title=f"generated {GENERATED_LABELS[species]}",
            xlabel=r"generated momentum $p_{\mathrm{gen}}$ [GeV]",
            ylabel="reconstructed-PID fraction",
            ylim=(0.0, 1.0),
        )
        axis.grid(alpha=0.25)
        axis.legend(ncol=4, fontsize=8, loc="upper center")
    figure.suptitle("Held-out reconstructed-PID response vs generated momentum\n"
                    "solid: full simulation labels; dashed: mean network probability")
    figure.tight_layout()
    response_png = output_dir / "pid_response_vs_gen_p.png"
    figure.savefig(response_png, dpi=180, bbox_inches="tight")
    plt.close(figure)

    worst = min(performance_rows, key=lambda row: float(row["top1_accuracy"]))
    best = max(performance_rows, key=lambda row: float(row["top1_accuracy"]))
    checkpoint_digest = sha256(checkpoint_path)
    report = f"""# Additional PID validation versus generated momentum

- Checkpoint: `{checkpoint_path}`
- Checkpoint SHA-256: `{checkpoint_digest}`
- Checkpoint best epoch: {checkpoint['best_epoch']}
- Dataset metadata SHA-256 recorded in checkpoint: `{checkpoint['dataset_metadata_sha256']}`
- Evaluation split: event-disjoint held-out test split
- Test rows: {len(split):,}
- Binning: {args.bins} equal-population momentum bins, computed separately per generated species
- PID decision for accuracy: `argmax` of reconstructed-PID probabilities
- No checkpoint weights were changed during this evaluation.

## Reproduce after cloning

The source Parquet data are external to Git. Place
`phase-space_parquet-Aug17-26/particle_responses/*.parquet` next to the cloned
repository as documented in the root README, install `requirements.txt`, and
run from the repository root:

```bash
python runs/tara_gpu_full/additional_validation/pid_vs_gen_p_validation.py
```

Defaults use the committed full checkpoint, portable full-data config,
automatic CUDA/MPS/CPU selection, and this directory for outputs. Run with
`--help` to override them.

## Saved artifacts

- `pid_performance_vs_gen_p.png`
- `pid_response_vs_gen_p.png`
- `pid_performance_vs_gen_p.csv`
- `pid_response_vs_gen_p.csv`
- `pid_vs_gen_p_validation.py`

## Quick range check

- Lowest binned top-1 accuracy: {float(worst['top1_accuracy']):.4%} for generated PID {worst['generated_pid']}, {float(worst['p_low_gev']):.3f}–{float(worst['p_high_gev']):.3f} GeV (`n={worst['n']}`).
- Highest binned top-1 accuracy: {float(best['top1_accuracy']):.4%} for generated PID {best['generated_pid']}, {float(best['p_low_gev']):.3f}–{float(best['p_high_gev']):.3f} GeV (`n={best['n']}`).
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
