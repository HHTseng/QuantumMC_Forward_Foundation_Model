#!/usr/bin/env python3
"""Compare two PID heads on the beta baseline's exact held-out particles.

This isolates network changes from the small change in selected rows caused by
the explicit beta-validity domain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from forwardfm_step1.config import load_config  # noqa: E402
from forwardfm_step1.data import Standardizer, load_all_splits  # noqa: E402
from forwardfm_step1.evaluation import (  # noqa: E402
    _raw_kinematics,
    conditional_pid_response_rows,
    integrated_correct_pid_response,
)
from forwardfm_step1.model import ConditionalMDN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-checkpoint",
        default="runs/tara_gpu_full/model.pt",
        help="Original three-target checkpoint to compare",
    )
    parser.add_argument(
        "--beta-config",
        default="configs/gpu_beta_baseline.yaml",
        help="Config defining the beta-valid test population",
    )
    parser.add_argument(
        "--beta-metrics",
        default="runs/gpu_beta_baseline/metrics.json",
        help="Completed beta-run metrics",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/gpu_beta_baseline",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    original_path = (REPOSITORY_ROOT / args.original_checkpoint).resolve()
    config = load_config(REPOSITORY_ROOT / args.beta_config)
    beta_metrics_path = (REPOSITORY_ROOT / args.beta_metrics).resolve()
    output_dir = (REPOSITORY_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits, beta_feature_scaler, _, beta_vocabulary, _ = load_all_splits(config)
    test = splits["test"]
    checkpoint = torch.load(original_path, map_location="cpu", weights_only=False)
    if checkpoint["rec_pid_vocabulary"] != beta_vocabulary:
        raise RuntimeError("PID vocabularies differ; direct class comparison is invalid")

    original_scaler = Standardizer.from_dict(checkpoint["feature_scaler"])
    original_features = original_scaler.transform(
        beta_feature_scaler.inverse(test.continuous)
    )
    device = choose_device(args.device)
    model = ConditionalMDN(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    probability_batches = []
    with torch.no_grad():
        for start in range(0, len(test), 8192):
            stop = min(start + 8192, len(test))
            output = model(
                torch.from_numpy(original_features[start:stop]).to(device),
                torch.from_numpy(test.species_index[start:stop]).to(device),
            )
            probability_batches.append(
                torch.softmax(output.pid_logits, dim=-1).cpu().numpy()
            )
    probabilities = np.concatenate(probability_batches)
    labels: list[int | str] = [*beta_vocabulary, "OTHER"]
    generated_momentum = _raw_kinematics(test, beta_feature_scaler)["gen_p"]
    _, original_summaries = conditional_pid_response_rows(
        test.raw_species,
        generated_momentum,
        test.rec_pid_index,
        probabilities,
        labels,
        np.asarray(config["evaluation"]["pid_momentum_edges_gev"]),
    )
    original_integrated = integrated_correct_pid_response(
        test.raw_species,
        test.rec_pid_index,
        probabilities,
        labels,
    )

    beta_metrics = json.loads(beta_metrics_path.read_text(encoding="utf-8"))
    beta_integrated = beta_metrics["pid_conditional_closure"]["integrated_correct_id"]
    beta_summaries = beta_metrics["pid_conditional_closure"]["bin_summary"]
    beta_by_pid = {row["generated_pid"]: row for row in beta_integrated}
    comparison_rows = []
    for original in original_integrated:
        beta = beta_by_pid[original["generated_pid"]]
        observed = original["coatjava_correct_fraction"]
        comparison_rows.append(
            {
                "generated_pid": original["generated_pid"],
                "generated_species": original["generated_species"],
                "n": original["n"],
                "coatjava_correct_fraction": observed,
                "original_fm_mean_probability": original["fm_correct_mean_probability"],
                "beta_fm_mean_probability": beta["fm_correct_mean_probability"],
                "original_absolute_error": abs(
                    original["fm_correct_mean_probability"] - observed
                ),
                "beta_absolute_error": abs(beta["fm_correct_mean_probability"] - observed),
            }
        )

    csv_path = output_dir / "pid_same_test_checkpoint_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    payload = {
        "definition": (
            "Both PID heads evaluated on the beta baseline's exact held-out rows; "
            "teacher fractions are identical and predictions are mean softmax probabilities."
        ),
        "test_rows": len(test),
        "original_checkpoint": {
            "path": str(original_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256(original_path),
        },
        "beta_metrics": str(beta_metrics_path.relative_to(REPOSITORY_ROOT)),
        "integrated_correct_id": comparison_rows,
        "original_worst_total_variation_bin": max(
            original_summaries, key=lambda row: row["total_variation_distance"]
        ),
        "beta_worst_total_variation_bin": max(
            beta_summaries, key=lambda row: row["total_variation_distance"]
        ),
    }
    json_path = output_dir / "pid_same_test_checkpoint_comparison.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
