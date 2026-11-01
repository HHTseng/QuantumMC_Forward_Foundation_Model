#!/usr/bin/env python3
"""Verify private transfer checkpoints/predictions without publishing events.

The aggregate JSON and SHA-256 inventory contain no event-level quantities.
CPU replay checks saved GPU probabilities on deterministic truth-row samples;
it does not refit a scaler or retrain a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from forwardfm_electron.data import EFFICIENCY_CONTINUOUS_FEATURES, OUTCOME_LABELS, feature_matrix
from forwardfm_hadron.inference import predict_checkpoint
from forwardfm_hadron.model import HadronEfficiencyNet
from forwardfm_step1.data import Standardizer


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(str(values.dtype).encode() + str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/rho_transfer")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-rows", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--expected-response-runs", type=int, default=14)
    args = parser.parse_args()
    if args.replay_rows < 1 or args.atol <= 0:
        parser.error("Replay row count and tolerance must be positive")
    base = args.runs_dir.resolve()
    output = args.output or base / "artifact_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    report = {"status": "running", "torch_version": str(torch.__version__),
              "probability_absolute_tolerance": args.atol, "replay_rows_per_dataset": args.replay_rows,
              "checkpoint_records": [], "response_records": [], "errors": []}
    source_cache, references, scaler_references, private = {}, {}, {}, []
    checkpoints = sorted(path for host in ("tara", "dgx_spark1")
                         for path in (base / host / "efficiency").glob("seed_*/pid_*/model.pt"))
    if len(checkpoints) != 18:
        report["errors"].append(f"Expected 18 gate checkpoints, found {len(checkpoints)}")
    for path in checkpoints:
        label = str(path.relative_to(base))
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            species = int(checkpoint["species"])
            if checkpoint["outcome_labels"] != list(OUTCOME_LABELS):
                raise AssertionError("Outcome vocabulary/order mismatch")
            if checkpoint["conditioning"] != "has_valid_trigger_electron = true":
                raise AssertionError("Trigger conditioning mismatch")
            if checkpoint["split_seed"] != 20261101:
                raise AssertionError("Unexpected fixed data split seed")
            model = HadronEfficiencyNet(**checkpoint["architecture"]).eval()
            model.load_state_dict(checkpoint["model_state"], strict=True)
            scaler = Standardizer.from_dict(checkpoint["feature_scaler"])
            if (len(scaler.mean) != len(checkpoint["feature_names"]) or
                    not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.scale).all() or
                    np.any(scaler.scale <= 0)):
                raise AssertionError("Invalid saved feature scaler")
            scaler_fingerprint = json.dumps(checkpoint["feature_scaler"], sort_keys=True)
            if species in scaler_references and scaler_references[species] != scaler_fingerprint:
                raise AssertionError("Training scaler differs across paired seeds/hardware")
            scaler_references[species] = scaler_fingerprint
            history = json.loads((path.parent / "history.json").read_text())
            config = json.loads((path.parents[1] / "resolved_config.json").read_text())
            audit = json.loads((path.parent / "data_audit.json").read_text())
            environment = json.loads((path.parents[1] / "environment.json").read_text())
            best_epoch = int(checkpoint["best_epoch"])
            losses = np.asarray([row["validation_loss"] for row in history], float)
            if (not np.isfinite(losses).all() or not 1 <= best_epoch <= len(history) <= 30 or
                    losses[best_epoch - 1] > losses.min() + 1.01e-6):
                raise AssertionError("Checkpoint is not the validation-best recorded epoch")
            if len(history) > int(config["epochs"]):
                raise AssertionError("History exceeds configured epoch budget")
            for name, saved in checkpoint["source_hashes"].items():
                if environment["source_hashes"].get(name) != saved:
                    raise AssertionError(f"Checkpoint/environment source hash disagreement: {name}")
                if name not in source_cache:
                    source_cache[name] = file_sha256(ROOT / name)
                if source_cache[name] != saved:
                    raise AssertionError(f"Live scientific source differs from training snapshot: {name}")
            record = {"checkpoint": label, "sha256": file_sha256(path), "species": species,
                      "seed": int(checkpoint["seed"]), "epochs": len(history), "best_epoch": best_epoch,
                      "validation_minimum_loss": float(losses.min()), "source_hashes_match": True,
                      "parameter_count": sum(p.numel() for p in model.parameters()), "datasets": []}
            private.append(path)
            for dataset, split_name in (("phase_test", "test"), ("rho_test", "rho")):
                prediction_path = path.parent / dataset / "predictions.npz"
                with np.load(prediction_path, allow_pickle=False) as predictions:
                    y, p, keys = (predictions[name] for name in
                                  ("outcome_target", "outcome_probability", "event_keys"))
                    n = len(y)
                    if n != int(config["max_test"]) or n != audit["splits"][split_name]["rows"]:
                        raise AssertionError(f"Unexpected denominator: {dataset}, n={n}")
                    if len(np.unique(keys)) != n:
                        raise AssertionError("Repeated event keys in generated-species denominator")
                    if p.shape != (n, 5) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                        raise AssertionError("Invalid 5-outcome probabilities")
                    if not np.allclose(p.sum(axis=1), 1, atol=2e-6, rtol=0):
                        raise AssertionError("Outcome probabilities do not sum to 1")
                    if np.any(p[:, 1] > 1 - p[:, 0] + 2e-6):
                        raise AssertionError("FD probability exceeds any-region reconstruction")
                    if not np.isin(y, np.arange(5)).all():
                        raise AssertionError("Outcome targets outside vocabulary")
                    event_order_sha = hashlib.sha256(keys.astype("<u8").tobytes()).hexdigest()
                    if event_order_sha != audit["splits"][split_name]["event_order_sha256"]:
                        raise AssertionError("Prediction event order differs from loader audit")
                    teacher_fingerprints = {name: array_sha256(predictions[name]) for name in
                                            ("event_keys", "outcome_target", "gen_p", "gen_theta", "gen_phi", "gen_vz")}
                    reference_key = (species, dataset)
                    if reference_key in references and teacher_fingerprints != references[reference_key]:
                        raise AssertionError("Teacher/events differ across paired seeds or platforms")
                    references[reference_key] = teacher_fingerprints
                    selected = np.linspace(0, n - 1, min(n, args.replay_rows), dtype=int)
                    frame = pd.DataFrame({name: predictions[name][selected]
                                          for name in ("gen_p", "gen_theta", "gen_phi", "gen_vz")})
                    frame["gen_pid"] = species
                    for name, constant in checkpoint["dropped_constant_feature_values"].items():
                        if name in ("gen_vx", "gen_vy"):
                            frame[name] = constant
                    feature_indices = [EFFICIENCY_CONTINUOUS_FEATURES.index(name)
                                       for name in checkpoint["feature_names"]]
                    scaled = scaler.transform(feature_matrix(frame)[:, feature_indices])
                    with torch.no_grad():
                        direct = model(torch.from_numpy(scaled)).softmax(-1).numpy()
                    replay = predict_checkpoint(path, frame, device="cpu")
                    cpu_probability = replay[[f"prob_{name}" for name in OUTCOME_LABELS]].to_numpy()
                    if not np.allclose(direct, cpu_probability, atol=1e-7, rtol=0):
                        raise AssertionError("Public inference differs from direct saved-scaler reconstruction")
                    maximum_difference = float(np.max(np.abs(cpu_probability - p[selected])))
                    if maximum_difference > args.atol:
                        raise AssertionError(f"CPU/GPU replay difference exceeds tolerance: {maximum_difference}")
                    metrics = json.loads((prediction_path.parent / "metrics.json").read_text())
                    for target, observed, probability in (("fd", y == 1, p[:, 1]),
                                                           ("any", y != 0, 1 - p[:, 0])):
                        if not np.isclose(observed.mean(), metrics[target]["observed_rate"], atol=1e-10):
                            raise AssertionError("Saved observed efficiency disagrees with event labels")
                        brier = float(np.mean((observed.astype(float) - probability.astype(float)) ** 2))
                        if not np.isclose(brier, metrics[target]["brier_score"], atol=1e-10):
                            raise AssertionError("Saved Brier score disagrees with event predictions")
                    record["datasets"].append({"dataset": dataset, "n": n,
                                                "cpu_replay_rows": len(selected),
                                                "max_abs_cpu_gpu_probability_difference": maximum_difference,
                                                "event_order_sha256": event_order_sha})
                private.append(prediction_path)
            report["checkpoint_records"].append(record)
        except Exception as error:
            report["errors"].append(f"{label}: {type(error).__name__}: {error}")
    response_paths = sorted(path for host in ("tara", "dgx_spark1")
                            for path in (base / host / "response").glob("seed_*/*/response_events.npz"))
    if len(response_paths) != args.expected_response_runs:
        report["errors"].append(f"Expected {args.expected_response_runs} response predictions, found {len(response_paths)}")
    for path in response_paths:
        try:
            with np.load(path, allow_pickle=False) as values:
                probabilities = values["pid_probabilities"]
                n = len(values["species"])
                if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-6):
                    raise AssertionError("Invalid direct PID probability normalization")
                if probabilities.shape[1] != len(values["pid_vocabulary"]):
                    raise AssertionError("PID probability/vocabulary dimension mismatch")
                if not np.array_equal(values["raw_truth"][:, 1:], values["corrected_truth"][:, 1:], equal_nan=True):
                    raise AssertionError("Momentum correction changed angular/beta targets")
                unchanged = values["was_corrected"] == 0
                if not np.array_equal(values["raw_truth"][unchanged], values["corrected_truth"][unchanged], equal_nan=True):
                    raise AssertionError("Uncorrected rows have changed response targets")
                keys = np.column_stack([values[name] for name in ("file_id", "event_id", "mcindex")])
                if len(np.unique(keys, axis=0)) != n:
                    raise AssertionError("Repeated generated-particle keys in response sample")
                report["response_records"].append({"predictions": str(path.relative_to(base)), "n": n,
                                                    "finite_sampled_response_fraction": float(np.isfinite(values["sampled_response"]).all(1).mean())})
            private.append(path)
        except Exception as error:
            report["errors"].append(f"{path.relative_to(base)}: {type(error).__name__}: {error}")
    inventory = [{"path": str(path.relative_to(base)), "bytes": path.stat().st_size,
                  "sha256": file_sha256(path)} for path in sorted(set(private))]
    inventory_path = base / "private_artifact_sha256.csv"
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory)
    report.update(status="pass" if not report["errors"] else "fail",
                  checkpoints_verified=len(report["checkpoint_records"]),
                  response_predictions_verified=len(report["response_records"]),
                  private_artifact_files=len(inventory), private_artifact_bytes=sum(r["bytes"] for r in inventory),
                  inventory_sha256=file_sha256(inventory_path),
                  publication_note="Only aggregate verification and checksums; model/event artifacts remain private.")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "checkpoints_verified", "response_predictions_verified", "private_artifact_files", "errors")}, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
