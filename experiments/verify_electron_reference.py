#!/usr/bin/env python3
"""Replay the frozen electron reference and verify its hadron-event subsets.

The electron denominator is every generated event. Each hadron denominator
is its T=1 subset. This check performs no training and does not change either
checkpoint: it requires identical dataset metadata and actual held-out keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forwardfm_electron.data import load_all_electron_splits
from forwardfm_electron.model import ElectronEfficiencyNet
from forwardfm_hadron.data import SPECIES
from forwardfm_step1.data import Standardizer
from forwardfm_step1.reporting import write_json
from forwardfm_step1.training import choose_device


def verify_reference(args: argparse.Namespace) -> dict:
    electron = args.electron_run.resolve()
    with (electron / "resolved_config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    config["data"]["parquet_glob"] = args.parquet_glob
    if any(value is not None for value in config["data"]["max_rows"].values()):
        raise ValueError("The electron reference must use all events, not a capped denominator")
    with (electron / "data_audit.json").open() as stream:
        original_audit = json.load(stream)
    with (electron / "metrics.json").open() as stream:
        original_metrics = json.load(stream)["trigger"]
    checkpoint_path = electron / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != "trigger_electron_efficiency":
        raise ValueError("The reference is not an electron-trigger efficiency checkpoint")
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported electron-reference checkpoint format")
    split_seed = int(config["project"]["seed"])
    if int(checkpoint["seed"]) != split_seed:
        raise ValueError("Electron configuration and checkpoint seeds differ")
    if checkpoint["label_definition"]["trigger"] != "has_valid_trigger_electron":
        raise ValueError("Unexpected electron-trigger label definition")

    print("Replaying the full all-event electron denominator (no retraining).", flush=True)
    splits, fresh_scaler, fresh_audit = load_all_electron_splits(config)
    fingerprint = fresh_audit["dataset_metadata_sha256"]
    if original_audit["dataset_metadata_sha256"] != fingerprint:
        raise ValueError("Fresh dataset fingerprint differs from the electron reference")
    if checkpoint["dataset_metadata_sha256"] != fingerprint:
        raise ValueError("Electron checkpoint dataset fingerprint differs from fresh data")
    if checkpoint["feature_names"] != fresh_audit["active_feature_names"]:
        raise ValueError("Electron checkpoint feature names/order differ from replay")
    old_scaler = Standardizer.from_dict(checkpoint["feature_scaler"])
    for name in ("mean", "scale"):
        if not np.allclose(getattr(old_scaler, name), getattr(fresh_scaler, name),
                           atol=1e-7, rtol=1e-6):
            raise ValueError(f"Electron checkpoint and replay training scaler {name} differ")
    split_counts = {}
    for name in ("train", "validation", "test"):
        fresh = fresh_audit["split_counts"][name]
        old = original_audit["split_counts"][name]
        for key in ("rows", "trigger_successes", "trigger_failures"):
            if int(fresh[key]) != int(old[key]):
                raise ValueError(f"Electron-reference split counts changed: {name}/{key}")
        split_counts[name] = {key: int(fresh[key]) for key in
                              ("rows", "trigger_successes", "trigger_failures")}
    test = splits["test"]
    triggered_keys = np.sort(test.event_keys[test.trigger_target > 0.5].astype(np.uint64))
    if len(np.unique(triggered_keys)) != len(triggered_keys):
        raise ValueError("Duplicate electron-trigger test keys")
    subset_hash = hashlib.sha256(triggered_keys.astype("<u8").tobytes()).hexdigest()
    matching_runs = []
    for seed in (20260822, 20260823, 20260824):
        for species in SPECIES:
            directory = args.root / f"seed_{seed}" / f"pid_{species}"
            with (directory / "data_audit.json").open() as stream:
                audit = json.load(stream)
            if int(audit["species"]) != species:
                raise ValueError(f"Hadron species metadata disagrees with {directory}")
            if audit["dataset_metadata_sha256"] != fingerprint:
                raise ValueError(f"Hadron dataset fingerprint differs: {directory}")
            if int(audit["split_seed"]) != split_seed:
                raise ValueError(f"Hadron/electron split seeds differ: {directory}")
            if audit["conditioning"] != "has_valid_trigger_electron = true":
                raise ValueError(f"Unexpected hadron conditioning: {directory}")
            for name in ("train", "validation", "test"):
                if int(audit["split_counts"][name]["rows"]) != split_counts[name]["trigger_successes"]:
                    raise ValueError(f"Hadron denominator differs from electron T=1 count: {directory}/{name}")
            with np.load(directory / "predictions.npz") as prediction:
                keys = np.sort(prediction["event_keys"].astype(np.uint64))
            if not np.array_equal(keys, triggered_keys):
                raise ValueError(f"Hadron held-out event keys differ from electron T=1 subset: {directory}")
            matching_runs.append({
                "network_seed": seed, "species": species, "test_rows": len(keys),
                "test_event_keys_match": True, "all_split_counts_match": True,
            })

    device = choose_device(args.device)
    torch.set_num_threads(4)
    model = ElectronEfficiencyNet(**checkpoint["architecture"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    # Use exactly the checkpoint's coordinate transform. If replay statistics
    # are bit-identical, retain the loader's original float32 features; otherwise
    # undo its affine map before applying the saved one.
    same_scaler = all(np.array_equal(getattr(old_scaler, name), getattr(fresh_scaler, name))
                      for name in ("mean", "scale"))
    features = test.continuous if same_scaler else old_scaler.transform(
        fresh_scaler.inverse(test.continuous)
    )
    batch_size = int(config["training"]["batch_size"])
    probability = []
    with torch.no_grad():
        for start in range(0, len(test), batch_size):
            output = model(torch.from_numpy(features[start:start + batch_size]).to(device))
            probability.append(torch.sigmoid(output.trigger_logit).cpu().numpy())
    probability = np.concatenate(probability).astype(np.float64)
    if not np.isfinite(probability).all():
        raise FloatingPointError("Nonfinite frozen-electron predictions")
    target = test.trigger_target.astype(np.float64)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    replayed_metrics = {
        "mean_predicted_probability": float(probability.mean()),
        "brier_score": float(np.mean((probability - target) ** 2)),
        "log_loss": float(-np.mean(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))),
    }
    tolerance = 1e-5
    differences = {}
    for name, replayed in replayed_metrics.items():
        original = float(original_metrics[name])
        difference = replayed - original
        differences[name] = {"original": original, "replayed": replayed,
                             "signed_difference": difference,
                             "absolute_difference": abs(difference)}
        if abs(difference) > tolerance:
            raise ValueError(f"Frozen electron metric failed replay tolerance: {name}, difference={difference}")
    return {
        "checked": True, "retrained": False,
        "electron_reference": str(electron), "device": str(device),
        "dataset_metadata_sha256": fingerprint,
        "dataset_file_count": int(fresh_audit["dataset_file_count"]),
        "dataset_total_bytes": int(fresh_audit["dataset_total_bytes"]),
        "split_seed": split_seed, "split_counts": split_counts,
        "triggered_test_subset_sha256": subset_hash,
        "electron_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "feature_names": checkpoint["feature_names"],
        "training_scaler_matches": True, "training_scaler_bit_identical": same_scaler,
        "matching_run_count": len(matching_runs), "matching_runs": matching_runs,
        "metric_absolute_tolerance": tolerance, "metric_differences": differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/hadron_efficiency"))
    parser.add_argument("--electron-run", type=Path, default=Path("runs/gpu_electron_efficiency_full"))
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    destination = args.root / "summary" / "electron_reference_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = verify_reference(args)
    except Exception as error:
        # Never leave an old successful report behind after a failed replay.
        write_json({"checked": False, "error": f"{type(error).__name__}: {error}"}, destination)
        raise
    write_json(report, destination)
    print(f"Verified frozen electron metrics and all {report['matching_run_count']} hadron runs: {destination}",
          flush=True)


if __name__ == "__main__":
    main()
