#!/usr/bin/env python3
"""Train eta_T(x_e)=P(T=1|x_e) on one generated electron per event."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from forwardfm_electron.data import (
    EFFICIENCY_CONTINUOUS_FEATURES,
    OUTCOME_LABELS,
    load_all_electron_splits,
)
from forwardfm_electron.evaluation import evaluate_and_write
from forwardfm_electron.model import ElectronEfficiencyNet, count_parameters
from forwardfm_electron.reporting import write_model_card
from forwardfm_electron.training import train_model
from forwardfm_step1.config import load_config, resolve_run_dir
from forwardfm_step1.reporting import write_history, write_json
from forwardfm_step1.training import choose_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/electron_efficiency_seed.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--device")
    return parser.parse_args()


def apply_smoke(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    config["data"]["max_rows"] = {"train": 20000, "validation": 5000, "test": 5000}
    config["model"].update({"hidden_width": 32, "hidden_layers": 2, "dropout": 0.0})
    config["training"].update({"epochs": 2, "batch_size": 1024, "early_stopping_patience": 2})
    config["evaluation"]["min_bin_count"] = 10
    config["output"]["run_dir"] = "runs/electron_efficiency_smoke"
    return config


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.smoke:
        config = apply_smoke(config)
    if args.run_dir:
        config["output"]["run_dir"] = args.run_dir
    if args.device:
        config["training"]["device"] = args.device
    run_dir = resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    torch.set_num_threads(int(config["training"].get("torch_threads", 4)))
    device = choose_device(str(config["training"]["device"]))
    print(f"device={device} run_dir={run_dir}")
    start = time.perf_counter()
    splits, feature_scaler, audit = load_all_electron_splits(config)
    print(
        "loaded " + ", ".join(f"{name}={len(split):,}" for name, split in splits.items())
        + f" in {time.perf_counter() - start:.2f}s"
    )
    model_config = config["model"]
    model = ElectronEfficiencyNet(
        n_continuous=splits["train"].continuous.shape[1],
        n_outcomes=len(OUTCOME_LABELS),
        hidden_width=int(model_config["hidden_width"]),
        hidden_layers=int(model_config["hidden_layers"]),
        dropout=float(model_config["dropout"]),
    )
    print(f"trainable_parameters={count_parameters(model):,}")
    model, history, best_epoch = train_model(model, splits, config, device)
    checkpoint = {
        "format_version": 1,
        "task": "trigger_electron_efficiency",
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "architecture": model.architecture_dict(),
        "feature_names": audit["active_feature_names"],
        "dropped_constant_feature_names": audit["dropped_constant_feature_names"],
        "label_definition": {
            "trigger": "has_valid_trigger_electron",
            "positive_association_invariant": "trigger_mcindex = mcindex",
            "outcome_classes": list(OUTCOME_LABELS),
        },
        "feature_scaler": feature_scaler.as_dict(),
        "denominator_sql": audit["denominator_sql"],
        "dataset_metadata_sha256": audit["dataset_metadata_sha256"],
        "seed": seed,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, run_dir / "model.pt")
    write_json(audit, run_dir / "data_audit.json")
    write_history(history, run_dir / "history.json")
    write_json(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "pid": os.getpid(),
        },
        run_dir / "environment.json",
    )
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable_config(config), handle, sort_keys=False)
    metrics = evaluate_and_write(model, splits, history, config, device, run_dir)
    write_model_card(
        run_dir / "MODEL_CARD.md", audit, metrics, best_epoch, count_parameters(model)
    )
    print(f"test_brier={metrics['trigger']['brier_score']:.6f}")
    print(f"test_ece={metrics['trigger']['expected_calibration_error']:.6f}")
    print(f"checkpoint={run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
