#!/usr/bin/env python3
r"""Train the first detector-surrogate factor, \(P(T=1\mid x_e)\)."""

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

from forwardfm_full.data import TRIGGER_FEATURES, load_trigger_splits
from forwardfm_full.efficiency import TriggerEfficiencyNet, count_parameters, train_model
from forwardfm_full.evaluation import evaluate_and_write
from forwardfm_step1.config import load_config, resolve_run_dir
from forwardfm_step1.data import data_order_seed, data_split_seed
from forwardfm_step1.reporting import write_history, write_json
from forwardfm_step1.training import choose_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/trigger_electron_efficiency.yaml"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--device")
    return parser.parse_args()


def apply_smoke(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    config["data"]["max_rows"] = {
        "train": 20_000,
        "validation": 5_000,
        "test": 5_000,
    }
    config["model"].update(
        {"hidden_width": 32, "hidden_layers": 2, "dropout": 0.0}
    )
    config["training"].update(
        {
            "epochs": 2,
            "batch_size": 1024,
            "early_stopping_patience": 2,
            "preload_to_device": False,
        }
    )
    config["evaluation"].update(
        {"min_bin_count": 10, "min_bin_count_2d": 10}
    )
    config["output"]["run_dir"] = "runs/trigger_efficiency_smoke"
    return config


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
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    device = choose_device(str(config["training"]["device"]))
    print(f"device={device} run_dir={run_dir}")
    start = time.perf_counter()
    splits, scaler, audit = load_trigger_splits(config)
    print(
        "loaded "
        + ", ".join(f"{name}={len(split):,}" for name, split in splits.items())
        + f" in {time.perf_counter() - start:.2f}s"
    )

    model_config = config["model"]
    model = TriggerEfficiencyNet(
        n_continuous=len(TRIGGER_FEATURES),
        hidden_width=int(model_config["hidden_width"]),
        hidden_layers=int(model_config["hidden_layers"]),
        dropout=float(model_config["dropout"]),
    )
    print(f"trainable_parameters={count_parameters(model):,}")
    model, history, best_epoch, best_value = train_model(
        model, splits, config, device
    )
    checkpoint = {
        "format_version": 1,
        "task": "trigger_electron_efficiency",
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "architecture": model.architecture_dict(),
        "feature_names": list(TRIGGER_FEATURES),
        "feature_scaler": scaler.as_dict(),
        "denominator_sql": audit["denominator_sql"],
        "label_definition": {
            "T": "has_valid_trigger_electron",
            "R_e": "reconstructed AND matched_pindex >= 0",
            "association_invariant": "trigger_mcindex = mcindex when T=1",
        },
        "dataset_metadata_sha256": audit["dataset_metadata_sha256"],
        "seed": seed,
        "data_split_seed": data_split_seed(config),
        "data_order_seed": data_order_seed(config),
        "best_epoch": best_epoch,
        "checkpoint_selection": {
            "metric": "validation binary cross-entropy",
            "value": best_value,
            "uses_test_data": False,
        },
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
        yaml.safe_dump(
            {key: value for key, value in config.items() if not key.startswith("_")},
            handle,
            sort_keys=False,
        )
    metrics = evaluate_and_write(model, splits, history, config, device, run_dir)
    test = metrics["test"]
    model_card = rf"""# Trigger-electron efficiency model

The checkpoint approximates

$$
\widehat\epsilon_e(x_e)\simeq P(T=1\mid x_e).
$$

- denominator: `{audit['denominator_sql']}` ({audit['global_counts']['generated_electrons']:,} events);
- features: `{', '.join(TRIGGER_FEATURES)}`;
- label: `has_valid_trigger_electron`;
- selection: epoch {best_epoch}, minimum validation BCE;
- test rows: {test['n']:,};
- observed/predicted rate: {test['observed_trigger_rate']:.6f}/{test['mean_predicted_probability']:.6f};
- BCE/Brier/ECE: {test['binary_cross_entropy']:.6f}/{test['brier_score']:.6f}/{test['expected_calibration_error']:.6f}.
"""
    (run_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    print(f"test_brier={test['brier_score']:.6f}")
    print(f"test_ece={test['expected_calibration_error']:.6f}")
    print(f"checkpoint={run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
