#!/usr/bin/env python3
"""Train P(Delta,s_rec | x,T=1,C=FD) from the selected Forward Detector data.

In the planned full detector surrogate this is the final conditional factor:

    P(Y|X) = P(T|x_e) prod_i P(C_i|x_i,T)
             P(Delta_i,s_rec,i|x_i,T,C_i).

This executable trains the last term for hadrons with C_i=FD. The earlier
trigger and reconstruction-region factors intentionally remain future heads.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from forwardfm_step1.config import apply_smoke_overrides, load_config, resolve_run_dir
from forwardfm_step1.data import CONTINUOUS_FEATURES, SPECIES, TARGET_COLUMNS, load_all_splits
from forwardfm_step1.evaluation import evaluate_and_write
from forwardfm_step1.model import ConditionalMDN, count_parameters
from forwardfm_step1.reporting import write_history, write_json, write_model_card
from forwardfm_step1.training import choose_device, seed_everything, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the step-one stochastic CLAS12 Forward Detector response model."
    )
    parser.add_argument(
        "--config", default="configs/fd_response_seed.yaml", help="YAML configuration path"
    )
    parser.add_argument("--smoke", action="store_true", help="Use a tiny sample and two epochs")
    parser.add_argument("--run-dir", help="Override the configured output directory")
    parser.add_argument("--device", help="Override auto/cpu/cuda/mps device selection")
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the configured seed, for seed-sensitivity repeats",
    )
    parser.add_argument(
        "--init-from",
        help="Warm start from a saved checkpoint's weights, for two-phase "
        "training. The checkpoint must come from the same data split, which is "
        "verified against its stored scalers.",
    )
    return parser.parse_args()


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def environment_manifest(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "duckdb_note": "Version recorded by pip/conda environment; see requirements.txt",
        "device": str(device),
        "pid": os.getpid(),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.smoke:
        config = apply_smoke_overrides(config)
    if args.run_dir:
        config["output"]["run_dir"] = args.run_dir
    if args.device:
        config["training"]["device"] = args.device
    if args.seed is not None:
        config["project"]["seed"] = args.seed
    run_dir = resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["project"]["seed"])
    seed_everything(seed)
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    device = choose_device(str(config["training"]["device"]))
    print(f"device={device} run_dir={run_dir}")
    print("loading deterministic, event-disjoint data splits...")
    start = time.perf_counter()
    splits, feature_scaler, target_scaler, rec_pid_vocabulary, audit = load_all_splits(config)
    print(
        "loaded "
        + ", ".join(f"{name}={len(split):,}" for name, split in splits.items())
        + f" in {time.perf_counter() - start:.2f}s"
    )

    model_config = config["model"]
    model = ConditionalMDN(
        n_continuous=len(CONTINUOUS_FEATURES),
        n_species=len(SPECIES),
        n_rec_pid_classes=len(rec_pid_vocabulary) + 1,
        hidden_width=int(model_config["hidden_width"]),
        hidden_layers=int(model_config["hidden_layers"]),
        pid_embedding_dim=int(model_config["pid_embedding_dim"]),
        mixture_components=int(model_config["mixture_components"]),
        target_dim=len(TARGET_COLUMNS),
        dropout=float(model_config["dropout"]),
    )
    if args.init_from:
        initial = torch.load(args.init_from, map_location="cpu", weights_only=False)
        # A checkpoint fitted on a different split carries different feature and
        # target standardizations, and warm starting across them would silently
        # train on a mismatched coordinate system and leak the other split's
        # training rows into this one's test set.
        for name, saved, current in (
            ("feature_scaler", initial["feature_scaler"], feature_scaler.as_dict()),
            ("target_scaler", initial["target_scaler"], target_scaler.as_dict()),
        ):
            if not np.allclose(saved["mean"], current["mean"], rtol=1e-5, atol=1e-8) or (
                not np.allclose(saved["scale"], current["scale"], rtol=1e-5, atol=1e-8)
            ):
                raise SystemExit(
                    f"--init-from checkpoint has a different {name}; it was fitted "
                    "on another split and must not be used to warm start this run"
                )
        if initial["architecture"] != model.architecture_dict():
            raise SystemExit("--init-from checkpoint has a different architecture")
        model.load_state_dict(initial["model_state"])
        print(f"warm started from {args.init_from} (epoch {initial['best_epoch']})")

    print(f"trainable_parameters={count_parameters(model):,}")
    model, history, best_epoch = train_model(model, splits, config, device)

    checkpoint = {
        "format_version": 1,
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "architecture": model.architecture_dict(),
        "feature_names": list(CONTINUOUS_FEATURES),
        "target_names": list(TARGET_COLUMNS),
        "species_pids": list(SPECIES),
        "rec_pid_vocabulary": rec_pid_vocabulary,
        "feature_scaler": feature_scaler.as_dict(),
        "target_scaler": target_scaler.as_dict(),
        "selection_sql": audit["selection_sql"],
        "dataset_metadata_sha256": audit["dataset_metadata_sha256"],
        "seed": seed,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, run_dir / "model.pt")
    write_json(audit, run_dir / "data_audit.json")
    write_history(history, run_dir / "history.json")
    write_json(environment_manifest(device), run_dir / "environment.json")
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable_config(config), handle, sort_keys=False)

    metrics = evaluate_and_write(
        model,
        splits,
        feature_scaler,
        target_scaler,
        rec_pid_vocabulary,
        history,
        config,
        device,
        run_dir,
    )
    write_model_card(
        run_dir / "MODEL_CARD.md",
        config,
        audit,
        metrics,
        best_epoch,
        count_parameters(model),
    )
    print(f"test_nll={metrics['test']['residual_nll']:.6f}")
    print(f"test_pid_accuracy={metrics['test']['pid_accuracy']:.4f}")
    print(f"checkpoint={run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
