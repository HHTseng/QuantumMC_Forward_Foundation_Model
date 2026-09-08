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
from forwardfm_step1.data import (
    BASE_CONTINUOUS_FEATURES,
    PARTICLE_MASS_GEV,
    SPECIES,
    data_order_seed,
    data_split_seed,
    load_all_splits,
)
from forwardfm_step1.evaluation import evaluate_and_write
from forwardfm_step1.model import (
    ConditionalMDN,
    count_parameters,
    initialize_beta_input_from_control,
)
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
    feature_names = splits["train"].feature_names
    target_names = splits["train"].target_names
    model_arguments = {
        "n_continuous": len(feature_names),
        "n_species": len(SPECIES),
        "n_rec_pid_classes": len(rec_pid_vocabulary) + 1,
        "hidden_width": int(model_config["hidden_width"]),
        "hidden_layers": int(model_config["hidden_layers"]),
        "pid_embedding_dim": int(model_config["pid_embedding_dim"]),
        "mixture_components": int(model_config["mixture_components"]),
        "target_dim": len(target_names),
        "dropout": float(model_config["dropout"]),
    }
    model = ConditionalMDN(**model_arguments)
    paired_initialization = bool(
        model_config.get("deterministic_component_initialization", False)
    )
    nested_beta_initialization = bool(
        model_config.get("nested_beta_input_initialization", False)
    )
    if paired_initialization:
        model.reset_parameters(seed=seed)
        if nested_beta_initialization and len(feature_names) > len(
            BASE_CONTINUOUS_FEATURES
        ):
            control_arguments = dict(model_arguments)
            control_arguments["n_continuous"] = len(BASE_CONTINUOUS_FEATURES)
            control_model = ConditionalMDN(**control_arguments)
            control_model.reset_parameters(seed=seed)
            initialize_beta_input_from_control(control_model, model)
        # Constructing a wider response head consumes more values from
        # PyTorch's global RNG.  Restore the model/training seed so paired
        # no-beta and joint-beta runs receive the same dropout stream; the
        # DataLoader already uses its own identically seeded generator.
        seed_everything(seed)
    if splits["train"].continuous.shape[1] != len(feature_names):
        raise AssertionError("Prepared feature width does not match feature metadata")
    if model.n_continuous != len(feature_names):
        raise AssertionError("Model input width does not match feature metadata")
    if model.target_dim != len(target_names):
        raise AssertionError("Model target width does not match target metadata")
    print(f"trainable_parameters={count_parameters(model):,}")
    model, history, best_epoch, checkpoint_candidates = train_model(
        model, splits, config, device
    )

    checkpoint_metric = str(config["training"].get("checkpoint_metric", "total_loss"))
    candidate_metadata = {
        name: {
            "epoch": int(candidate["epoch"]),
            "validation_value": float(candidate["validation_value"]),
        }
        for name, candidate in checkpoint_candidates.items()
    }

    checkpoint = {
        "format_version": 1,
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "architecture": model.architecture_dict(),
        "feature_names": list(feature_names),
        "target_names": list(target_names),
        "particle_mass_gev": dict(PARTICLE_MASS_GEV),
        "generated_beta_definition": "p/sqrt(p^2+m_species^2), c=1",
        "beta_response": config["data"].get("beta_response", {"enabled": False}),
        "species_pids": list(SPECIES),
        "rec_pid_vocabulary": rec_pid_vocabulary,
        "feature_scaler": feature_scaler.as_dict(),
        "target_scaler": target_scaler.as_dict(),
        "selection_sql": audit["selection_sql"],
        "dataset_metadata_sha256": audit["dataset_metadata_sha256"],
        "seed": seed,
        "data_split_seed": data_split_seed(config),
        "data_order_seed": data_order_seed(config),
        "initialization_policy": (
            "nested_zero_beta_column_from_four_input_control"
            if paired_initialization
            and nested_beta_initialization
            and len(feature_names) > len(BASE_CONTINUOUS_FEATURES)
            else "deterministic_component_streams_and_training_rng_reset"
            if paired_initialization
            else "legacy_global_stream"
        ),
        "best_epoch": best_epoch,
        "checkpoint_selection": {
            "primary_metric": checkpoint_metric,
            "primary_epoch": best_epoch,
            "candidates": candidate_metadata,
            "uses_test_data": False,
        },
    }
    torch.save(checkpoint, run_dir / "model.pt")
    for metric_name, candidate in checkpoint_candidates.items():
        candidate_checkpoint = dict(checkpoint)
        candidate_checkpoint["model_state"] = candidate["model_state"]
        candidate_checkpoint["best_epoch"] = int(candidate["epoch"])
        candidate_checkpoint["checkpoint_selection"] = {
            **checkpoint["checkpoint_selection"],
            "saved_candidate_metric": metric_name,
            "saved_candidate_epoch": int(candidate["epoch"]),
        }
        torch.save(
            candidate_checkpoint,
            run_dir / f"model_min_validation_{metric_name}.pt",
        )
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
