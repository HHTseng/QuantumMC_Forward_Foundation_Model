#!/usr/bin/env python3
"""Fit a hadron reconstruction gate on every generated hadron in T=1 events."""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch
import yaml

from forwardfm_hadron.data import OUTCOME_LABELS, SPECIES, load_hadron_splits
from forwardfm_hadron.evaluation import evaluate_and_write
from forwardfm_hadron.model import HadronEfficiencyNet
from forwardfm_hadron.training import fit
from forwardfm_step1.config import load_config, resolve_run_dir
from forwardfm_step1.reporting import write_json
from forwardfm_step1.training import choose_device, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hadron_efficiency.yaml")
    parser.add_argument("--species", required=True, type=int, choices=SPECIES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--parquet-glob")
    parser.add_argument("--run-dir")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["project"]["seed"] = args.seed
    if args.device:
        config["training"]["device"] = args.device
    if args.parquet_glob:
        config["data"]["parquet_glob"] = args.parquet_glob
    seed = int(config["project"]["seed"])
    config["species"] = args.species
    base = Path(config["output"]["run_dir"])
    config["output"]["run_dir"] = str(base / f"seed_{seed}" / f"pid_{args.species}")
    if args.smoke:
        config["data"]["max_rows"] = dict(train=2000, validation=500, test=500)
        config["model"].update(hidden_width=32, hidden_layers=2, dropout=0.0)
        config["training"].update(epochs=2, batch_size=1024)
        config["evaluation"]["min_bin_count"] = 10
        config["output"]["run_dir"] = f"runs/hadron_efficiency_smoke/pid_{args.species}"
    if args.run_dir:
        config["output"]["run_dir"] = args.run_dir
    run_dir = resolve_run_dir(config)
    if (run_dir / "model.pt").exists():
        raise FileExistsError(f"Refusing to replace existing checkpoint: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    device = choose_device(config["training"]["device"])
    print(f"species={args.species} seed={seed} device={device}", flush=True)
    splits, scaler, audit = load_hadron_splits(config, args.species)
    print({name: len(split) for name, split in splits.items()}, flush=True)
    write_json(audit, run_dir / "data_audit.json")
    model = HadronEfficiencyNet(n_continuous=splits["train"].continuous.shape[1], **config["model"])
    model, history, best_epoch = fit(model, splits, config, device)
    root = Path(__file__).resolve().parent
    source_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [root / "train_hadron_efficiency.py", root / "forwardfm_electron/data.py",
                     root / "forwardfm_step1/data.py", root / "forwardfm_step1/training.py",
                     *sorted((root / "forwardfm_hadron").glob("*.py"))]
    }
    checkpoint = dict(
        format_version=1, task="trigger_conditioned_hadron_efficiency", species=args.species,
        model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        architecture=model.architecture_dict(), feature_scaler=scaler.as_dict(),
        feature_names=audit["active_feature_names"],
        dropped_constant_feature_values=audit["dropped_constant_feature_values"],
        outcome_labels=list(OUTCOME_LABELS), conditioning=audit["conditioning"],
        seed=seed, split_seed=audit["split_seed"], best_epoch=best_epoch,
        dataset_metadata_sha256=audit["dataset_metadata_sha256"], source_hashes=source_hashes,
        label_definition=audit["label_definition"], denominator_sql=audit["denominator_sql"],
    )
    torch.save(checkpoint, run_dir / "model.pt")
    write_json(history, run_dir / "history.json")
    write_json(dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                    torch=torch.__version__, duckdb=duckdb.__version__, device=str(device),
                    gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    source_hashes=source_hashes,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()),
               run_dir / "environment.json")
    with (run_dir / "resolved_config.yaml").open("w") as stream:
        yaml.safe_dump({k: v for k, v in config.items() if not k.startswith("_")}, stream, sort_keys=False)
    metrics = evaluate_and_write(model, splits, history, config, device, run_dir)
    print(f"complete: {run_dir}; FD metrics={metrics['fd']}", flush=True)


if __name__ == "__main__":
    main()
