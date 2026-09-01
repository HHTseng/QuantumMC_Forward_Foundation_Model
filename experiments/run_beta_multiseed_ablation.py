#!/usr/bin/env python3
"""Run the controlled ten-pair no-beta versus joint-Delta-beta ablation.

The two configurations use one fixed event split and query order.  For each
model seed, both variants receive the same shuffled batches and component-wise
initial weights for all shared and PID modules.  The treatment is enabling the
fourth continuous response target Delta-beta; beta-validity row selection is
held fixed in both variants.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = tuple(range(20260822, 20260832))
VARIANTS = ("no_beta", "joint_beta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-beta-config",
        default=str(REPOSITORY_ROOT / "configs/gpu_beta_ablation_no_beta.yaml"),
    )
    parser.add_argument(
        "--joint-beta-config",
        default=str(REPOSITORY_ROOT / "configs/gpu_beta_ablation_joint_beta.yaml"),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated paired model/training seeds",
    )
    parser.add_argument(
        "--run-root",
        default=str(REPOSITORY_ROOT / "runs/gpu_beta_multiseed_ablation"),
    )
    parser.add_argument(
        "--parquet-glob",
        help="Optional absolute dataset glob for an isolated worktree",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--manifest-name",
        default="experiment_manifest.json",
        help="Unique manifest filename when independent seed shards run concurrently",
    )
    parser.add_argument(
        "--first-variant",
        choices=VARIANTS,
        default="no_beta",
        help=(
            "Variant trained first for the shard's first pair; subsequent pairs "
            "alternate. This permits balanced execution order across GPU shards."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration {path} is not a mapping")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def normalized_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only labels/output and the intended treatment indicator."""
    value = deepcopy(config)
    value["project"]["name"] = "PAIRED"
    value["output"]["run_dir"] = "PAIRED"
    value["data"]["beta_response"]["enabled"] = "TREATMENT"
    return value


def preflight(
    no_beta: dict[str, Any], joint_beta: dict[str, Any], seeds: tuple[int, ...]
) -> dict[str, Any]:
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from forwardfm_step1.data import (  # noqa: PLC0415
        BASE_TARGET_COLUMNS,
        BETA_TARGET_COLUMN,
        data_order_seed,
        data_split_seed,
        response_target_names,
        selection_sql,
    )
    from forwardfm_step1.model import ConditionalMDN  # noqa: PLC0415

    if not seeds or len(seeds) > 10 or len(set(seeds)) != len(seeds):
        raise ValueError("Worker seeds must be one to ten distinct paired seeds")
    if normalized_training_config(no_beta) != normalized_training_config(joint_beta):
        raise AssertionError(
            "Paired configs differ outside project label, output path, or beta target"
        )
    if no_beta["data"]["beta_response"]["enabled"] is not False:
        raise AssertionError("Control configuration unexpectedly enables beta target")
    if joint_beta["data"]["beta_response"]["enabled"] is not True:
        raise AssertionError("Treatment configuration does not enable beta target")
    if selection_sql(no_beta) != selection_sql(joint_beta):
        raise AssertionError("Teacher selections differ between paired configurations")
    if response_target_names(no_beta) != BASE_TARGET_COLUMNS:
        raise AssertionError("Control response target is not the three-vector baseline")
    if response_target_names(joint_beta) != (*BASE_TARGET_COLUMNS, BETA_TARGET_COLUMN):
        raise AssertionError("Treatment response target does not include delta-beta")
    split_seed = data_split_seed(no_beta)
    order_seed = data_order_seed(no_beta)
    if split_seed != data_split_seed(joint_beta) or order_seed != data_order_seed(
        joint_beta
    ):
        raise AssertionError("Fixed data seeds differ between variants")
    if not no_beta["model"].get("deterministic_component_initialization", False):
        raise AssertionError("Component-paired initialization is not enabled")

    def build(target_dim: int, seed: int) -> ConditionalMDN:
        model_config = no_beta["model"]
        model = ConditionalMDN(
            n_continuous=4,
            n_species=3,
            n_rec_pid_classes=12,
            hidden_width=int(model_config["hidden_width"]),
            hidden_layers=int(model_config["hidden_layers"]),
            pid_embedding_dim=int(model_config["pid_embedding_dim"]),
            mixture_components=int(model_config["mixture_components"]),
            target_dim=target_dim,
            dropout=float(model_config["dropout"]),
        )
        model.reset_parameters(seed=seed)
        return model

    control = build(3, seeds[0])
    treatment = build(4, seeds[0])
    control_state = control.state_dict()
    treatment_state = treatment.state_dict()
    paired_names = [
        name
        for name in control_state
        if name.startswith("species_embedding")
        or name.startswith("backbone")
        or name.startswith("mixture_head")
        or name.startswith("pid_head")
    ]
    import torch  # noqa: PLC0415

    unequal = [
        name
        for name in paired_names
        if not torch.equal(control_state[name], treatment_state[name])
    ]
    if unequal:
        raise AssertionError(f"Paired initial weights differ: {unequal}")
    return {
        "paired_seeds": list(seeds),
        "data_split_seed": split_seed,
        "data_order_seed": order_seed,
        "selection_sql": selection_sql(no_beta),
        "control_targets": list(response_target_names(no_beta)),
        "treatment_targets": list(response_target_names(joint_beta)),
        "identical_initialized_tensor_count": len(paired_names),
        "only_config_treatment": "data.beta_response.enabled",
    }


def completed_run(run_dir: Path, seed: int, target_dim: int) -> bool:
    required = ("model.pt", "metrics.json", "data_audit.json", "resolved_config.yaml")
    if not all((run_dir / name).is_file() for name in required):
        return False
    with (run_dir / "resolved_config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if int(config["project"]["seed"]) != seed:
        return False
    target_names = json.loads((run_dir / "data_audit.json").read_text())["target_names"]
    return len(target_names) == target_dim


def materialize_config(
    template: dict[str, Any],
    variant: str,
    seed: int,
    run_root: Path,
    generated_config_dir: Path,
    device: str,
) -> tuple[Path, Path]:
    config = deepcopy(template)
    config["project"]["seed"] = seed
    config["project"]["name"] = f"beta-multiseed-{variant}-{seed}"
    config["training"]["device"] = device
    run_dir = run_root / f"seed_{seed}" / variant
    config["output"]["run_dir"] = str(run_dir.resolve())
    path = generated_config_dir / f"seed_{seed}_{variant}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path, run_dir


def train_one(config_path: Path, run_dir: Path, device: str) -> float:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "train.py"),
        "--config",
        str(config_path),
        "--device",
        device,
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    elapsed = time.perf_counter() - start
    if return_code:
        raise RuntimeError(f"Training failed with code {return_code}: {config_path}")
    return elapsed


def validate_pair(run_root: Path, seed: int) -> dict[str, Any]:
    run_dirs = {variant: run_root / f"seed_{seed}" / variant for variant in VARIANTS}
    audits = {
        variant: json.loads((path / "data_audit.json").read_text())
        for variant, path in run_dirs.items()
    }
    checkpoints: dict[str, Any] = {}
    import torch

    for variant, path in run_dirs.items():
        checkpoints[variant] = torch.load(
            path / "model.pt", map_location="cpu", weights_only=False
        )
    control = audits["no_beta"]
    treatment = audits["joint_beta"]
    equality_fields = (
        "dataset_metadata_sha256",
        "selection_sql",
        "sampled_counts",
        "data_split_seed",
        "data_order_seed",
    )
    for field in equality_fields:
        if control[field] != treatment[field]:
            raise AssertionError(f"Seed {seed} pair differs in audit field {field}")
    if checkpoints["no_beta"]["seed"] != checkpoints["joint_beta"]["seed"]:
        raise AssertionError("Pair checkpoint model seeds differ")
    expected_policy = "deterministic_component_streams_and_training_rng_reset"
    if {
        checkpoints[variant].get("initialization_policy") for variant in VARIANTS
    } != {expected_policy}:
        raise AssertionError("Pair checkpoint initialization policy is not controlled")
    return {
        "seed": seed,
        "dataset_metadata_sha256": control["dataset_metadata_sha256"],
        "selection_identical": True,
        "sampled_counts_identical": True,
        "best_epoch_no_beta": int(checkpoints["no_beta"]["best_epoch"]),
        "best_epoch_joint_beta": int(checkpoints["joint_beta"]["best_epoch"]),
    }


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    no_beta = load_yaml(Path(args.no_beta_config).resolve())
    joint_beta = load_yaml(Path(args.joint_beta_config).resolve())
    if args.parquet_glob:
        no_beta["data"]["parquet_glob"] = args.parquet_glob
        joint_beta["data"]["parquet_glob"] = args.parquet_glob
    preflight_record = preflight(no_beta, joint_beta, seeds)
    run_root = Path(args.run_root).resolve()
    generated_config_dir = run_root / "generated_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.manifest_name).name != args.manifest_name:
        raise ValueError("--manifest-name must be a filename, not a path")
    manifest_stem = Path(args.manifest_name).stem
    write_json(run_root / f"preflight_{manifest_stem}.json", preflight_record)

    manifest: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "device": args.device,
        "first_variant": args.first_variant,
        "preflight": preflight_record,
        "runs": [],
        "pairs": [],
    }
    manifest_path = run_root / args.manifest_name
    write_json(manifest_path, manifest)
    templates = {"no_beta": no_beta, "joint_beta": joint_beta}

    for pair_index, seed in enumerate(seeds):
        # Balance possible time/order effects across the two treatments.
        initial_order = (
            VARIANTS
            if args.first_variant == "no_beta"
            else tuple(reversed(VARIANTS))
        )
        order = initial_order if pair_index % 2 == 0 else tuple(reversed(initial_order))
        print(f"PAIR_START seed={seed} order={','.join(order)}", flush=True)
        for variant in order:
            target_dim = 3 if variant == "no_beta" else 4
            config_path, run_dir = materialize_config(
                templates[variant],
                variant,
                seed,
                run_root,
                generated_config_dir,
                args.device,
            )
            skipped = completed_run(run_dir, seed, target_dim) and not args.force
            elapsed = 0.0
            if skipped:
                print(f"RUN_SKIP seed={seed} variant={variant}", flush=True)
            else:
                print(f"RUN_START seed={seed} variant={variant}", flush=True)
                elapsed = train_one(config_path, run_dir, args.device)
                if not completed_run(run_dir, seed, target_dim):
                    raise AssertionError("Training returned without complete artifacts")
                print(
                    f"RUN_DONE seed={seed} variant={variant} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
            manifest["runs"].append(
                {
                    "seed": seed,
                    "variant": variant,
                    "order_in_pair": order.index(variant) + 1,
                    "run_dir": str(run_dir),
                    "elapsed_seconds": elapsed,
                    "resumed": skipped,
                    "status": "complete",
                }
            )
            write_json(manifest_path, manifest)
        pair_record = validate_pair(run_root, seed)
        manifest["pairs"].append(pair_record)
        write_json(manifest_path, manifest)
        print(f"PAIR_DONE seed={seed}", flush=True)

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(f"EXPERIMENT_DONE pairs={len(seeds)} run_root={run_root}", flush=True)


if __name__ == "__main__":
    main()
