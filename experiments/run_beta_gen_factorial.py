#!/usr/bin/env python3
"""Run the matched beta-gen input / delta-beta target factorial experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
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
VARIANT_CONFIGS = {
    "A_original": "gpu_beta_factorial_A_original.yaml",
    "B_beta_input": "gpu_beta_factorial_B_beta_input.yaml",
    "C_beta_target": "gpu_beta_factorial_C_beta_target.yaml",
    "D_input_target_pid02": "gpu_beta_factorial_D_beta_input_target.yaml",
    "A_original_pid1": "gpu_beta_factorial_A_original_pid1.yaml",
    "B_beta_input_pid1": "gpu_beta_factorial_B_beta_input_pid1.yaml",
    "D_input_target_pid1": "gpu_beta_factorial_D_beta_input_target_pid1.yaml",
}
EXPECTED_TREATMENTS = {
    "A_original": (False, False, 0.20),
    "B_beta_input": (True, False, 0.20),
    "C_beta_target": (False, True, 0.20),
    "D_input_target_pid02": (True, True, 0.20),
    "A_original_pid1": (False, False, 1.00),
    "B_beta_input_pid1": (True, False, 1.00),
    "D_input_target_pid1": (True, True, 1.00),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANT_CONFIGS),
        help="Comma-separated condition labels",
    )
    parser.add_argument(
        "--run-root",
        default=str(REPOSITORY_ROOT / "runs/gpu_beta_gen_factorial"),
    )
    parser.add_argument("--parquet-glob")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest-name", default="experiment_manifest.json")
    parser.add_argument("--first-variant", choices=tuple(VARIANT_CONFIGS), default="A_original")
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


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove labels/output and exactly the three intended treatments."""
    value = deepcopy(config)
    value["project"]["name"] = "FACTORIAL"
    value["output"]["run_dir"] = "FACTORIAL"
    beta = value["data"]["beta_response"]
    beta["include_generated_beta"] = "TREATMENT"
    beta["enabled"] = "TREATMENT"
    value["training"]["pid_loss_weight"] = "TREATMENT"
    return value


def build_model(config: dict[str, Any], n_continuous: int, target_dim: int):
    from forwardfm_step1.model import ConditionalMDN

    model_config = config["model"]
    return ConditionalMDN(
        n_continuous=n_continuous,
        n_species=3,
        n_rec_pid_classes=12,
        hidden_width=int(model_config["hidden_width"]),
        hidden_layers=int(model_config["hidden_layers"]),
        pid_embedding_dim=int(model_config["pid_embedding_dim"]),
        mixture_components=int(model_config["mixture_components"]),
        target_dim=target_dim,
        dropout=0.0,
    )


def preflight(
    configs: dict[str, dict[str, Any]], seeds: tuple[int, ...]
) -> dict[str, Any]:
    sys.path.insert(0, str(REPOSITORY_ROOT))
    import torch

    from forwardfm_step1.data import (
        continuous_feature_names,
        data_order_seed,
        data_split_seed,
        response_target_names,
        selection_sql,
    )
    from forwardfm_step1.model import initialize_beta_input_from_control

    if not seeds or len(seeds) > 10 or len(set(seeds)) != len(seeds):
        raise ValueError("Worker seeds must be one to ten distinct seeds")
    reference = normalized_config(next(iter(configs.values())))
    if any(normalized_config(config) != reference for config in configs.values()):
        raise AssertionError("Factorial configs differ outside declared treatments")

    selections = {selection_sql(config) for config in configs.values()}
    if len(selections) != 1:
        raise AssertionError("Factorial conditions do not select identical rows")
    split_seeds = {data_split_seed(config) for config in configs.values()}
    order_seeds = {data_order_seed(config) for config in configs.values()}
    if len(split_seeds) != 1 or len(order_seeds) != 1:
        raise AssertionError("Factorial conditions use different data seeds")

    conditions: dict[str, Any] = {}
    for variant, config in configs.items():
        beta_input, beta_target, pid_weight = EXPECTED_TREATMENTS[variant]
        beta_config = config["data"]["beta_response"]
        observed = (
            bool(beta_config["include_generated_beta"]),
            bool(beta_config["enabled"]),
            float(config["training"]["pid_loss_weight"]),
        )
        if observed != (beta_input, beta_target, pid_weight):
            raise AssertionError(f"Unexpected treatment fields for {variant}: {observed}")
        if config["training"].get("checkpoint_metric") != "pid_cross_entropy":
            raise AssertionError("Primary checkpoint rule must be validation PID CE")
        conditions[variant] = {
            "feature_names": list(continuous_feature_names(config)),
            "target_names": list(response_target_names(config)),
            "pid_loss_weight": pid_weight,
        }

    exact_initial_function_pairs = []
    for target_dim, pair in (
        (3, ("A_original", "B_beta_input")),
        (4, ("C_beta_target", "D_input_target_pid02")),
        (3, ("A_original_pid1", "B_beta_input_pid1")),
    ):
        if any(variant not in configs for variant in pair):
            continue
        control = build_model(configs[pair[0]], 4, target_dim)
        treatment = build_model(configs[pair[1]], 5, target_dim)
        control.reset_parameters(seed=seeds[0])
        treatment.reset_parameters(seed=seeds[0])
        initialize_beta_input_from_control(control, treatment)
        base = torch.randn(17, 4, generator=torch.Generator().manual_seed(9))
        informed = torch.cat([base, torch.rand(17, 1)], dim=1)
        species = torch.arange(17) % 3
        left = control(base, species)
        right = treatment(informed, species)
        fields = ("mixture_logits", "means", "log_scales", "pid_logits")
        if any(not torch.equal(getattr(left, name), getattr(right, name)) for name in fields):
            raise AssertionError(f"Nested initialization failed for {pair}")
        exact_initial_function_pairs.append(list(pair))

    return {
        "seeds": list(seeds),
        "conditions": conditions,
        "data_split_seed": next(iter(split_seeds)),
        "data_order_seed": next(iter(order_seeds)),
        "selection_sql": next(iter(selections)),
        "primary_checkpoint_metric": "validation pid_cross_entropy",
        "exact_initial_function_pairs": exact_initial_function_pairs,
    }


def completed_run(run_dir: Path, seed: int, condition: tuple[bool, bool, float]) -> bool:
    required = (
        "model.pt",
        "model_min_validation_total_loss.pt",
        "model_min_validation_pid_cross_entropy.pt",
        "metrics.json",
        "data_audit.json",
        "resolved_config.yaml",
        "pid_correct_id_closure_mae.csv",
    )
    if not all((run_dir / name).is_file() for name in required):
        return False
    config = load_yaml(run_dir / "resolved_config.yaml")
    beta_input, beta_target, pid_weight = condition
    beta_config = config["data"]["beta_response"]
    return (
        int(config["project"]["seed"]) == seed
        and bool(beta_config["include_generated_beta"]) == beta_input
        and bool(beta_config["enabled"]) == beta_target
        and float(config["training"]["pid_loss_weight"]) == pid_weight
    )


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
    config["project"]["name"] = f"beta-gen-factorial-{variant}-{seed}"
    config["training"]["device"] = device
    run_dir = run_root / f"seed_{seed}" / variant
    config["output"]["run_dir"] = str(run_dir.resolve())
    path = generated_config_dir / f"seed_{seed}_{variant}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path, run_dir


def train_one(config_path: Path, run_dir: Path, device: str) -> float:
    run_dir.mkdir(parents=True, exist_ok=True)
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
    lock_path = run_dir / ".training.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another worker is training in {run_dir}") from error
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        with (run_dir / "training.log").open("w", encoding="utf-8") as log:
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
    if return_code:
        raise RuntimeError(f"Training failed with code {return_code}: {config_path}")
    return time.perf_counter() - start


def validate_seed(run_root: Path, seed: int, variants: tuple[str, ...]) -> dict[str, Any]:
    import torch

    audits = {
        variant: json.loads(
            (run_root / f"seed_{seed}" / variant / "data_audit.json").read_text()
        )
        for variant in variants
    }
    reference = audits[variants[0]]
    equality_fields = (
        "dataset_metadata_sha256",
        "selection_sql",
        "sampled_counts",
        "data_split_seed",
        "data_order_seed",
    )
    for variant, audit in audits.items():
        for field in equality_fields:
            if audit[field] != reference[field]:
                raise AssertionError(f"Seed {seed} {variant} differs in {field}")

    epochs: dict[str, int] = {}
    for variant in variants:
        checkpoint = torch.load(
            run_root / f"seed_{seed}" / variant / "model.pt",
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint["seed"] != seed:
            raise AssertionError(f"Checkpoint seed mismatch for {variant}")
        if checkpoint["checkpoint_selection"]["primary_metric"] != "pid_cross_entropy":
            raise AssertionError(f"Test checkpoint rule changed for {variant}")
        epochs[variant] = int(checkpoint["best_epoch"])
    return {
        "seed": seed,
        "dataset_metadata_sha256": reference["dataset_metadata_sha256"],
        "identical_teacher_population": True,
        "primary_checkpoint_metric": "validation pid_cross_entropy",
        "selected_epochs": epochs,
    }


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    variants = tuple(value for value in args.variants.split(",") if value)
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("--variants must contain distinct condition labels")
    unknown = sorted(set(variants).difference(VARIANT_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    configs = {
        variant: load_yaml(REPOSITORY_ROOT / "configs" / VARIANT_CONFIGS[variant])
        for variant in variants
    }
    if args.parquet_glob:
        for config in configs.values():
            config["data"]["parquet_glob"] = args.parquet_glob
    preflight_record = preflight(configs, seeds)

    run_root = Path(args.run_root).resolve()
    generated_config_dir = run_root / "generated_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.manifest_name).name != args.manifest_name:
        raise ValueError("--manifest-name must be a filename")
    manifest_path = run_root / args.manifest_name
    manifest: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "device": args.device,
        "preflight": preflight_record,
        "runs": [],
        "seeds": [],
    }
    write_json(manifest_path, manifest)

    start_index = variants.index(args.first_variant) if args.first_variant in variants else 0
    base_order = variants[start_index:] + variants[:start_index]
    for seed_index, seed in enumerate(seeds):
        shift = seed_index % len(base_order)
        order = base_order[shift:] + base_order[:shift]
        if seed_index % 2:
            order = tuple(reversed(order))
        print(f"SEED_START seed={seed} order={','.join(order)}", flush=True)
        for variant in order:
            config_path, run_dir = materialize_config(
                configs[variant], variant, seed, run_root, generated_config_dir, args.device
            )
            skipped = completed_run(run_dir, seed, EXPECTED_TREATMENTS[variant]) and not args.force
            elapsed = 0.0
            if skipped:
                print(f"RUN_SKIP seed={seed} variant={variant}", flush=True)
            else:
                print(f"RUN_START seed={seed} variant={variant}", flush=True)
                elapsed = train_one(config_path, run_dir, args.device)
                if not completed_run(run_dir, seed, EXPECTED_TREATMENTS[variant]):
                    raise AssertionError(f"Incomplete artifacts for {seed} {variant}")
                print(
                    f"RUN_DONE seed={seed} variant={variant} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
            manifest["runs"].append(
                {
                    "seed": seed,
                    "variant": variant,
                    "run_dir": str(run_dir),
                    "elapsed_seconds": elapsed,
                    "resumed": skipped,
                }
            )
            write_json(manifest_path, manifest)
        manifest["seeds"].append(validate_seed(run_root, seed, variants))
        write_json(manifest_path, manifest)
        print(f"SEED_DONE seed={seed}", flush=True)

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(f"EXPERIMENT_DONE seeds={len(seeds)} run_root={run_root}", flush=True)


if __name__ == "__main__":
    main()
