from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} is not a mapping")
    config = deepcopy(config)
    config["_config_path"] = str(path)
    return config


def resolve_run_dir(config: dict[str, Any]) -> Path:
    config_path = Path(config["_config_path"])
    project_root = config_path.parent.parent
    run_dir = Path(config["output"]["run_dir"])
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    return run_dir.resolve()


def apply_smoke_overrides(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    config["data"]["max_rows_per_species"] = {
        "train": 2_000,
        "validation": 500,
        "test": 500,
    }
    config["model"]["hidden_width"] = 32
    config["model"]["hidden_layers"] = 2
    config["model"]["mixture_components"] = 3
    config["training"]["epochs"] = 2
    config["training"]["batch_size"] = 512
    config["training"]["early_stopping_patience"] = 2
    config["output"]["run_dir"] = "runs/smoke"
    return config
