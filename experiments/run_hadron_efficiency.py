#!/usr/bin/env python3
"""Run 3 independent species models per seed; data split is fixed in config."""

import argparse
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260822, 20260823, 20260824])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--root", type=Path, default=Path("runs/hadron_efficiency"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for seed in args.seeds:
        for species in (-211, 211, 2212):
            directory = args.root.resolve() / f"seed_{seed}" / f"pid_{species}"
            if (directory / "model.pt").exists() or (directory / "training.log").exists():
                raise FileExistsError(f"Refusing to overwrite existing experiment: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "train_hadron_efficiency.py", "--seed", str(seed),
                       "--species", str(species), "--device", args.device,
                       "--parquet-glob", args.parquet_glob, "--run-dir", str(directory)]
            print(f"Starting seed={seed}, species={species}; log={directory / 'training.log'}", flush=True)
            with (directory / "training.log").open("w") as stream:
                subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
