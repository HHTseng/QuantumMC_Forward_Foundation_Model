#!/usr/bin/env python3
"""Run a fixed transfer protocol; rho never enters fitting or model selection."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gate-epochs", type=int, default=30)
    parser.add_argument("--skip-response", action="store_true")
    parser.add_argument("--skip-efficiency", action="store_true")
    args = parser.parse_args()
    inputs, output = Path(args.inputs).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((inputs / "input_manifest.json").read_text())
    for item in manifest["datasets"].values():
        digest = hashlib.sha256()
        with (inputs / item["file"]).open("rb") as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Input checksum mismatch: {item['file']}")
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    commands = []
    if not args.skip_response:
        for seed in (20260822, 20260823, 20260824):
            checkpoint = ROOT / f"runs/raw_momentum_ablation/seed_{seed}/baseline/model.pt"
            for dataset, flag, path in (
                ("august", "--parquet-glob", inputs / "august_test.parquet"),
                ("rho", "--native-parquet", inputs / "rho.parquet"),
            ):
                target = output / f"response/seed_{seed}/{dataset}"
                commands.append((target, [sys.executable, str(ROOT / "experiments/evaluate_transfer_response.py"),
                    "--checkpoint", str(checkpoint), flag, str(path), "--output", str(target),
                    "--device", args.device]))
    if not args.skip_efficiency:
        for seed in (20261101, 20261102, 20261103):
            target = output / f"efficiency/seed_{seed}"
            commands.append((target, [sys.executable, str(ROOT / "experiments/run_transfer_efficiency.py"),
                "--phase-parquet", str(inputs / "phase.parquet"), "--rho-parquet", str(inputs / "rho.parquet"),
                "--output", str(target), "--device", args.device, "--seed", str(seed),
                "--epochs", str(args.gate_epochs)]))
    execution_path = output / "execution.json"
    record = (json.loads(execution_path.read_text()) if execution_path.exists() else
              {"commands": [], "protocol": "frozen August response; phase-only gate fitting; external rho test",
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    ran_command = False
    for target, command in commands:
        target.mkdir(parents=True, exist_ok=True)
        if (target / "complete.json").exists():
            previous = json.loads((target / "complete.json").read_text())
            if previous["arguments"] != command:
                raise ValueError(f"Completed output has different arguments: {target}; choose a new output directory")
            print(f"Already completed: {target}", flush=True)
            continue
        print("RUN", " ".join(command), flush=True)
        started = time.monotonic()
        with (target / "run.log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        ran_command = True
        item = {"arguments": command, "seconds": time.monotonic()-started,
                "returncode": process.returncode}
        record["commands"].append(item)
        execution_path.write_text(json.dumps(record, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"Experiment failed; inspect {target / 'run.log'}")
        (target / "complete.json").write_text(json.dumps(item, indent=2) + "\n")
        print(f"DONE {target} {item['seconds']:.1f}s", flush=True)
    if ran_command:
        record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        execution_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
