#!/usr/bin/env python3
"""Evaluate a frozen split-asinh checkpoint on August or native external MC."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forwardfm_transfer.response import evaluate_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--native-parquet")
    parser.add_argument("--parquet-glob")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-rows-per-species", type=int, default=100000)
    parser.add_argument("--sample-seed", type=int, default=20261101)
    parser.add_argument("--trigger", choices=("associated_fd", "identified_fd"), default="associated_fd")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--min-bin-count", type=int, default=200)
    args = parser.parse_args()
    if args.max_rows_per_species <= 0 or args.min_bin_count <= 0:
        parser.error("Sample and bin counts must be positive")
    metrics = evaluate_response(args.checkpoint, args.output, args.native_parquet,
        args.parquet_glob, args.device, args.max_rows_per_species, args.sample_seed,
        args.trigger, args.split, args.min_bin_count)
    print(json.dumps({"species": metrics["species"], "inference_seconds": metrics["inference_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
