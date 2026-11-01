#!/usr/bin/env python3
"""Audit all rows of prepared native inputs before interpreting closure."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forwardfm_transfer.data import audit_native, connect_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {}
    for name in ("phase", "rho"):
        con = connect_native(Path(args.inputs) / f"{name}.parquet")
        try:
            audit = audit_native(con)
        finally:
            con.close()
        if any(audit["violations"].values()) or audit["duplicate_generated_keys"]["keys"]:
            raise AssertionError(f"Invalid native labels: {name}, {audit}")
        result[name] = audit
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v["counts"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
