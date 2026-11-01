#!/usr/bin/env python3
"""Prepare identical whole-event inputs for the two-platform transfer study.

Native samples preserve every particle and outcome of each selected event.
The response reference contains only the original held-out event partition. No
kinematic, reconstruction, PID, or trigger selection enters this preparation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--response-glob", "--august-glob", dest="august_glob", metavar="RESPONSE_GLOB", required=True,
                        help="Response-training phase-space files; the old flag remains an alias")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-modulus", type=int, default=10)
    args = parser.parse_args()
    if args.sample_modulus < 1:
        parser.error("sample-modulus must be positive")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='2GB'")
    manifest = {
        "duckdb_version": duckdb.__version__,
        "native_selection": f"hash(file_id,event)%{args.sample_modulus}=0",
        "august_selection": "hash(source_file_id,event_id,20260822)%10000>=9000",
        "weights": "Native exports omit HEPGen event weights; unweighted MC study.",
        "datasets": {},
    }
    for name, source in (("rho", args.rho), ("phase", args.phase), ("august_test", args.august_glob)):
        con.read_parquet(source).create_view("src", replace=True)
        native = name != "august_test"
        condition = (manifest["native_selection"] if native else manifest["august_selection"])
        target = out / f"{name}.parquet"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing sample {target}")
        literal = str(target).replace("'", "''")
        con.execute(f"COPY (SELECT * FROM src WHERE {condition}) TO '{literal}' "
                    "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
        con.read_parquet(str(target)).create_view("sample", replace=True)
        keys = "(file_id,event)" if native else "(source_file_id,event_id)"
        pid = "pid_gen" if native else "gen_pid"
        count, events = con.execute(f"SELECT count(*),count(DISTINCT {keys}) FROM sample").fetchone()
        bad = con.execute(f"""SELECT count(*) FROM (
            SELECT {keys} AS k,count(*) AS n,count(DISTINCT {pid}) AS ns
            FROM sample GROUP BY k HAVING n<>4 OR ns<>4)""").fetchone()[0]
        if bad:
            raise AssertionError(f"{bad} incomplete events in {name}")
        checksum = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(block)
        manifest["datasets"][name] = {
            "file": target.name, "source": source, "rows": count, "events": events,
            "bytes": target.stat().st_size, "sha256": checksum.hexdigest(),
        }
        print(name, manifest["datasets"][name], flush=True)
        (out / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    con.close()


if __name__ == "__main__":
    main()
