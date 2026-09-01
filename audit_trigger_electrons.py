#!/usr/bin/env python3
"""Audit the all-event generated-electron denominator and label semantics.

This script is deliberately read-only with respect to the Parquet teacher
sample.  It must run before training because trigger efficiency is only
meaningful when the denominator contains one truth-selected electron from
every generated event, including trigger and reconstruction failures.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


REQUIRED_COLUMNS = {
    "source_file_id",
    "event_id",
    "mcindex",
    "gen_pid",
    "is_generated_trigger_electron",
    "has_valid_trigger_electron",
    "trigger_electron_count",
    "trigger_status",
    "trigger_pindex",
    "trigger_mcindex",
    "reconstructed",
    "matched_pindex",
    "match_reciprocal",
    "rec_pid",
    "rec_detector_region",
    "rec_theta",
    "rec_beta",
    "gen_vz",
    "delta_p",
    "delta_theta",
    "delta_phi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def connect(parquet_glob: str) -> duckdb.DuckDBPyConnection:
    quoted = parquet_glob.replace("'", "''")
    connection = duckdb.connect()
    connection.execute(
        f"CREATE OR REPLACE VIEW particles AS SELECT * FROM '{quoted}'"
    )
    return connection


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def dataset_fingerprint(parquet_glob: str) -> dict[str, Any]:
    files = sorted(glob.glob(parquet_glob))
    file_records = [
        {"name": Path(path).name, "bytes": Path(path).stat().st_size}
        for path in files
    ]
    payload = json.dumps(file_records, sort_keys=True).encode("utf-8")
    return {
        "dataset_file_count": len(file_records),
        "dataset_total_bytes": sum(row["bytes"] for row in file_records),
        "dataset_metadata_sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    con = connect(args.parquet_glob)
    columns = [row[0] for row in con.execute("DESCRIBE particles").fetchall()]
    missing = sorted(REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"Dataset is missing required electron-audit columns: {missing}")

    global_summary = con.execute(
        """
        SELECT
          count(*) AS particle_rows,
          count(DISTINCT (source_file_id, event_id)) AS events,
          count(*) FILTER (WHERE gen_pid = 11) AS pid11_rows,
          count(*) FILTER (WHERE is_generated_trigger_electron) AS flagged_rows,
          count(*) FILTER (
            WHERE is_generated_trigger_electron AND gen_pid <> 11
          ) AS flagged_non_electrons,
          count(*) FILTER (
            WHERE gen_pid = 11 AND NOT is_generated_trigger_electron
          ) AS pid11_not_flagged,
          count(*) FILTER (WHERE is_generated_trigger_electron IS NULL)
            AS null_role_flags
        FROM particles
        """
    ).fetch_df()

    multiplicity = con.execute(
        """
        WITH per_event AS (
          SELECT source_file_id, event_id,
                 count(*) FILTER (WHERE gen_pid = 11) AS n_pid11,
                 count(*) FILTER (WHERE is_generated_trigger_electron) AS n_flagged,
                 count(DISTINCT has_valid_trigger_electron) AS n_trigger_flag_values,
                 min(CAST(has_valid_trigger_electron AS INTEGER)) AS min_trigger_flag,
                 max(CAST(has_valid_trigger_electron AS INTEGER)) AS max_trigger_flag,
                 min(trigger_electron_count) AS min_trigger_count,
                 max(trigger_electron_count) AS max_trigger_count
          FROM particles GROUP BY source_file_id, event_id
        )
        SELECT n_pid11, n_flagged, n_trigger_flag_values,
               min_trigger_flag, max_trigger_flag,
               min_trigger_count, max_trigger_count,
               count(*) AS events
        FROM per_event
        GROUP BY ALL
        ORDER BY events DESC, n_pid11, n_flagged
        """
    ).fetch_df()

    label_counts = con.execute(
        """
        SELECT has_valid_trigger_electron,
               count(*) AS generated_electrons,
               count(*) FILTER (WHERE trigger_mcindex = mcindex) AS mcindex_matches,
               count(*) FILTER (WHERE reconstructed) AS reconstructed_rows,
               count(*) FILTER (WHERE match_reciprocal) AS reciprocal_rows,
               count(*) FILTER (WHERE rec_pid = 11) AS rec_pid11_rows
        FROM particles
        WHERE gen_pid = 11
        GROUP BY has_valid_trigger_electron
        ORDER BY has_valid_trigger_electron
        """
    ).fetch_df()

    association = con.execute(
        """
        SELECT
          count(*) FILTER (WHERE has_valid_trigger_electron) AS trigger_successes,
          count(*) FILTER (
            WHERE has_valid_trigger_electron AND trigger_mcindex = mcindex
          ) AS successful_mcindex_matches,
          count(*) FILTER (
            WHERE has_valid_trigger_electron
              AND trigger_mcindex IS DISTINCT FROM mcindex
          ) AS successful_mcindex_mismatches,
          count(*) FILTER (
            WHERE NOT has_valid_trigger_electron AND trigger_mcindex >= 0
          ) AS failures_with_nonnegative_trigger_mcindex,
          count(*) FILTER (
            WHERE is_generated_trigger_electron
              IS DISTINCT FROM has_valid_trigger_electron
          ) AS role_flag_trigger_mismatches,
          min(trigger_mcindex) AS min_trigger_mcindex,
          max(trigger_mcindex) AS max_trigger_mcindex,
          min(trigger_pindex) AS min_trigger_pindex,
          max(trigger_pindex) AS max_trigger_pindex
        FROM particles
        WHERE gen_pid = 11
        """
    ).fetch_df()

    electron_mcindex = con.execute(
        """
        SELECT mcindex, is_generated_trigger_electron,
               has_valid_trigger_electron,
               count(*) AS generated_electrons
        FROM particles
        WHERE gen_pid = 11
        GROUP BY ALL
        ORDER BY generated_electrons DESC
        """
    ).fetch_df()

    role_flag_vs_trigger = con.execute(
        """
        SELECT is_generated_trigger_electron, has_valid_trigger_electron,
               count(*) AS generated_electrons
        FROM particles
        WHERE gen_pid = 11
        GROUP BY ALL
        ORDER BY generated_electrons DESC
        """
    ).fetch_df()

    trigger_status = con.execute(
        """
        SELECT has_valid_trigger_electron, trigger_status,
               trigger_electron_count, count(*) AS generated_electrons
        FROM particles
        WHERE gen_pid = 11
        GROUP BY ALL
        ORDER BY generated_electrons DESC
        """
    ).fetch_df()

    match_encoding = con.execute(
        """
        SELECT has_valid_trigger_electron, reconstructed,
               matched_pindex, match_reciprocal,
               coalesce(rec_detector_region, '<NULL>') AS rec_detector_region,
               rec_pid, count(*) AS generated_electrons
        FROM particles
        WHERE gen_pid = 11
        GROUP BY ALL
        ORDER BY generated_electrons DESC
        LIMIT 100
        """
    ).fetch_df()

    outcome_counts = con.execute(
        """
        SELECT
          CASE
            WHEN NOT reconstructed OR matched_pindex < 0 THEN 'unreconstructed'
            WHEN rec_detector_region IN ('FD', 'FT', 'CD') THEN rec_detector_region
            ELSE 'other'
          END AS outcome,
          count(*) AS generated_electrons,
          count(*) FILTER (WHERE has_valid_trigger_electron) AS trigger_successes
        FROM particles
        WHERE gen_pid = 11
        GROUP BY 1 ORDER BY generated_electrons DESC
        """
    ).fetch_df()

    successful_pid = con.execute(
        """
        SELECT rec_pid,
               coalesce(rec_detector_region, '<NULL>') AS rec_detector_region,
               match_reciprocal,
               count(*) AS generated_electrons
        FROM particles
        WHERE gen_pid = 11
          AND has_valid_trigger_electron
          AND trigger_mcindex = mcindex
        GROUP BY ALL
        ORDER BY generated_electrons DESC
        """
    ).fetch_df()

    response_cutflow = con.execute(
        """
        SELECT
          count(*) AS primary_truth_electrons,
          count(*) FILTER (
            WHERE has_valid_trigger_electron AND trigger_mcindex = mcindex
          ) AS trigger_associated,
          count(*) FILTER (
            WHERE has_valid_trigger_electron AND trigger_mcindex = mcindex
              AND reconstructed AND matched_pindex >= 0
          ) AS reconstructed_matched,
          count(*) FILTER (
            WHERE has_valid_trigger_electron AND trigger_mcindex = mcindex
              AND reconstructed AND matched_pindex >= 0
              AND rec_detector_region = 'FD'
          ) AS reconstructed_fd,
          count(*) FILTER (
            WHERE has_valid_trigger_electron AND trigger_mcindex = mcindex
              AND reconstructed AND matched_pindex >= 0
              AND rec_detector_region = 'FD' AND match_reciprocal
          ) AS reciprocal_fd,
          count(*) FILTER (
            WHERE has_valid_trigger_electron AND trigger_mcindex = mcindex
              AND reconstructed AND matched_pindex >= 0
              AND rec_detector_region = 'FD' AND match_reciprocal
              AND rec_theta < radians(33.0)
              AND gen_vz > -5.5 AND gen_vz < -0.5
              AND delta_p IS NOT NULL AND delta_theta IS NOT NULL
              AND delta_phi IS NOT NULL
              AND isfinite(delta_p) AND isfinite(delta_theta)
              AND isfinite(delta_phi)
          ) AS shared_fiducial_residual,
          count(*) FILTER (
            WHERE has_valid_trigger_electron AND trigger_mcindex = mcindex
              AND reconstructed AND matched_pindex >= 0
              AND rec_detector_region = 'FD' AND match_reciprocal
              AND rec_theta < radians(33.0)
              AND gen_vz > -5.5 AND gen_vz < -0.5
              AND delta_p IS NOT NULL AND delta_theta IS NOT NULL
              AND delta_phi IS NOT NULL
              AND isfinite(delta_p) AND isfinite(delta_theta)
              AND isfinite(delta_phi)
              AND rec_pid <> 0 AND rec_beta > -99
              AND abs(delta_p) <= 10
          ) AS final_response_quality
        FROM particles
        WHERE gen_pid = 11
        """
    ).fetch_df()

    residual_failure_encoding = con.execute(
        """
        SELECT has_valid_trigger_electron, reconstructed,
               (matched_pindex >= 0) AS has_nonnegative_match,
               count(*) AS generated_electrons,
               count(*) FILTER (WHERE delta_p IS NULL) AS null_delta_p,
               count(*) FILTER (WHERE delta_theta IS NULL) AS null_delta_theta,
               count(*) FILTER (WHERE delta_phi IS NULL) AS null_delta_phi,
               count(*) FILTER (WHERE rec_pid = 0) AS rec_pid_zero,
               count(*) FILTER (WHERE rec_beta <= -99) AS beta_sentinel
        FROM particles
        WHERE gen_pid = 11
        GROUP BY 1,2,3
        ORDER BY generated_electrons DESC
        """
    ).fetch_df()

    tables = {
        "global_summary": global_summary,
        "event_multiplicity": multiplicity,
        "trigger_label_counts": label_counts,
        "trigger_association": association,
        "electron_mcindex": electron_mcindex,
        "role_flag_vs_trigger": role_flag_vs_trigger,
        "trigger_status": trigger_status,
        "match_failure_encoding_top100": match_encoding,
        "reconstruction_outcome_counts": outcome_counts,
        "successful_trigger_pid_distribution": successful_pid,
        "electron_response_cutflow": response_cutflow,
        "residual_failure_encoding": residual_failure_encoding,
    }
    for name, frame in tables.items():
        write_csv(frame, output_dir / f"{name}.csv")

    fingerprint = dataset_fingerprint(args.parquet_glob)
    summary = {
        "dataset_glob": args.parquet_glob,
        **fingerprint,
        "schema_column_count": len(columns),
        "schema_columns": columns,
        "denominator_sql": (
            "SELECT ... FROM particles WHERE gen_pid = 11"
        ),
        "candidate_trigger_label": "has_valid_trigger_electron",
        "trigger_association_invariant": (
            "For positive gen_pid=11 rows, trigger_mcindex = mcindex"
        ),
        "tables": {name: records(frame) for name, frame in tables.items()},
    }
    (output_dir / "electron_data_audit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    con.close()
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
