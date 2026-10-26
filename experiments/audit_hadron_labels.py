#!/usr/bin/env python3
"""Diagnose inclusive skim-defined labels; do not filter or relabel training rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forwardfm_electron.data import _fingerprint
from forwardfm_step1.data import connect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/hadron_efficiency/summary/matching_diagnostics.csv"))
    args = parser.parse_args()
    con = connect(args.parquet_glob)
    try:
        columns = {row[0] for row in con.execute("DESCRIBE particles").fetchall()}
        required = {"source_file_id", "event_id", "gen_pid", "has_valid_trigger_electron",
                    "reconstructed", "matched_pindex", "rec_detector_region", "gen_p", "gen_theta"}
        if required - columns:
            raise ValueError(f"Missing label-audit columns: {sorted(required - columns)}")
        con.execute("""
            CREATE TEMP VIEW scoped AS
            SELECT *, coalesce(reconstructed AND matched_pindex >= 0, false) AS matched
            FROM particles WHERE has_valid_trigger_electron
              AND gen_pid IN (11, -211, 211, 2212)
        """)
        # Include the generated electron: an electron/hadron collision in the
        # same REC::Particle index is just as ambiguous as two hadron matches.
        con.execute("""
            CREATE TEMP VIEW shared AS
            SELECT source_file_id, event_id, matched_pindex, count(*) AS shared_count
            FROM scoped WHERE matched
            GROUP BY source_file_id, event_id, matched_pindex HAVING count(*) > 1
        """)
        counts = {
            "matched_any": "matched",
            "matched_fd": "matched AND rec_detector_region = 'FD'",
            "matched_cd": "matched AND rec_detector_region = 'CD'",
            "matched_ft": "matched AND rec_detector_region = 'FT'",
            "matched_other": "matched AND coalesce(rec_detector_region, '') NOT IN ('FD','CD','FT')",
            "inconsistent_reconstructed_flag": "reconstructed IS DISTINCT FROM coalesce(matched_pindex >= 0, false)",
            "shared_reco_associations": "matched AND shared_count > 1",
        }
        optional = {
            "matched_nonreciprocal": ("match_reciprocal", "matched AND match_reciprocal IS FALSE"),
            "matched_reciprocity_missing": ("match_reciprocal", "matched AND match_reciprocal IS NULL"),
            "matched_wrong_pid": ("rec_pid", "matched AND rec_pid != gen_pid"),
            "matched_pid_missing": ("rec_pid", "matched AND rec_pid IS NULL"),
            "matched_invalid_beta": ("rec_beta", "matched AND NOT isnan(rec_beta) AND (rec_beta <= 0 OR rec_beta > 1.2)"),
            "matched_beta_missing_or_nonfinite": ("rec_beta", "matched AND (rec_beta IS NULL OR NOT isfinite(rec_beta))"),
            "matched_abs_delta_p_gt_10gev": ("delta_p", "matched AND NOT isnan(delta_p) AND abs(delta_p) > 10"),
            "matched_delta_p_missing_or_nonfinite": ("delta_p", "matched AND (delta_p IS NULL OR NOT isfinite(delta_p))"),
        }
        expressions = [f"count(*) FILTER (WHERE {predicate}) AS {name}"
                       for name, predicate in counts.items()]
        expressions += [(f"count(*) FILTER (WHERE {predicate})" if column in columns else "NULL::BIGINT")
                        + f" AS {name}" for name, (column, predicate) in optional.items()]
        frame = con.execute(f"""
            SELECT gen_pid, count(*) AS triggered_generated_denominator,
              {', '.join(expressions)},
              min(gen_p) FILTER (WHERE isfinite(gen_p)) AS min_gen_p_gev,
              max(gen_p) FILTER (WHERE isfinite(gen_p)) AS max_gen_p_gev,
              min(gen_theta) FILTER (WHERE isfinite(gen_theta)) AS min_gen_theta_rad,
              max(gen_theta) FILTER (WHERE isfinite(gen_theta)) AS max_gen_theta_rad
            FROM scoped LEFT JOIN shared USING (source_file_id, event_id, matched_pindex)
            WHERE gen_pid IN (-211, 211, 2212) GROUP BY gen_pid ORDER BY gen_pid
        """).fetch_df()
    finally:
        con.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    metadata = {
        "denominator": "all generated hadrons of each species with has_valid_trigger_electron=true",
        "label_definition": "reconstructed AND matched_pindex >= 0; inclusive one-way skim association, then detector region",
        "interpretation": "label diagnostics only; no filters applied, no models changed; not intrinsic detector efficiency",
        "shared_reco_definition": "same (source_file_id,event_id,matched_pindex) associated to >1 generated row among e-,pi-,pi+,proton",
        "wrong_pid_definition": "rec_pid != gen_pid, including PID 0; missing PID reported separately",
        "missing_optional_columns": sorted({column for column, _ in optional.values()} - columns),
        "null_counts": "absent optional columns yield NULL, not zero",
        "dataset_glob": args.parquet_glob, **_fingerprint(args.parquet_glob),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(frame.to_string(index=False))
    print(f"Saved label diagnostics: {args.output}")


if __name__ == "__main__":
    main()
