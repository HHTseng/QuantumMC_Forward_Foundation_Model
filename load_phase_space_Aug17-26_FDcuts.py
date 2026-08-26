"""Fiducial FD selection for the Aug 17--26 Forward-FM skim.

``particles`` applies exactly three cuts:
  C_i = FD,  rec_theta < 33 deg,  -5.5 cm < gen_vz < -0.5 cm.
Angles are stored in radians; ``gen_vz`` is in centimetres.  Because the
selection uses reconstructed quantities, it is only for the conditional
response Delta_i = (delta_p, delta_theta, delta_phi), not trigger or
reconstruction-efficiency denominators.

The fiducial cuts do not reject non-reciprocal matches or rare sentinel/
pathological momenta, either of which can dominate residual moments.
``particles_reciprocal`` adds both quality predicates for residual fitting;
``particles`` retains the unmodified three-cut sample for audit.
"""

from pathlib import Path

import duckdb

# Resolve data relative to this module, independent of the caller's cwd.
DATA_GLOB = str(
    Path(__file__).resolve().parent.parent
    / "phase-space_parquet-Aug17-26" / "particle_responses" / "*.parquet"
)

EVENT_KEY_COLS = "source_file_id, event_id"  # Globally unique event key.
EVENT_KEY = f"({EVENT_KEY_COLS})"

# Fiducial-cut parameters; SQL below derives from these values.
THETA_MAX_DEG = 33.0
VZ_MIN_CM = -5.5
VZ_MAX_CM = -0.5

CUT_SQL = f"""
    rec_detector_region = 'FD'
    AND rec_theta < radians({THETA_MAX_DEG})
    AND gen_vz > {VZ_MIN_CM}
    AND gen_vz < {VZ_MAX_CM}
"""

# Optional residual-quality predicates; not part of the fiducial definition.
RECIPROCAL_SQL = "match_reciprocal"
SANE_MOMENTUM_SQL = "rec_pid <> 0 AND rec_beta > -99"


def connect(data_glob: str = DATA_GLOB) -> duckdb.DuckDBPyConnection:
    """Expose the full, fiducial, and quality-filtered fiducial views."""
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW particles_all AS SELECT * FROM '{data_glob}'")
    con.execute(f"CREATE OR REPLACE VIEW particles AS SELECT * FROM particles_all WHERE {CUT_SQL}")
    con.execute(
        "CREATE OR REPLACE VIEW particles_reciprocal AS "
        f"SELECT * FROM particles WHERE {RECIPROCAL_SQL} AND {SANE_MOMENTUM_SQL}"
    )
    return con


def cutflow(con: duckdb.DuckDBPyConnection):
    """Return cumulative counts and total/marginal survival fractions."""
    stages = [
        ("0. all rows (no cut)", "TRUE"),
        ("1. + reconstructed", "reconstructed"),
        ("2. + FD only", "rec_detector_region = 'FD'"),
        (
            f"3. + rec_theta < {THETA_MAX_DEG:g} deg",
            f"rec_detector_region = 'FD' AND rec_theta < radians({THETA_MAX_DEG})",
        ),
        (
            f"4. + {VZ_MIN_CM:g} < gen_vz < {VZ_MAX_CM:g} cm  [FINAL]",
            CUT_SQL,
        ),
    ]
    total = con.execute("SELECT count(*) FROM particles_all").fetchone()[0]
    rows, prev = [], None
    for label, pred in stages:
        n = con.execute(f"SELECT count(*) FROM particles_all WHERE {pred}").fetchone()[0]
        rows.append(
            {
                "stage": label,
                "kept": n,
                "frac_of_all": n / total,
                "frac_of_prev": (n / prev) if prev else 1.0,
            }
        )
        prev = n
    return rows


def summary(con: duckdb.DuckDBPyConnection) -> None:
    print("=" * 78)
    print("CUT FLOW (sequential, cumulative)")
    print("=" * 78)
    for r in cutflow(con):
        print(
            f"{r['stage']:<44s} {r['kept']:>12,}  "
            f"{r['frac_of_all']:>7.2%} of all  {r['frac_of_prev']:>7.2%} of prev"
        )

    print()
    print("=" * 78)
    print("SELECTED SAMPLE")
    print("=" * 78)
    n_rows, n_events = con.execute(
        f"SELECT count(*), count(DISTINCT {EVENT_KEY}) FROM particles"
    ).fetchone()
    print(f"rows={n_rows:,}  events_represented={n_events:,}")

    print()
    print("By species (gen_pid), selected vs all:")
    print(
        con.execute(
            """
            SELECT a.gen_pid,
                   a.n_all,
                   coalesce(s.n_sel, 0) AS n_selected,
                   coalesce(s.n_sel, 0) / a.n_all AS keep_frac
            FROM (SELECT gen_pid, count(*) n_all FROM particles_all GROUP BY 1) a
            LEFT JOIN (SELECT gen_pid, count(*) n_sel FROM particles GROUP BY 1) s
              ON a.gen_pid = s.gen_pid
            ORDER BY a.gen_pid
            """
        ).df().to_string(index=False)
    )

    print()
    print("Electron vs hadron rows in the selected sample:")
    print(
        con.execute(
            "SELECT is_hadron, count(*) c FROM particles GROUP BY 1 ORDER BY 1"
        ).df().to_string(index=False)
    )

    print()
    print("Trigger composition of the selected sample:")
    print(
        con.execute(
            """
            SELECT has_valid_trigger_electron, count(*) c
            FROM particles GROUP BY 1 ORDER BY 1
            """
        ).df().to_string(index=False)
    )

    print()
    print("=" * 78)
    print("RESIDUAL QUALITY CHECK")
    print("=" * 78)
    print("Fiducial vs. reciprocal, non-sentinel residual moments:")
    print(
        con.execute(
            """
            SELECT 'particles (3 cuts only)' AS sample, count(*) n,
                   round(avg(delta_p),5) mean_dp, round(stddev(delta_p),5) std_dp,
                   round(degrees(avg(delta_theta)),5) mean_dtheta_deg,
                   round(degrees(avg(delta_phi)),5) mean_dphi_deg
            FROM particles
            UNION ALL
            SELECT 'particles_reciprocal (opt-in)', count(*),
                   round(avg(delta_p),5), round(stddev(delta_p),5),
                   round(degrees(avg(delta_theta)),5),
                   round(degrees(avg(delta_phi)),5)
            FROM particles_reciprocal
            """
        ).df().to_string(index=False)
    )
    print()
    print("Breakdown of what the opt-in filter removes:")
    print(
        con.execute(
            f"""
            SELECT count(*) FILTER (WHERE NOT {RECIPROCAL_SQL})        AS non_reciprocal,
                   count(*) FILTER (WHERE NOT ({SANE_MOMENTUM_SQL}))   AS pathological_mom,
                   count(*) FILTER (WHERE abs(delta_p) > 10)           AS abs_dp_gt_10gev,
                   count(*) FILTER (WHERE delta_p IS NULL)             AS null_delta_p,
                   count(*)                                            AS total
            FROM particles
            """
        ).df().to_string(index=False)
    )

    print()
    print(f"rec_theta range in selection (deg, must be < {THETA_MAX_DEG:g}):")
    print(
        con.execute(
            "SELECT round(min(degrees(rec_theta)),4) lo, round(max(degrees(rec_theta)),4) hi FROM particles"
        ).df().to_string(index=False)
    )
    print(f"gen_vz range in selection (cm, must be within {VZ_MIN_CM:g}..{VZ_MAX_CM:g}):")
    print(
        con.execute(
            "SELECT round(min(gen_vz),4) lo, round(max(gen_vz),4) hi FROM particles"
        ).df().to_string(index=False)
    )


def fd_residual_training_view(
    con: duckdb.DuckDBPyConnection, quality_filtered: bool = True
) -> duckdb.DuckDBPyRelation:
    """Rows for P(Delta_i, rec_pid | x_i, C_i=FD, T=1) for hadrons.

    By default require reciprocal matching and non-sentinel momentum fields.
    Set ``quality_filtered=False`` to retain the exact fiducial selection.
    """
    src = "particles_reciprocal" if quality_filtered else "particles"
    return con.execute(
        f"SELECT * FROM {src} WHERE usable_for_hadron_response_training"
    )


if __name__ == "__main__":
    summary(connect())
