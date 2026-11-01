"""Normalize Alex's native exports without discarding generated particles.

Each row is a generated particle i in event E=(file_id,event). Reconstruction
R_i=1 means the authoritative mutual MC_i -> REC_j -> MC_i association.
The event trigger is

    T_e(E) = exists i: is_trigger_particle_i and pid_gen_i=11 and detector_i=1.

An optional identified-electron variant additionally requires pid_rec_i=11.
Both are distinct from the native any-negative-status trigger. Event labels
are broadcast to every generated row, preserving efficiency denominators.

Native angles are degrees and native residuals are GEN-REC. The canonical
view uses radians and REC-GEN residuals, computed from the selected momentum
view rather than trusting the exported residual columns. The immutable native
table remains accessible as ``particles_native``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


REQUIRED_NATIVE_COLUMNS = {
    "file_id", "event", "mcrow", "recrow", "reconstructed", "pid_gen", "pid_rec",
    "triggered", "trigger_recrow", "trigger_pid_rec", "is_trigger_particle",
    "detector", "status_rec", "px_gen", "py_gen", "pz_gen", "p_gen",
    "theta_gen", "phi_gen", "vx_gen", "vy_gen", "vz_gen",
    "px_rec", "py_rec", "pz_rec", "p_rec", "theta_rec", "phi_rec",
    "vx_rec", "vy_rec", "vz_rec", "beta_rec",
}
RAW_MOMENTUM_COLUMNS = {
    "p_rec_uncorrected", "px_rec_uncorrected", "py_rec_uncorrected",
    "pz_rec_uncorrected",
}


def connect_native(
    path: str | Path,
    momentum: str = "raw",
    trigger: str = "associated_fd",
    threads: int = 4,
) -> duckdb.DuckDBPyConnection:
    """Expose native/canonical views and 1-row ``transfer_metadata``.

    ``raw`` uses the pre-XGBoost backups when present. If backups are absent,
    it uses the native export and explicitly records that its pre-correction
    provenance is unavailable. ``corrected`` always uses native p_rec.
    No fiducial, PID-quality, residual, trigger, or reconstruction cuts are
    applied to the canonical particle population.
    """
    if momentum not in {"raw", "corrected"}:
        raise ValueError("momentum must be 'raw' or 'corrected'")
    if trigger not in {"associated_fd", "identified_fd"}:
        raise ValueError("trigger must be 'associated_fd' or 'identified_fd'")
    if int(threads) < 1:
        raise ValueError("threads must be positive")
    con = duckdb.connect()
    try:
        con.execute(f"SET threads={int(threads)}")
        con.read_parquet(str(path)).create_view("particles_native")
        columns = {row[0] for row in con.execute("DESCRIBE particles_native").fetchall()}
        missing = REQUIRED_NATIVE_COLUMNS - columns
        if missing:
            raise ValueError(f"Native export is missing required columns: {sorted(missing)}")
        raw_columns = RAW_MOMENTUM_COLUMNS & columns
        if raw_columns and raw_columns != RAW_MOMENTUM_COLUMNS:
            raise ValueError("Native export contains an incomplete raw-momentum backup")

        has_raw = bool(raw_columns)
        raw_p = "p_rec_uncorrected" if has_raw else "p_rec"
        suffix = "_uncorrected" if momentum == "raw" and has_raw else ""
        selected_p = f"p_rec{suffix}"
        raw_source = "uncorrected_backup" if has_raw else "native_export_no_raw_backup"
        # Keep only positive trigger events in this small lookup table.
        # Grouping prevents duplicate electron rows from multiplying the join;
        # audit_native separately reports duplicate generated row keys.
        con.execute("""
            CREATE TEMP TABLE transfer_triggers AS
            SELECT file_id,event,true AS associated_fd,
                   bool_or(pid_rec=11) AS identified_fd
            FROM particles_native
            WHERE pid_gen=11 AND is_trigger_particle=1 AND detector=1
            GROUP BY file_id,event
        """)
        con.execute("""
            CREATE TEMP TABLE transfer_metadata (
                momentum_view VARCHAR, trigger_definition VARCHAR,
                raw_momentum_source VARCHAR, has_raw_backup BOOLEAN)
        """)
        con.execute(
            "INSERT INTO transfer_metadata VALUES (?,?,?,?)",
            [momentum, trigger, raw_source, has_raw],
        )
        excluded = {"reconstructed", "delta_p", "delta_theta", "delta_phi"} & columns
        native_select = "p.* EXCLUDE (" + ",".join(sorted(excluded)) + ")"
        additions = []
        if "was_corrected" not in columns:
            additions.append("0 AS was_corrected")
        if "charge_gen" not in columns:
            additions.append("CASE WHEN pid_gen IN (11,-211) THEN -1 WHEN pid_gen IN (211,2212) THEN 1 ELSE 0 END AS charge_gen")
        if "charge_rec" not in columns:
            additions.append("0 AS charge_rec")
        if "chi2pid_rec" not in columns:
            additions.append("CAST(NULL AS DOUBLE) AS chi2pid_rec")
        if "genmatch_quality" not in columns:
            additions.append("CAST(NULL AS DOUBLE) AS genmatch_quality")
        if "recmatch_quality" not in columns:
            additions.append("CAST(NULL AS DOUBLE) AS recmatch_quality")
        optional_columns = ",".join(additions)
        if optional_columns:
            optional_columns = "," + optional_columns
        # C_i is a reconstruction outcome, independent of the detector=0
        # shorthand, which also includes successfully reconstructed FT rows.
        con.execute(f"""
            CREATE VIEW particles AS
            SELECT {native_select}{optional_columns},
              p.file_id AS source_file_id,p.event AS event_id,p.mcrow AS mcindex,
              p.recrow AS matched_pindex,p.trigger_recrow AS trigger_pindex,
              p.pid_gen AS gen_pid,p.pid_rec AS rec_pid,
              CAST(p.px_gen AS DOUBLE) AS gen_px,
              CAST(p.py_gen AS DOUBLE) AS gen_py,
              CAST(p.pz_gen AS DOUBLE) AS gen_pz,
              CAST(p.p_gen AS DOUBLE) AS gen_p,
              radians(CAST(p.theta_gen AS DOUBLE)) AS gen_theta,
              radians(CAST(p.phi_gen AS DOUBLE)) AS gen_phi,
              CAST(p.vx_gen AS DOUBLE) AS gen_vx,
              CAST(p.vy_gen AS DOUBLE) AS gen_vy,
              CAST(p.vz_gen AS DOUBLE) AS gen_vz,
              CAST(p.px_rec{suffix} AS DOUBLE) AS rec_px,
              CAST(p.py_rec{suffix} AS DOUBLE) AS rec_py,
              CAST(p.pz_rec{suffix} AS DOUBLE) AS rec_pz,
              CAST(p.{selected_p} AS DOUBLE) AS rec_p,
              radians(CAST(p.theta_rec AS DOUBLE)) AS rec_theta,
              radians(CAST(p.phi_rec AS DOUBLE)) AS rec_phi,
              CAST(p.vx_rec AS DOUBLE) AS rec_vx,
              CAST(p.vy_rec AS DOUBLE) AS rec_vy,
              CAST(p.vz_rec AS DOUBLE) AS rec_vz,
              CAST(p.beta_rec AS DOUBLE) AS rec_beta,
              p.status_rec AS rec_status,
              abs(p.status_rec)//1000 AS rec_detector_code,
              p.reconstructed<>0 AS reconstructed,
              p.reconstructed<>0 AS match_reciprocal,
              CASE WHEN p.reconstructed=0 THEN NULL
                WHEN abs(p.status_rec)>=2000 AND abs(p.status_rec)<4000 THEN 'FD'
                WHEN abs(p.status_rec)>=4000 AND abs(p.status_rec)<5000 THEN 'CD'
                WHEN abs(p.status_rec)>=1000 AND abs(p.status_rec)<2000 THEN 'FT'
                ELSE 'other' END AS rec_detector_region,
              p.reconstructed<>0 AND abs(p.status_rec)>=1000 AND abs(p.status_rec)<2000 AS rec_in_ft,
              p.reconstructed<>0 AND abs(p.status_rec)>=2000 AND abs(p.status_rec)<4000 AS rec_in_fd,
              p.reconstructed<>0 AND abs(p.status_rec)>=4000 AND abs(p.status_rec)<5000 AS rec_in_cd,
              false AS rec_in_band,
              CAST(p.{selected_p} AS DOUBLE)-CAST(p.p_gen AS DOUBLE) AS delta_p,
              radians(CAST(p.theta_rec AS DOUBLE)-CAST(p.theta_gen AS DOUBLE)) AS delta_theta,
              radians(((CAST(p.phi_rec AS DOUBLE)-CAST(p.phi_gen AS DOUBLE)+180)%360+360)%360-180) AS delta_phi,
              CAST(p.{raw_p} AS DOUBLE) AS raw_rec_p,
              CAST(p.p_rec AS DOUBLE) AS corrected_rec_p,
              CAST(p.{raw_p} AS DOUBLE)-CAST(p.p_gen AS DOUBLE) AS raw_delta_p,
              CAST(p.p_rec AS DOUBLE)-CAST(p.p_gen AS DOUBLE) AS corrected_delta_p,
              p.triggered<>0 AS has_valid_trigger,
              coalesce(t.associated_fd,false) AS has_associated_fd_trigger,
              coalesce(t.identified_fd,false) AS has_identified_fd_trigger,
              coalesce(t.{trigger},false) AS has_valid_trigger_electron,
              p.is_trigger_particle<>0 AS is_generated_trigger_particle,
              p.pid_gen=11 AND p.is_trigger_particle<>0 AS is_generated_trigger_electron,
              p.pid_gen IN (-211,211,2212) AS is_hadron,
              p.pid_gen IN (-211,211,2212) AND coalesce(t.{trigger},false)
                AS usable_for_hadron_response_training,
              '{momentum}' AS momentum_view,'{trigger}' AS trigger_definition,
              '{raw_source}' AS raw_momentum_source
            FROM particles_native p
            LEFT JOIN transfer_triggers t USING (file_id,event)
        """)
        # Aliases below depend on optional native columns, so add them after
        # those columns have been normalized once.
        con.execute("ALTER VIEW particles RENAME TO particles_normalized")
        con.execute("""
            CREATE VIEW particles AS SELECT *,charge_rec AS rec_charge,
              chi2pid_rec AS rec_chi2pid FROM particles_normalized
        """)
        return con
    except Exception:
        con.close()
        raise


def audit_native(con: duckdb.DuckDBPyConnection) -> dict:
    """Full-table denominator/label checks; no training or selection occurs."""
    result = {}
    for key, query in {
        "metadata": "SELECT * FROM transfer_metadata",
        "counts": """SELECT count(*) AS particle_rows,
            count(DISTINCT (file_id,event)) AS events,
            count(DISTINCT file_id) AS source_files FROM particles_native""",
        "species": """SELECT gen_pid,count(*) AS generated,
            count(*) FILTER(WHERE reconstructed) AS reconstructed,
            count(*) FILTER(WHERE has_valid_trigger_electron) AS triggered,
            count(*) FILTER(WHERE has_valid_trigger_electron AND rec_in_fd) AS triggered_fd,
            count(*) FILTER(WHERE has_valid_trigger_electron AND rec_in_cd) AS triggered_cd,
            count(*) FILTER(WHERE has_valid_trigger_electron AND rec_in_ft) AS triggered_ft
            FROM particles GROUP BY gen_pid ORDER BY gen_pid""",
        "triggers": """SELECT count(*) AS generated_electrons,
            count(*) FILTER(WHERE has_valid_trigger) AS any_negative_status,
            count(*) FILTER(WHERE has_associated_fd_trigger) AS associated_fd,
            count(*) FILTER(WHERE has_identified_fd_trigger) AS identified_fd
            FROM particles WHERE gen_pid=11""",
        "violations": """SELECT
            count(*) FILTER(WHERE reconstructed IS NULL OR
              reconstructed IS DISTINCT FROM (recrow>=0)) AS reconstructed_index,
            count(*) FILTER(WHERE is_trigger_particle=1 AND
              (reconstructed<>1 OR triggered<>1 OR recrow<>trigger_recrow OR status_rec>=0))
              AS selected_trigger,
            count(*) FILTER(WHERE (detector=1 AND NOT(abs(status_rec)>=2000 AND abs(status_rec)<4000))
              OR (detector=2 AND NOT(abs(status_rec)>=4000 AND abs(status_rec)<5000)))
              AS detector_status,
            count(*) FILTER(WHERE reconstructed=0 AND (isfinite(p_rec) OR isfinite(beta_rec)))
              AS unmatched_finite_response
            FROM particles_native""",
        "duplicate_generated_keys": """SELECT count(*) AS keys FROM (
            SELECT file_id,event,mcrow FROM particles_native
            GROUP BY file_id,event,mcrow HAVING count(*)<>1)""",
    }.items():
        cursor = con.execute(query)
        names = [column[0] for column in cursor.description]
        values = [dict(zip(names, row)) for row in cursor.fetchall()]
        result[key] = values if key == "species" else values[0]
    return result
