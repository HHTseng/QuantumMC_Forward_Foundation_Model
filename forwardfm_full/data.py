r"""All-event generated-electron view for \(P(T=1\mid x_e)\).

Every generated event contributes its PID-11 truth row, including T=0.  No
reconstructed quantity is used as an input feature.
"""

from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from forwardfm_step1.data import (
    Standardizer,
    connect,
    data_order_seed,
    data_split_seed,
    split_predicate,
)

from .labels import reconstruction_exists, response_quality_pass, trigger_electron_pass


TRIGGER_FEATURES = (
    "log1p_gen_p",
    "gen_theta",
    "sin_gen_phi",
    "cos_gen_phi",
    "gen_vz",
)
REQUIRED_COLUMNS = {
    "source_file_id",
    "event_id",
    "mcindex",
    "gen_pid",
    "gen_p",
    "gen_theta",
    "gen_phi",
    "gen_vz",
    "has_valid_trigger_electron",
    "trigger_mcindex",
    "reconstructed",
    "matched_pindex",
    "match_reciprocal",
    "rec_detector_region",
    "delta_p",
    "delta_theta",
    "delta_phi",
}


@dataclass
class TriggerSplit:
    name: str
    event_keys: np.ndarray
    continuous: np.ndarray
    trigger_target: np.ndarray
    electron_reconstructed: np.ndarray
    raw_gen_p: np.ndarray
    raw_gen_theta: np.ndarray
    raw_gen_phi: np.ndarray
    raw_gen_vz: np.ndarray
    feature_names: tuple[str, ...] = TRIGGER_FEATURES

    def __len__(self) -> int:
        return len(self.trigger_target)


def denominator_sql() -> str:
    """Truth-only denominator: one generated electron per event."""
    return "gen_pid = 11"


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    r"""Return \(\Phi_e=(\log(1+p),\theta,\sin\phi,\cos\phi,v_z)\)."""
    phi = frame["gen_phi"].to_numpy(dtype=np.float64)
    return np.column_stack(
        [
            np.log1p(frame["gen_p"].to_numpy(dtype=np.float64)),
            frame["gen_theta"].to_numpy(dtype=np.float64),
            np.sin(phi),
            np.cos(phi),
            frame["gen_vz"].to_numpy(dtype=np.float64),
        ]
    ).astype(np.float32)


def _assert_schema(con: duckdb.DuckDBPyConnection) -> list[str]:
    columns = [row[0] for row in con.execute("DESCRIBE particles").fetchall()]
    missing = sorted(REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"Dataset is missing trigger-efficiency columns: {missing}")
    return columns


def _load_frame(
    con: duckdb.DuckDBPyConnection,
    split: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    configured_limit = config["data"]["max_rows"][split]
    limit = int(configured_limit) if configured_limit is not None else None
    seed = data_order_seed(config)
    qualify = ""
    if limit is not None:
        qualify = f"""
        QUALIFY row_number() OVER (
          ORDER BY hash(source_file_id, event_id, {seed + 41})
        ) <= {limit}
        """
    query = f"""
    SELECT source_file_id, event_id, mcindex,
           gen_pid, gen_p, gen_theta, gen_phi, gen_vz,
           has_valid_trigger_electron, trigger_mcindex,
           reconstructed, matched_pindex, match_reciprocal,
           rec_detector_region, delta_p, delta_theta, delta_phi
    FROM particles
    WHERE {denominator_sql()}
      AND {split_predicate(split, config)}
    {qualify}
    ORDER BY hash(source_file_id, event_id, {seed + 53})
    """
    frame = con.execute(query).fetch_df()
    if frame.empty:
        raise ValueError(f"Electron denominator produced no rows for {split!r}")
    return frame


def _event_keys(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [f"{source}:{event}" for source, event in zip(frame.source_file_id, frame.event_id)],
        dtype=object,
    )


def _assert_invariants(frames: dict[str, pd.DataFrame]) -> None:
    key_sets: dict[str, set[str]] = {}
    for name, frame in frames.items():
        keys = _event_keys(frame)
        if len(keys) != len(set(keys.tolist())):
            raise AssertionError(f"{name} does not contain exactly one electron per event")
        if not np.all(frame["mcindex"].to_numpy(dtype=np.int64) == 0):
            raise AssertionError(f"{name} generated electron is not consistently mcindex=0")
        trigger = trigger_electron_pass(frame)
        reconstructed = reconstruction_exists(frame)
        if np.any(trigger & ~reconstructed):
            raise AssertionError(f"{name} has T=1 without R_e=1")
        key_sets[name] = set(keys.tolist())
    if len(np.unique(trigger_electron_pass(frames["train"]))) != 2:
        raise AssertionError("Training denominator must contain T=0 and T=1")
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if key_sets[left].intersection(key_sets[right]):
            raise AssertionError(f"Event leakage between {left} and {right}")


def _prepare(
    frame: pd.DataFrame,
    name: str,
    scaler: Standardizer,
) -> TriggerSplit:
    return TriggerSplit(
        name=name,
        event_keys=_event_keys(frame),
        continuous=scaler.transform(feature_matrix(frame)),
        trigger_target=trigger_electron_pass(frame).astype(np.float32),
        electron_reconstructed=reconstruction_exists(frame).astype(np.int64),
        raw_gen_p=frame["gen_p"].to_numpy(dtype=np.float32),
        raw_gen_theta=frame["gen_theta"].to_numpy(dtype=np.float32),
        raw_gen_phi=frame["gen_phi"].to_numpy(dtype=np.float32),
        raw_gen_vz=frame["gen_vz"].to_numpy(dtype=np.float32),
    )


def _dataset_fingerprint(parquet_glob: str) -> dict[str, Any]:
    records = [
        {"name": Path(path).name, "bytes": Path(path).stat().st_size}
        for path in sorted(glob.glob(parquet_glob))
    ]
    payload = json.dumps(records, sort_keys=True).encode("utf-8")
    return {
        "dataset_file_count": len(records),
        "dataset_total_bytes": sum(row["bytes"] for row in records),
        "dataset_metadata_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _build_audit(
    con: duckdb.DuckDBPyConnection,
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    global_row = con.execute(
        """
        SELECT count(DISTINCT (source_file_id,event_id)) AS events,
               count(*) FILTER (WHERE gen_pid=11) AS generated_electrons,
               count(*) FILTER (WHERE gen_pid=11 AND has_valid_trigger_electron)
                 AS trigger_successes,
               count(*) FILTER (WHERE gen_pid=11 AND NOT has_valid_trigger_electron)
                 AS trigger_failures,
               count(*) FILTER (
                 WHERE gen_pid=11 AND has_valid_trigger_electron
                   AND (NOT reconstructed OR matched_pindex IS NULL OR matched_pindex < 0)
               ) AS trigger_without_reconstruction
        FROM particles
        """
    ).fetch_df().iloc[0].to_dict()
    split_counts: dict[str, Any] = {}
    for name, frame in frames.items():
        trigger = trigger_electron_pass(frame)
        reconstructed = reconstruction_exists(frame)
        quality = response_quality_pass(frame)
        split_counts[name] = {
            "rows": len(frame),
            "trigger_successes": int(trigger.sum()),
            "trigger_failures": int((~trigger).sum()),
            "trigger_rate": float(trigger.mean()),
            "electron_reconstructed": int(reconstructed.sum()),
            "response_quality_pass": int(quality.sum()),
            "t_r_contingency": {
                f"T{int(t)}_R{int(r)}": int(np.sum((trigger == t) & (reconstructed == r)))
                for t in (False, True)
                for r in (False, True)
            },
        }
    train_features = feature_matrix(frames["train"])
    return {
        "dataset_glob": config["data"]["parquet_glob"],
        **_dataset_fingerprint(config["data"]["parquet_glob"]),
        "schema_column_count": len(columns),
        "denominator_sql": denominator_sql(),
        "feature_names": list(TRIGGER_FEATURES),
        "trigger_label": "has_valid_trigger_electron",
        "reconstruction_label": "reconstructed AND matched_pindex >= 0",
        "positive_association_invariant": "trigger_mcindex = mcindex",
        "global_counts": global_row,
        "split_counts": split_counts,
        "training_feature_range": {
            feature: {
                "min": float(train_features[:, index].min()),
                "max": float(train_features[:, index].max()),
            }
            for index, feature in enumerate(TRIGGER_FEATURES)
        },
        "model_seed": int(config["project"]["seed"]),
        "data_split_seed": data_split_seed(config),
        "data_order_seed": data_order_seed(config),
        "event_split_overlap_count": 0,
    }


def load_trigger_splits(
    config: dict[str, Any],
) -> tuple[dict[str, TriggerSplit], Standardizer, dict[str, Any]]:
    con = connect(config["data"]["parquet_glob"])
    columns = _assert_schema(con)
    frames = {
        name: _load_frame(con, name, config)
        for name in ("train", "validation", "test")
    }
    _assert_invariants(frames)
    scaler = Standardizer.fit(feature_matrix(frames["train"]))
    splits = {name: _prepare(frame, name, scaler) for name, frame in frames.items()}
    audit = _build_audit(con, frames, columns, config)
    con.close()
    return splits, scaler, audit
