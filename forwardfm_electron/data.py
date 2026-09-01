"""All-event generated-electron denominator for trigger efficiency.

For every generated event, the teacher sample contains exactly one PID-11
truth row.  The binary label is

    T = 1{the generated electron yields the valid trigger electron},

and the learned efficiency is eta_T(x_e)=P(T=1|x_e).  No reconstructed
quantity enters x_e and, critically, T=0 events remain in the denominator.
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

from forwardfm_step1.data import Standardizer, connect, split_predicate


EFFICIENCY_CONTINUOUS_FEATURES = (
    "log1p_gen_p",
    "gen_theta",
    "sin_gen_phi",
    "cos_gen_phi",
    "gen_vx",
    "gen_vy",
    "gen_vz",
)
OUTCOME_LABELS = ("unreconstructed", "FD", "FT", "CD", "other")
REQUIRED_COLUMNS = {
    "source_file_id",
    "event_id",
    "mcindex",
    "gen_pid",
    "gen_p",
    "gen_theta",
    "gen_phi",
    "gen_vx",
    "gen_vy",
    "gen_vz",
    "is_generated_trigger_electron",
    "has_valid_trigger_electron",
    "trigger_mcindex",
    "reconstructed",
    "matched_pindex",
    "rec_detector_region",
}


@dataclass
class ElectronEfficiencySplit:
    name: str
    event_keys: np.ndarray
    continuous: np.ndarray
    trigger_target: np.ndarray
    outcome_target: np.ndarray
    raw_gen_p: np.ndarray
    raw_gen_theta: np.ndarray
    raw_gen_phi: np.ndarray
    raw_gen_vz: np.ndarray

    def __len__(self) -> int:
        return len(self.trigger_target)


def assert_schema(con: duckdb.DuckDBPyConnection) -> list[str]:
    columns = [row[0] for row in con.execute("DESCRIBE particles").fetchall()]
    missing = sorted(REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"Dataset is missing required electron columns: {missing}")
    return columns


def denominator_sql() -> str:
    # The audit established gen_pid=11 as the truth-only, one-row-per-event
    # denominator. `is_generated_trigger_electron` is success-derived here and
    # would incorrectly remove every trigger failure.
    return "gen_pid = 11"


def _load_frame(
    con: duckdb.DuckDBPyConnection,
    split: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    configured_limit = config["data"]["max_rows"][split]
    limit = int(configured_limit) if configured_limit is not None else None
    seed = int(config["project"]["seed"])
    qualify = ""
    if limit is not None:
        qualify = f"""
        QUALIFY row_number() OVER (
          ORDER BY hash(source_file_id, event_id, {seed + 41})
        ) <= {limit}
        """
    query = f"""
    SELECT source_file_id, event_id, mcindex,
           hash(source_file_id, event_id) AS event_key,
           gen_pid, gen_p, gen_theta, gen_phi, gen_vx, gen_vy, gen_vz,
           is_generated_trigger_electron, has_valid_trigger_electron,
           trigger_mcindex, reconstructed, matched_pindex, rec_detector_region
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


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Truth-only electron coordinates used by eta_T(x_e).

    f(x_e)=[log(1+p), theta, sin(phi), cos(phi), vx, vy, vz].
    The circular representation makes phi=-pi and +pi adjacent.
    """
    phi = frame["gen_phi"].to_numpy(dtype=np.float64)
    return np.column_stack(
        [
            np.log1p(frame["gen_p"].to_numpy(dtype=np.float64)),
            frame["gen_theta"].to_numpy(dtype=np.float64),
            np.sin(phi),
            np.cos(phi),
            frame["gen_vx"].to_numpy(dtype=np.float64),
            frame["gen_vy"].to_numpy(dtype=np.float64),
            frame["gen_vz"].to_numpy(dtype=np.float64),
        ]
    ).astype(np.float32)


def encode_outcomes(frame: pd.DataFrame) -> np.ndarray:
    reconstructed = frame["reconstructed"].fillna(False).to_numpy(dtype=bool)
    matched = frame["matched_pindex"].notna().to_numpy() & (
        frame["matched_pindex"].fillna(-1).to_numpy(dtype=np.int64) >= 0
    )
    region = frame["rec_detector_region"].fillna("").to_numpy(dtype=str)
    labels = np.full(len(frame), OUTCOME_LABELS.index("other"), dtype=np.int64)
    labels[~(reconstructed & matched)] = OUTCOME_LABELS.index("unreconstructed")
    valid = reconstructed & matched
    for name in ("FD", "FT", "CD"):
        labels[valid & (region == name)] = OUTCOME_LABELS.index(name)
    return labels


def _prepare(
    frame: pd.DataFrame,
    name: str,
    scaler: Standardizer,
    active_feature_indices: np.ndarray,
) -> ElectronEfficiencySplit:
    trigger_target = frame["has_valid_trigger_electron"].to_numpy(dtype=np.float32)
    return ElectronEfficiencySplit(
        name=name,
        event_keys=frame["event_key"].to_numpy(dtype=np.uint64),
        continuous=scaler.transform(feature_matrix(frame)[:, active_feature_indices]),
        trigger_target=trigger_target,
        outcome_target=encode_outcomes(frame),
        raw_gen_p=frame["gen_p"].to_numpy(dtype=np.float32),
        raw_gen_theta=frame["gen_theta"].to_numpy(dtype=np.float32),
        raw_gen_phi=frame["gen_phi"].to_numpy(dtype=np.float32),
        raw_gen_vz=frame["gen_vz"].to_numpy(dtype=np.float32),
    )


def _assert_invariants(frames: dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        if len(frame) != frame["event_key"].nunique():
            raise AssertionError(f"{name} does not contain exactly one electron per event")
        if not np.all(frame["mcindex"].to_numpy() == 0):
            raise AssertionError(f"{name} generated electron is not consistently mcindex=0")
        positive = frame["has_valid_trigger_electron"].to_numpy(dtype=bool)
        associated = (
            frame["trigger_mcindex"].fillna(-1).to_numpy(dtype=np.int64)
            == frame["mcindex"].to_numpy(dtype=np.int64)
        )
        if np.any(positive & ~associated):
            raise AssertionError(f"{name} contains a positive trigger without mcindex association")
    train_classes = np.unique(frames["train"]["has_valid_trigger_electron"])
    if len(train_classes) != 2:
        raise AssertionError("Training denominator must contain trigger successes and failures")
    key_sets = {
        name: set(frame["event_key"].to_numpy(dtype=np.uint64).tolist())
        for name, frame in frames.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if key_sets[left].intersection(key_sets[right]):
            raise AssertionError(f"Event leakage between electron {left} and {right}")


def _fingerprint(parquet_glob: str) -> dict[str, Any]:
    files = sorted(glob.glob(parquet_glob))
    records = [
        {"name": Path(path).name, "bytes": Path(path).stat().st_size}
        for path in files
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
    active_feature_indices: np.ndarray,
) -> dict[str, Any]:
    global_counts = con.execute(
        """
        SELECT count(DISTINCT (source_file_id,event_id)) AS events,
               count(*) FILTER (WHERE gen_pid=11) AS generated_electrons,
               count(*) FILTER (WHERE gen_pid=11 AND has_valid_trigger_electron)
                 AS trigger_successes,
               count(*) FILTER (WHERE gen_pid=11 AND NOT has_valid_trigger_electron)
                 AS trigger_failures,
               count(*) FILTER (
                 WHERE gen_pid=11 AND is_generated_trigger_electron
                   IS DISTINCT FROM has_valid_trigger_electron
               ) AS role_flag_trigger_mismatches
        FROM particles
        """
    ).fetch_df().iloc[0].to_dict()
    split_counts: dict[str, Any] = {}
    for name, frame in frames.items():
        targets = frame["has_valid_trigger_electron"].to_numpy(dtype=bool)
        outcomes = encode_outcomes(frame)
        split_counts[name] = {
            "rows": len(frame),
            "trigger_successes": int(targets.sum()),
            "trigger_failures": int((~targets).sum()),
            "trigger_rate": float(targets.mean()),
            "outcome_counts": {
                label: int(np.sum(outcomes == index))
                for index, label in enumerate(OUTCOME_LABELS)
            },
        }
    train_features = feature_matrix(frames["train"])
    feature_stats = {
        name: {
            "min": float(np.min(train_features[:, index])),
            "max": float(np.max(train_features[:, index])),
            "mean": float(np.mean(train_features[:, index])),
            "std": float(np.std(train_features[:, index])),
        }
        for index, name in enumerate(EFFICIENCY_CONTINUOUS_FEATURES)
    }
    return {
        "dataset_glob": config["data"]["parquet_glob"],
        **_fingerprint(config["data"]["parquet_glob"]),
        "schema_column_count": len(columns),
        "denominator_sql": denominator_sql(),
        "trigger_label": "has_valid_trigger_electron",
        "positive_association_invariant": "trigger_mcindex = mcindex",
        "global_counts": global_counts,
        "split_counts": split_counts,
        "training_feature_stats": feature_stats,
        "active_feature_names": [
            EFFICIENCY_CONTINUOUS_FEATURES[int(index)]
            for index in active_feature_indices
        ],
        "dropped_constant_feature_names": [
            name
            for index, name in enumerate(EFFICIENCY_CONTINUOUS_FEATURES)
            if index not in set(active_feature_indices.tolist())
        ],
        "event_split_overlap_count": 0,
        "audit_note": (
            "is_generated_trigger_electron equals trigger success in this production; "
            "it is not used as the denominator."
        ),
    }


def load_all_electron_splits(
    config: dict[str, Any],
) -> tuple[dict[str, ElectronEfficiencySplit], Standardizer, dict[str, Any]]:
    con = connect(config["data"]["parquet_glob"])
    columns = assert_schema(con)
    frames = {name: _load_frame(con, name, config) for name in ("train", "validation", "test")}
    _assert_invariants(frames)
    train_features = feature_matrix(frames["train"])
    active_feature_indices = np.flatnonzero(train_features.std(axis=0) >= 1e-12)
    if len(active_feature_indices) == 0:
        raise AssertionError("All generated-electron features are constant")
    scaler = Standardizer.fit(train_features[:, active_feature_indices])
    splits = {
        name: _prepare(frame, name, scaler, active_feature_indices)
        for name, frame in frames.items()
    }
    audit = _build_audit(con, frames, columns, config, active_feature_indices)
    con.close()
    return splits, scaler, audit
