"""Keep the failures needed for P(C_h | x_h, s_h, T=1).

T is the event's valid-electron-trigger proxy. C_h is a truth-matched
reconstruction outcome, NOT a correct-PID label. In particular, a proton
reconstructed as a pion remains a reconstruction success.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from forwardfm_electron.data import (
    EFFICIENCY_CONTINUOUS_FEATURES,
    OUTCOME_LABELS,
    _fingerprint,
    encode_outcomes,
    feature_matrix,
)
from forwardfm_step1.data import Standardizer, connect, split_predicate

SPECIES = (-211, 211, 2212)
REQUIRED_COLUMNS = {
    "source_file_id", "event_id", "mcindex", "gen_pid", "gen_p", "gen_theta",
    "gen_phi", "gen_vx", "gen_vy", "gen_vz", "has_valid_trigger_electron",
    "reconstructed", "matched_pindex", "rec_detector_region",
}


@dataclass
class HadronSplit:
    name: str
    event_keys: np.ndarray
    continuous: np.ndarray
    outcome_target: np.ndarray
    raw_gen_p: np.ndarray
    raw_gen_theta: np.ndarray
    raw_gen_phi: np.ndarray
    raw_gen_vz: np.ndarray

    def __len__(self) -> int:
        return len(self.outcome_target)


def denominator_sql(species: int) -> str:
    if species not in SPECIES:
        raise ValueError(f"Unsupported generated hadron species: {species}")
    # Never use reconstructed, usable_for_hadron_response_training, rec_pid,
    # beta, reciprocal matching or residual cuts in this denominator.
    return f"gen_pid = {int(species)} AND has_valid_trigger_electron"


def assert_invariants(frames: dict[str, pd.DataFrame]) -> None:
    key_sets = {}
    for name, frame in frames.items():
        if frame.empty:
            raise ValueError(f"Empty hadron denominator for {name}")
        if frame.duplicated(["source_file_id", "event_id", "mcindex"]).any():
            raise AssertionError(f"Duplicate generated-particle keys in {name}")
        if frame["event_key"].nunique() != len(frame):
            raise AssertionError(f"Expected 1 generated species row per event in {name}")
        reconstructed = frame["reconstructed"]
        matched = frame["matched_pindex"].notna() & (frame["matched_pindex"].fillna(-1) >= 0)
        if reconstructed.isna().any() or np.any(reconstructed.to_numpy(bool) != matched):
            raise AssertionError(f"Inconsistent reconstructed/matched labels in {name}")
        raw = frame[["gen_p", "gen_theta", "gen_phi", "gen_vx", "gen_vy", "gen_vz"]].to_numpy(float)
        if not np.isfinite(raw).all() or (raw[:, 0] < 0).any():
            raise ValueError(f"Nonfinite or negative generated momentum/features in {name}")
        if ((raw[:, 1] < 0) | (raw[:, 1] > np.pi)).any():
            raise ValueError(f"Generated theta outside [0, pi] in {name}")
        key_sets[name] = set(frame["event_key"].to_numpy(np.uint64).tolist())
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if key_sets[left].intersection(key_sets[right]):
            raise AssertionError(f"Event leakage between {left} and {right}")


def load_hadron_splits(config: dict[str, Any], species: int):
    denominator = denominator_sql(species)
    con = connect(config["data"]["parquet_glob"])
    try:
        columns = {row[0] for row in con.execute("DESCRIBE particles").fetchall()}
        if REQUIRED_COLUMNS - columns:
            raise ValueError(f"Missing hadron columns: {sorted(REQUIRED_COLUMNS - columns)}")
        # Audit before selecting T=1: NULL trigger labels must not silently
        # disappear, and all rows of the same physical event must agree on T.
        bad_trigger = con.execute("""
            SELECT source_file_id, event_id FROM particles
            GROUP BY source_file_id, event_id
            HAVING count(DISTINCT has_valid_trigger_electron) != 1
                OR count(*) FILTER (WHERE has_valid_trigger_electron IS NULL) > 0
            LIMIT 1
        """).fetchone()
        if bad_trigger:
            raise AssertionError(f"Null/inconsistent event trigger label: {bad_trigger}")
        duplicate = con.execute(f"""
            SELECT source_file_id,event_id FROM particles WHERE gen_pid={int(species)}
            GROUP BY source_file_id,event_id HAVING count(*) != 1 LIMIT 1
        """).fetchone()
        if duplicate:
            raise AssertionError(f"Duplicate generated species/event: {duplicate}")
        n_events = int(con.execute("SELECT count(DISTINCT (source_file_id,event_id)) FROM particles").fetchone()[0])
        n_species = int(con.execute(f"SELECT count(*) FROM particles WHERE gen_pid={int(species)}").fetchone()[0])
        if n_species != n_events:
            raise AssertionError(f"Expected 1 generated species row in every event: {n_species} != {n_events}")
        counts = con.execute(f"""
            SELECT count(*) AS generated_rows,
              count(*) FILTER (WHERE has_valid_trigger_electron) AS triggered_rows,
              count(*) FILTER (WHERE NOT has_valid_trigger_electron) AS untriggered_rows,
              count(*) FILTER (WHERE has_valid_trigger_electron AND reconstructed
                    AND matched_pindex >= 0) AS reconstructed_rows
            FROM particles WHERE gen_pid={int(species)}
        """).fetch_df().iloc[0].to_dict()
        split_config = deepcopy(config)
        split_seed = int(config["data"].get("split_seed", 20260822))
        split_config["project"]["seed"] = split_seed
        frames = {}
        for name in ("train", "validation", "test"):
            limit = config["data"]["max_rows"][name]
            limit_sql = "" if limit is None else f"LIMIT {int(limit)}"
            frames[name] = con.execute(f"""
                SELECT source_file_id,event_id,mcindex,
                  hash(source_file_id,event_id) AS event_key,
                  gen_pid,gen_p,gen_theta,gen_phi,gen_vx,gen_vy,gen_vz,
                  has_valid_trigger_electron,reconstructed,matched_pindex,rec_detector_region
                FROM particles
                WHERE {denominator} AND ({split_predicate(name, split_config)})
                ORDER BY source_file_id,event_id,mcindex {limit_sql}
            """).fetch_df()
        assert_invariants(frames)
        train_features = feature_matrix(frames["train"])
        active = np.flatnonzero(train_features.std(axis=0, dtype=np.float64) >= 1e-12)
        if not len(active):
            raise ValueError("No nonconstant truth features")
        scaler = Standardizer.fit(train_features[:, active])
        splits, split_counts = {}, {}
        for name, frame in frames.items():
            labels = encode_outcomes(frame)
            keys = frame["event_key"].to_numpy(np.uint64)
            splits[name] = HadronSplit(
                name, keys, scaler.transform(feature_matrix(frame)[:, active]), labels,
                frame["gen_p"].to_numpy(np.float32),
                frame["gen_theta"].to_numpy(np.float32),
                frame["gen_phi"].to_numpy(np.float32),
                frame["gen_vz"].to_numpy(np.float32),
            )
            split_counts[name] = {
                "rows": len(frame),
                "outcome_counts": {label: int(np.sum(labels == j)) for j, label in enumerate(OUTCOME_LABELS)},
                "particle_order_sha256": hashlib.sha256(keys.astype("<u8").tobytes()).hexdigest(),
            }
        dropped = {
            label: float(train_features[0, j])
            for j, label in enumerate(EFFICIENCY_CONTINUOUS_FEATURES) if j not in active
        }
        audit = {
            "species": species, "conditioning": "has_valid_trigger_electron = true",
            "denominator_sql": denominator, "split_seed": split_seed,
            "global_counts": {"events": n_events, **{k: int(v) for k, v in counts.items()}},
            "split_counts": split_counts, "event_split_overlap_count": 0,
            "active_feature_names": [EFFICIENCY_CONTINUOUS_FEATURES[j] for j in active],
            "dropped_constant_feature_values": dropped,
            "outcome_labels": list(OUTCOME_LABELS),
            "label_definition": "reconstructed AND matched_pindex IS NOT NULL AND matched_pindex >= 0; then detector region",
            "dataset_glob": config["data"]["parquet_glob"],
            **_fingerprint(config["data"]["parquet_glob"]),
        }
        return splits, scaler, audit
    finally:
        con.close()
