"""Native-Parquet samples for eta_h(x)=P(C_h|x_h,s_h,T_e=1).

T_e denotes the generated-electron-associated FD trigger. C_h is reciprocal
reconstruction followed by detector region; no hadron PID, beta, residual,
or fiducial requirement enters this denominator. Training uses phase space;
exclusive rho is an external test dataset, never a selection/tuning set.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from forwardfm_electron.data import (
    EFFICIENCY_CONTINUOUS_FEATURES, OUTCOME_LABELS, encode_outcomes, feature_matrix,
)
from forwardfm_step1.data import Standardizer


SPECIES = (-211, 211, 2212)
FRAME_COLUMNS = """source_file_id,event_id,mcindex,
    hash(source_file_id,event_id) AS event_key,
    gen_pid,gen_p,gen_theta,gen_phi,gen_vx,gen_vy,gen_vz,
    has_valid_trigger_electron,reconstructed,matched_pindex,rec_detector_region"""


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
    return f"gen_pid={int(species)} AND has_valid_trigger_electron"


def load_frame(con, species: int, split: str, limit: int, split_seed: int) -> pd.DataFrame:
    """Choose event-disjoint phase splits and deterministic, uniform hash caps.

    Network seed never enters this selection. Rho uses an independently
    generated dataset and a uniform cap over its entire T_e denominator.
    """
    if limit <= 0:
        raise ValueError("Sample limits must be positive")
    bucket = f"hash(source_file_id,event_id,{int(split_seed)}) % 10000"
    predicates = {
        "train": f"{bucket}<8000", "validation": f"{bucket}>=8000 AND {bucket}<9000",
        "test": f"{bucket}>=9000", "rho": "true",
    }
    if split not in predicates:
        raise ValueError(f"Unknown split: {split}")
    frame = con.execute(f"""
        SELECT {FRAME_COLUMNS} FROM particles
        WHERE {denominator_sql(species)} AND ({predicates[split]})
        ORDER BY hash(source_file_id,event_id,{int(split_seed)+17}),source_file_id,event_id
        LIMIT {int(limit)}
    """).fetch_df()
    if frame.empty:
        raise ValueError(f"Empty {split} sample for species {species}")
    if frame.duplicated(["source_file_id", "event_id"]).any():
        raise AssertionError(f"Expected 1 generated species per event: {species}, {split}")
    reconstructed = frame["reconstructed"].to_numpy(bool)
    valid_index = frame["matched_pindex"].fillna(-1).to_numpy(np.int64) >= 0
    if not np.array_equal(reconstructed, valid_index):
        raise AssertionError("Reciprocal flag and reconstruction index disagree")
    raw = frame[["gen_p", "gen_theta", "gen_phi", "gen_vx", "gen_vy", "gen_vz"]].to_numpy(float)
    if not np.isfinite(raw).all() or (raw[:, 0] < 0).any():
        raise ValueError("Invalid generated coordinates")
    if ((raw[:, 1] < 0) | (raw[:, 1] > np.pi)).any():
        raise ValueError("Polar angles must be in radians")
    return frame


def prepare_splits(frames: dict[str, pd.DataFrame]):
    """Fit the coordinate standardizer on phase-space training rows only."""
    phase_keys = {
        name: set(zip(frame["source_file_id"], frame["event_id"]))
        for name, frame in frames.items() if name in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if phase_keys[left] & phase_keys[right]:
            raise AssertionError(f"Event overlap: {left}/{right}")
    values = feature_matrix(frames["train"])
    active = np.flatnonzero(values.std(axis=0, dtype=np.float64) >= 1e-12)
    if not len(active):
        raise ValueError("No nonconstant training coordinates")
    scaler = Standardizer.fit(values[:, active])
    split_objects, audits = {}, {}
    for name, frame in frames.items():
        features = feature_matrix(frame)
        labels = encode_outcomes(frame)
        keys = frame["event_key"].to_numpy(np.uint64)
        split_objects[name] = HadronSplit(
            name, keys, scaler.transform(features[:, active]), labels,
            frame["gen_p"].to_numpy(np.float32), frame["gen_theta"].to_numpy(np.float32),
            frame["gen_phi"].to_numpy(np.float32), frame["gen_vz"].to_numpy(np.float32),
        )
        outside = np.any((features[:, active] < values[:, active].min(axis=0)) |
                         (features[:, active] > values[:, active].max(axis=0)), axis=1)
        audits[name] = {
            "rows": len(frame), "outcome_counts": {
                label: int(np.count_nonzero(labels == j)) for j, label in enumerate(OUTCOME_LABELS)
            },
            "event_order_sha256": hashlib.sha256(keys.astype("<u8").tobytes()).hexdigest(),
            "outside_training_feature_box_rows": int(outside.sum()),
        }
    dropped = {name: float(values[0, j]) for j, name in enumerate(EFFICIENCY_CONTINUOUS_FEATURES)
               if j not in active}
    metadata = {
        "feature_names": [EFFICIENCY_CONTINUOUS_FEATURES[j] for j in active],
        "dropped_constant_feature_values": dropped,
        "splits": audits, "phase_event_split_overlap": 0,
        "feature_box_note": "Coordinate bounding box only; not a joint-density support estimate.",
        "training_feature_min": values[:, active].min(axis=0).tolist(),
        "training_feature_max": values[:, active].max(axis=0).tolist(),
    }
    return split_objects, scaler, metadata
