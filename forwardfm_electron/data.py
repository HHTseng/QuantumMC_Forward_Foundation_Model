"""Gate coordinates reused from the trigger-conditioned efficiency branch.

For generated truth x=(p,theta,phi,vx,vy,vz), use
f(x)=(log(1+p),theta,sin(phi),cos(phi),vx,vy,vz). Only training-constant
coordinates may be dropped. Reconstruction failures remain in the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EFFICIENCY_CONTINUOUS_FEATURES = (
    "log1p_gen_p", "gen_theta", "sin_gen_phi", "cos_gen_phi",
    "gen_vx", "gen_vy", "gen_vz",
)
OUTCOME_LABELS = ("unreconstructed", "FD", "FT", "CD", "other")


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    phi = frame["gen_phi"].to_numpy(dtype=np.float64)
    return np.column_stack([
        np.log1p(frame["gen_p"].to_numpy(dtype=np.float64)),
        frame["gen_theta"].to_numpy(dtype=np.float64), np.sin(phi), np.cos(phi),
        frame["gen_vx"].to_numpy(dtype=np.float64),
        frame["gen_vy"].to_numpy(dtype=np.float64),
        frame["gen_vz"].to_numpy(dtype=np.float64),
    ]).astype(np.float32)


def encode_outcomes(frame: pd.DataFrame) -> np.ndarray:
    """C=0 for no reciprocal reconstruction; otherwise C is the REC region.

    Native detector=0 is not a failure indicator: matched FT and other
    regions also have that value. PID disagreement does not imply failure.
    """
    reconstructed = frame["reconstructed"].fillna(False).to_numpy(dtype=bool)
    matched = frame["matched_pindex"].notna().to_numpy() & (
        frame["matched_pindex"].fillna(-1).to_numpy(dtype=np.int64) >= 0
    )
    region = frame["rec_detector_region"].fillna("").to_numpy(dtype=str)
    labels = np.full(len(frame), OUTCOME_LABELS.index("other"), dtype=np.int64)
    labels[~(reconstructed & matched)] = 0
    for name in ("FD", "FT", "CD"):
        labels[reconstructed & matched & (region == name)] = OUTCOME_LABELS.index(name)
    return labels
