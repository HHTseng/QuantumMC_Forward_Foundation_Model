"""On-load detector labels derived from immutable particle-response rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reconstruction_exists(frame: pd.DataFrame) -> np.ndarray:
    """Return R=1 when a reconstructed candidate is actually associated."""
    reconstructed = frame["reconstructed"].fillna(False).to_numpy(dtype=bool)
    matched = frame["matched_pindex"].notna().to_numpy() & (
        frame["matched_pindex"].fillna(-1).to_numpy(dtype=np.int64) >= 0
    )
    return reconstructed & matched


def response_quality_pass(frame: pd.DataFrame) -> np.ndarray:
    """Return Q=1 for an associated, reciprocal, finite response row."""
    reciprocal = frame["match_reciprocal"].fillna(False).to_numpy(dtype=bool)
    finite = np.ones(len(frame), dtype=bool)
    for name in ("delta_p", "delta_theta", "delta_phi"):
        finite &= np.isfinite(frame[name].to_numpy(dtype=np.float64, na_value=np.nan))
    return reconstruction_exists(frame) & reciprocal & finite


def trigger_electron_pass(frame: pd.DataFrame) -> np.ndarray:
    """Return T=1 from the producer's accepted-trigger label.

    Positive labels must also point to the same generated electron through
    ``trigger_mcindex``; a violation is a dataset error rather than a negative.
    """
    accepted = frame["has_valid_trigger_electron"].fillna(False).to_numpy(dtype=bool)
    associated = frame["trigger_mcindex"].fillna(-1).to_numpy(dtype=np.int64) == frame[
        "mcindex"
    ].to_numpy(dtype=np.int64)
    if np.any(accepted & ~associated):
        raise AssertionError("Accepted trigger electron lacks its truth association")
    return accepted

