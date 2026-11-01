"""Checkpoint-driven probabilities; apply only after the electron trigger gate."""

from __future__ import annotations

import numpy as np
import torch

from forwardfm_electron.data import EFFICIENCY_CONTINUOUS_FEATURES, OUTCOME_LABELS, feature_matrix
from forwardfm_step1.data import Standardizer
from .model import HadronEfficiencyNet


@torch.no_grad()
def predict_checkpoint(checkpoint_path, frame, device="cpu", batch_size=8192):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != "trigger_conditioned_hadron_efficiency":
        raise ValueError("Not a trigger-conditioned hadron efficiency checkpoint")
    if checkpoint.get("format_version") != 1 or checkpoint.get("conditioning") != "has_valid_trigger_electron = true":
        raise ValueError("Unsupported checkpoint format or conditioning")
    if checkpoint.get("outcome_labels") != list(OUTCOME_LABELS):
        raise ValueError("Incompatible outcome vocabulary/order")
    if frame.empty or "gen_pid" not in frame or not np.all(frame["gen_pid"] == checkpoint["species"]):
        raise ValueError("Every input must have the checkpoint's generated species")
    values = frame.copy()
    # Unlike filling arbitrary missing truth with zero, only exact constants
    # learned on training data may be supplied from checkpoint metadata.
    for name, constant in checkpoint["dropped_constant_feature_values"].items():
        if name in ("gen_vx", "gen_vy") and name not in values:
            values[name] = constant
    required = ("gen_p", "gen_theta", "gen_phi", "gen_vx", "gen_vy", "gen_vz")
    if set(required) - set(values):
        raise ValueError(f"Missing truth features: {sorted(set(required) - set(values))}")
    raw = values[list(required)].to_numpy(float)
    if not np.isfinite(raw).all() or (raw[:, 0] < 0).any() or ((raw[:, 1] < 0) | (raw[:, 1] > np.pi)).any():
        raise ValueError("Invalid generated truth kinematics")
    candidates = feature_matrix(values)
    for name, constant in checkpoint["dropped_constant_feature_values"].items():
        if not np.allclose(candidates[:, EFFICIENCY_CONTINUOUS_FEATURES.index(name)], constant, atol=1e-8, rtol=0):
            raise ValueError(f"Outside training support: constant feature {name}")
    indices = [EFFICIENCY_CONTINUOUS_FEATURES.index(name) for name in checkpoint["feature_names"]]
    features = Standardizer.from_dict(checkpoint["feature_scaler"]).transform(candidates[:, indices])
    model = HadronEfficiencyNet(**checkpoint["architecture"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    probabilities = np.concatenate([
        model(torch.from_numpy(features[start:start + batch_size]).to(device)).softmax(-1).cpu().numpy()
        for start in range(0, len(features), batch_size)
    ])
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Nonfinite efficiency probability")
    result = frame.copy()
    for j, name in enumerate(OUTCOME_LABELS):
        result[f"prob_{name}"] = probabilities[:, j]
    result["prob_reconstructed_any"] = 1 - probabilities[:, 0]
    result["prob_reconstructed_fd"] = probabilities[:, 1]
    return result
