#!/usr/bin/env python3
"""Predict and sample trigger acceptance for generated electron truth rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forwardfm_electron.data import EFFICIENCY_CONTINUOUS_FEATURES, OUTCOME_LABELS, feature_matrix
from forwardfm_electron.model import ElectronEfficiencyNet
from forwardfm_step1.data import Standardizer
from forwardfm_step1.training import choose_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != "trigger_electron_efficiency":
        raise ValueError("Checkpoint is not a trigger-electron efficiency model")
    model = ElectronEfficiencyNet(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    frame = pd.read_csv(args.input)
    required = {"gen_pid", "gen_p", "gen_theta", "gen_phi", "gen_vz"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing generated-electron columns: {missing}")
    if not np.all(frame["gen_pid"].to_numpy(dtype=int) == 11):
        raise ValueError("Every row must be the generated event electron (gen_pid=11)")
    # Constant transverse-vertex columns are optional at inference; the
    # training audit removed them from this checkpoint.
    for name in ("gen_vx", "gen_vy"):
        if name not in frame:
            frame[name] = 0.0
    feature_indices = np.asarray(
        [EFFICIENCY_CONTINUOUS_FEATURES.index(name) for name in checkpoint["feature_names"]],
        dtype=np.int64,
    )
    scaler = Standardizer.from_dict(checkpoint["feature_scaler"])
    features = scaler.transform(feature_matrix(frame)[:, feature_indices])
    with torch.no_grad():
        output = model(torch.from_numpy(features).to(device))
        trigger_probability = torch.sigmoid(output.trigger_logit).cpu().numpy()
        outcome_probability = torch.softmax(output.outcome_logits, dim=-1).cpu().numpy()
    result = frame.copy()
    result["predicted_trigger_probability"] = trigger_probability
    result["sampled_has_valid_trigger_electron"] = rng.random(len(result)) < trigger_probability
    for index, label in enumerate(checkpoint["label_definition"]["outcome_classes"]):
        result[f"predicted_outcome_probability_{label}"] = outcome_probability[:, index]
    result["most_likely_outcome"] = [OUTCOME_LABELS[index] for index in outcome_probability.argmax(axis=1)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"wrote {len(result):,} trigger-efficiency predictions to {args.output}")


if __name__ == "__main__":
    main()
