#!/usr/bin/env python3
"""Evaluate and Bernoulli-sample trigger acceptance from generated electrons."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forwardfm_full.data import TRIGGER_FEATURES, feature_matrix
from forwardfm_full.efficiency import TriggerEfficiencyNet
from forwardfm_step1.data import Standardizer
from forwardfm_step1.training import choose_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != "trigger_electron_efficiency":
        raise ValueError("Checkpoint is not a trigger-electron efficiency model")
    if tuple(checkpoint["feature_names"]) != TRIGGER_FEATURES:
        raise ValueError("Checkpoint feature order is unsupported")
    frame = pd.read_csv(args.input)
    required = {"gen_pid", "gen_p", "gen_theta", "gen_phi", "gen_vz"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    if not np.all(frame["gen_pid"].to_numpy(dtype=int) == 11):
        raise ValueError("Every input row must be a generated electron (PDG 11)")

    scaler = Standardizer.from_dict(checkpoint["feature_scaler"])
    values = scaler.transform(feature_matrix(frame))
    model = TriggerEfficiencyNet(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(torch.from_numpy(values).to(device))).cpu().numpy()
    result = frame.copy()
    result["predicted_trigger_probability"] = probability
    result["sampled_trigger"] = rng.random(len(result)) < probability
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"wrote {len(result):,} rows to {args.output}")


if __name__ == "__main__":
    main()

