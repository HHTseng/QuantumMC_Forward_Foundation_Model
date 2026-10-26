#!/usr/bin/env python3
"""Sample the split-asinh response using checkpoint-recorded coordinates."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forwardfm_step1.data_asinh_dp import (
    SPECIES, GEN_MASS_GEV, TARGET_COLUMNS, TARGET_TRANSFORM, Standardizer, build_features,
    inverse_targets_from_model,
)
from forwardfm_step1.model_split_dp import ConditionalSplitMDN, sample_split_standardized_residuals
from forwardfm_step1.training import choose_device


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "split_delta_p_asinh" or checkpoint.get("target_transform") != TARGET_TRANSFORM:
        raise ValueError("Expected a split-asinh checkpoint with explicit 0.020 GeV transform metadata")
    if tuple(checkpoint["species_pids"]) != SPECIES:
        raise ValueError("This response checkpoint supports pi-, pi+, and proton")
    if tuple(checkpoint["target_names"]) != TARGET_COLUMNS or checkpoint["particle_masses_gev"] != {str(k):v for k,v in GEN_MASS_GEV.items()}:
        raise ValueError("Checkpoint target order or particle masses are unsupported")
    model = ConditionalSplitMDN(**checkpoint["architecture"])
    if model.n_continuous != len(checkpoint["feature_names"]):
        raise ValueError("Checkpoint feature order and architecture disagree")
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def sample_frame(checkpoint_path, frame, device, seed):
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    feature_scaler = Standardizer.from_dict(checkpoint["feature_scaler"])
    target_scaler = Standardizer.from_dict(checkpoint["target_scaler"])
    feature_names = tuple(checkpoint["feature_names"])
    features = feature_scaler.transform(build_features(frame, feature_names))
    species_index = np.array([SPECIES.index(int(pid)) for pid in frame.gen_pid], dtype=np.int64)
    generator = torch.Generator(device=device.type).manual_seed(seed) if device.type in ("cpu", "cuda") else None
    samples, pids = [], []
    for start in range(0, len(frame), 8192):
        sl = slice(start, start + 8192)
        output = model(torch.from_numpy(features[sl]).to(device), torch.from_numpy(species_index[sl]).to(device))
        standardized = sample_split_standardized_residuals(output, generator).cpu().numpy()
        # Undo affine scaling before delta_p = 0.020 GeV * sinh(u_p).
        physical = inverse_targets_from_model(target_scaler.inverse(standardized))
        samples.append(physical)
        pids.append(torch.multinomial(torch.softmax(output.pid_logits, -1), 1, generator=generator).cpu().numpy().ravel())
    residuals = np.concatenate(samples)
    if not np.isfinite(residuals).all():
        raise FloatingPointError("Nonfinite response draws; no clipping was applied")
    residuals[:, 2] = (residuals[:, 2] + np.pi) % (2*np.pi) - np.pi
    result = frame.copy()
    for i, name in enumerate(checkpoint["target_names"]):
        result[f"sampled_{name}"] = residuals[:, i]
    result["sampled_rec_p"] = frame.gen_p.to_numpy() + residuals[:, 0]
    result["sampled_rec_theta"] = frame.gen_theta.to_numpy() + residuals[:, 1]
    result["sampled_rec_phi"] = (frame.gen_phi.to_numpy() + residuals[:, 2] + np.pi) % (2*np.pi) - np.pi
    masses = np.array([checkpoint["particle_masses_gev"][str(int(pid))] for pid in frame.gen_pid])
    momentum = frame.gen_p.to_numpy()
    result["sampled_rec_beta"] = momentum / np.sqrt(momentum**2 + masses**2) + residuals[:, 3]
    vocabulary = [*checkpoint["rec_pid_vocabulary"], "OTHER"]
    result["sampled_rec_pid"] = [vocabulary[i] for i in np.concatenate(pids)]
    result["sample_is_physical"] = (result.sampled_rec_p > 0) & result.sampled_rec_theta.between(0, np.pi)
    # An unbounded asinh-Gaussian may generate finite but enormous residuals.
    # Expose the diagnostic threshold explicitly; do not clip/reject draws.
    result["sample_abs_delta_p_gt_10GeV"] = np.abs(residuals[:, 0]) > 10.0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    result = sample_frame(args.checkpoint, pd.read_csv(args.input), choose_device(args.device), args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"wrote {len(result)} responses to {args.output}")


if __name__ == "__main__":
    main()
