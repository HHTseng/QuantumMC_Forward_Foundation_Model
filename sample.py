#!/usr/bin/env python3
"""Sample reconstructed-like FD kinematics from generated truth particles.

The inference physics chain is

    x=(p_gen,theta_gen,phi_gen,s_gen)
      -> neural conditional density p(Delta,s_rec|x,T=1,C=FD)
      -> draw (Delta,s_rec)
      -> p_rec=p_gen+Delta p,
         theta_rec=theta_gen+Delta theta,
         phi_rec=wrap(phi_gen+Delta phi).

This script begins after trigger and FD-outcome decisions; it does not generate
T or C and therefore cannot turn an arbitrary truth particle into a complete
detector event by itself.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forwardfm_step1.data import BETA_TARGET_COLUMN, SPECIES, Standardizer, generated_beta
from forwardfm_step1.model import ConditionalMDN, sample_standardized_residuals
from forwardfm_step1.training import choose_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample FD residuals and REC PID from a trained step-one checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="CSV with gen_pid/gen_p/gen_theta/gen_phi")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def build_features(frame: pd.DataFrame) -> np.ndarray:
    required = {"gen_pid", "gen_p", "gen_theta", "gen_phi"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    phi = frame["gen_phi"].to_numpy(dtype=np.float64)
    return np.column_stack(
        [
            np.log1p(frame["gen_p"].to_numpy(dtype=np.float64)),
            frame["gen_theta"].to_numpy(dtype=np.float64),
            np.sin(phi),
            np.cos(phi),
        ]
    ).astype(np.float32)


def wrap_phi(phi: np.ndarray) -> np.ndarray:
    """Return wrap(phi)=((phi+pi) mod 2pi)-pi in the principal interval."""
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ConditionalMDN(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    frame = pd.read_csv(args.input)
    species_to_index = {pid: index for index, pid in enumerate(checkpoint["species_pids"])}
    unsupported = sorted(set(int(pid) for pid in frame.gen_pid).difference(species_to_index))
    if unsupported:
        raise ValueError(f"Unsupported generated PIDs: {unsupported}; supported={list(SPECIES)}")
    species_index = np.asarray([species_to_index[int(pid)] for pid in frame.gen_pid], dtype=np.int64)
    feature_scaler = Standardizer.from_dict(checkpoint["feature_scaler"])
    target_scaler = Standardizer.from_dict(checkpoint["target_scaler"])
    features = feature_scaler.transform(build_features(frame))

    generator = (
        torch.Generator(device=device.type).manual_seed(args.seed)
        if device.type in {"cpu", "cuda"}
        else None
    )
    with torch.no_grad():
        output = model(
            torch.from_numpy(features).to(device),
            torch.from_numpy(species_index).to(device),
        )
        normalized_residuals = sample_standardized_residuals(output, generator=generator)
        residuals = target_scaler.inverse(normalized_residuals.cpu().numpy())
        pid_index = torch.multinomial(
            torch.softmax(output.pid_logits, dim=-1), 1, generator=generator
        ).squeeze(1).cpu().numpy()

    vocabulary = [*checkpoint["rec_pid_vocabulary"], "OTHER"]
    sampled_pid = [vocabulary[int(index)] for index in pid_index]
    result = frame.copy()
    target_names = tuple(
        checkpoint.get("target_names", ("delta_p", "delta_theta", "delta_phi"))
    )
    if len(target_names) != residuals.shape[1]:
        raise ValueError(
            "Checkpoint target_names do not match the model target dimension: "
            f"{len(target_names)} versus {residuals.shape[1]}"
        )
    for target_index, target_name in enumerate(target_names):
        values = residuals[:, target_index]
        if target_name == "delta_phi":
            values = wrap_phi(values)
        result[f"sampled_{target_name}"] = values
    result["sampled_rec_pid"] = sampled_pid
    # Invert the residual definitions Delta q = q_rec-q_gen.
    result["sampled_rec_p"] = result["gen_p"] + result["sampled_delta_p"]
    result["sampled_rec_theta"] = result["gen_theta"] + result["sampled_delta_theta"]
    result["sampled_rec_phi"] = wrap_phi(
        result["gen_phi"].to_numpy() + result["sampled_delta_phi"].to_numpy()
    )
    if BETA_TARGET_COLUMN in target_names:
        beta_reference = generated_beta(
            result["gen_p"].to_numpy(dtype=np.float64),
            result["gen_pid"].to_numpy(dtype=np.int64),
        )
        result["sampled_beta_gen_reference"] = beta_reference
        result["sampled_rec_beta"] = beta_reference + result["sampled_delta_beta"]
        # This is an audit flag, not a clipping operation. The generated sample
        # remains available for studying likelihood leakage beyond the fitted
        # teacher domain.
        beta_config = checkpoint.get("beta_response", {})
        beta_min = float(beta_config.get("rec_beta_min_exclusive", 0.0))
        beta_max = float(beta_config.get("rec_beta_max_inclusive", 1.2))
        result["sampled_beta_in_training_domain"] = (
            (result["sampled_rec_beta"] > beta_min)
            & (result["sampled_rec_beta"] <= beta_max)
        )
    result["sample_is_physical"] = (
        (result["sampled_rec_p"] > 0)
        & (result["sampled_rec_theta"] >= 0)
        & (result["sampled_rec_theta"] <= np.pi)
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    invalid = int((~result["sample_is_physical"]).sum())
    print(f"wrote {len(result):,} samples to {args.output}; physical_guard_failures={invalid}")


if __name__ == "__main__":
    main()
