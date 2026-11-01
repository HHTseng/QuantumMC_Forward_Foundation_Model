"""Frozen-checkpoint transfer of the selected FD hadron response.

The measured object is q(Delta, PID_rec | x, T_e=1, R_mut=1, D=FD, F),
where F is the original checkpoint's fiducial/quality selection. This module
does not estimate an efficiency. Native raw/corrected comparisons use the
same particles selected with the raw momentum residual.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from forwardfm_step1 import data_asinh_dp as data
from forwardfm_step1.evaluation_split_dp import fit_gaussian_core
from forwardfm_step1.model_split_dp import SplitModelOutput, sample_split_standardized_residuals
from sample_split_dp_asinh import load_checkpoint


SPECIES_LABELS = {-211: "pi-", 211: "pi+", 2212: "proton"}
MOMENTUM_EDGES = np.arange(0., 10.)


def load_response_frame(checkpoint_path, native_parquet=None, parquet_glob=None,
                        max_rows_per_species=100000, sample_seed=20261101,
                        trigger="associated_fd", split="test"):
    """Keep the training selection and deterministic, bounded test sampling.

    August data use the saved event-disjoint test partition. The independent
    rho sample has no model-training partition; hash ranking only limits the
    evaluation cost and is identical on both machines/checkpoints.
    """
    config_path = Path(checkpoint_path).parent / "resolved_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint config is required: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    if native_parquet:
        from .data import connect_native
        con = connect_native(native_parquet, momentum="raw", trigger=trigger)
        provenance = con.execute("SELECT raw_momentum_source,has_raw_backup FROM transfer_metadata").fetchone()
        if not provenance[1]:
            con.close()
            raise ValueError("Raw/corrected response transfer requires pre-XGBoost momentum backups; this native export has none")
        extra = "raw_rec_p, corrected_rec_p, was_corrected"
        partition = "TRUE"
        dataset = "native_external"
    else:
        source = parquet_glob or config["data"]["parquet_glob"]
        con = data.connect(source)
        extra = "rec_p AS raw_rec_p, rec_p AS corrected_rec_p, 0 AS was_corrected"
        partition = data.split_predicate(split, config)
        dataset = "august_heldout"
    try:
        # For native input delta_p is canonical raw REC-GEN here. Correction
        # is a target view, never a way to select a more favorable sample.
        selection = data.selection_sql(config)
        query = f"""
          SELECT source_file_id,event_id,mcindex,gen_pid,gen_p,gen_theta,gen_phi,
                 rec_beta,rec_pid,delta_p,delta_theta,delta_phi,{extra}
          FROM particles
          WHERE {selection} AND ({partition})
          QUALIFY row_number() OVER (
            PARTITION BY gen_pid
            ORDER BY hash(source_file_id,event_id,mcindex,{int(sample_seed)})
          ) <= {int(max_rows_per_species)}
          ORDER BY gen_pid,hash(source_file_id,event_id,mcindex,{int(sample_seed)+1})
        """
        frame = con.execute(query).fetch_df()
    finally:
        con.close()
    if frame.empty:
        raise ValueError("The frozen-response selection contains no particles")
    mass = frame.gen_pid.map(data.GEN_MASS_GEV).to_numpy(dtype=float)
    p = frame.gen_p.to_numpy(dtype=float)
    frame["delta_beta"] = frame.rec_beta - p / np.sqrt(p*p + mass*mass)
    keys = frame[["source_file_id", "event_id", "mcindex", "gen_pid"]]
    metadata = {
        "dataset": dataset, "native_parquet": str(native_parquet) if native_parquet else None,
        "parquet_glob": str(source) if not native_parquet else None,
        "trigger_definition": trigger if native_parquet else "saved_August_trigger_proxy",
        "raw_momentum_source": provenance[0] if native_parquet else "August_original_export",
        "split": "external" if native_parquet else split,
        "selection_sql_raw_momentum": selection, "partition_sql": partition,
        "sample_seed": int(sample_seed), "max_rows_per_species": int(max_rows_per_species),
        "particle_order_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes()).hexdigest(),
    }
    return frame, metadata


def inverse_sample_coordinates(standardized, target_scaler):
    """Undo z=(u-mu)/sigma, then Delta p=0.020 GeV sinh(u_p).

    Float64 avoids imposing a float32 overflow boundary on the unbounded
    asinh-Gaussian. Extreme/nonfinite draws are counted, not clipped/rejected.
    """
    physical = (np.asarray(standardized, dtype=np.float64)
                * target_scaler.scale.astype(np.float64)
                + target_scaler.mean.astype(np.float64))
    with np.errstate(over="ignore", invalid="ignore"):
        physical[:, 0] = data.DELTA_P_ASINH_SCALE_GEV * np.sinh(physical[:, 0])
    physical[:, 2] = (physical[:, 2] + np.pi) % (2*np.pi) - np.pi
    return physical


@torch.no_grad()
def predict_response(checkpoint_path, frame, device, sample_seed=20261101, batch_size=8192):
    """Use saved coordinates; draw with a CPU RNG on either GPU architecture."""
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    feature_scaler = data.Standardizer.from_dict(checkpoint["feature_scaler"])
    target_scaler = data.Standardizer.from_dict(checkpoint["target_scaler"])
    features = feature_scaler.transform(data.build_features(frame, tuple(checkpoint["feature_names"])))
    species = np.array([data.SPECIES.index(int(pid)) for pid in frame.gen_pid], dtype=np.int64)
    generator = torch.Generator().manual_seed(int(sample_seed))
    draws, probabilities = [], []
    started = time.perf_counter()
    for start in range(0, len(frame), batch_size):
        sl = slice(start, start+batch_size)
        output = model(torch.from_numpy(features[sl]).to(device),
                       torch.from_numpy(species[sl]).to(device))
        # CPU random numbers make the draw sequence independent of CUDA RNG
        # implementations. Neural outputs may still differ at roundoff level.
        cpu_output = SplitModelOutput(**{k: v.cpu() for k, v in vars(output).items()})
        draws.append(sample_split_standardized_residuals(cpu_output, generator).numpy())
        probabilities.append(torch.softmax(cpu_output.pid_logits, -1).numpy())
    elapsed = time.perf_counter() - started
    return (inverse_sample_coordinates(np.concatenate(draws), target_scaler),
            np.concatenate(probabilities), checkpoint,
            {"inference_seconds": elapsed, "particles_per_second": len(frame)/elapsed})


def robust_summary(values):
    """Quantiles of finite draws; the missing probability mass is explicit."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"finite_fraction": 0., "q16": None, "q50": None, "q84": None, "sigma68": None}
    q16, q50, q84 = np.quantile(finite, [.16, .5, .84])
    return {"finite_fraction": float(len(finite)/len(values)),
            "q16": float(q16), "q50": float(q50), "q84": float(q84),
            "sigma68": float((q84-q16)/2)}


def response_tables(frame, sampled, probabilities, vocabulary, min_bin_count=200):
    """Return detector-response and conditional PID closure, never top-1 accuracy.

    PID closure compares mean softmax probabilities to reconstructed class
    fractions. The correct-ID MAE averages eligible 1-GeV bins equally.
    """
    species = frame.gen_pid.to_numpy(dtype=int)
    momentum = frame.gen_p.to_numpy(dtype=float)
    raw = frame.loc[:, data.TARGET_COLUMNS].to_numpy(dtype=float)
    corrected = raw.copy()
    corrected[:, 0] = frame.corrected_rec_p.to_numpy(dtype=float) - momentum
    views = {"raw": raw, "corrected": corrected}
    summary, resolution, pid_rows, channels = [], [], [], []
    for pid, label in SPECIES_LABELS.items():
        mask = species == pid
        if not mask.any():
            continue
        for view, observed in views.items():
            for target_index, target in enumerate(data.TARGET_COLUMNS):
                row = {"generated_pid": pid, "species": label, "teacher": view,
                       "target": target, "n": int(mask.sum())}
                for prefix, values in (("observed", observed[mask, target_index]),
                                       ("fm", sampled[mask, target_index])):
                    row.update({f"{prefix}_{k}": v for k, v in robust_summary(values).items()})
                row["median_difference"] = (row["fm_q50"]-row["observed_q50"]
                    if row["fm_q50"] is not None and row["observed_q50"] is not None else None)
                width = row["observed_sigma68"]
                row["sigma68_ratio"] = (row["fm_sigma68"]/width
                    if width and row["fm_sigma68"] is not None else None)
                summary.append(row)
        for bin_index, (low, high) in enumerate(zip(MOMENTUM_EDGES[:-1], MOMENTUM_EDGES[1:])):
            cell = mask & (momentum >= low) & ((momentum <= high) if bin_index == len(MOMENTUM_EDGES)-2 else (momentum < high))
            n = int(cell.sum())
            if not n:
                continue
            common = {"generated_pid": pid, "species": label, "p_low_GeV": low,
                      "p_high_GeV": high, "n": n, "eligible": n >= min_bin_count}
            total_variation = 0.
            truth_index = np.array([vocabulary.index(int(p)) if int(p) in vocabulary else len(vocabulary)
                                    for p in frame.rec_pid.to_numpy()[cell]])
            class_names = [*vocabulary, "OTHER"]
            for class_index, class_name in enumerate(class_names):
                observed_fraction = float(np.mean(truth_index == class_index))
                probs = probabilities[cell, class_index]
                fm_fraction = float(probs.mean())
                total_variation += abs(observed_fraction-fm_fraction)/2
                channels.append({**common, "reconstructed_pid": class_name,
                                 "observed_fraction": observed_fraction,
                                 "fm_fraction": fm_fraction,
                                 "observed_binomial_se": float(np.sqrt(observed_fraction*(1-observed_fraction)/n)),
                                 "fm_mean_probability_se": float(probs.std(ddof=1)/np.sqrt(n)) if n > 1 else None})
                if class_name == pid:
                    correct = channels[-1].copy()
                    correct.pop("reconstructed_pid")
                    correct["absolute_error_pp"] = 100*abs(fm_fraction-observed_fraction)
            if pid not in vocabulary:
                raise ValueError(f"Checkpoint vocabulary lacks generated species {pid}")
            correct["total_variation"] = float(total_variation)
            pid_rows.append(correct)
            for view, observed in views.items():
                row = {**common, "teacher": view}
                for prefix, values in (("observed", observed[cell, 0]), ("fm", sampled[cell, 0])):
                    row.update({f"{prefix}_{k}": v for k, v in robust_summary(values).items()})
                    fit = fit_gaussian_core(values) if n >= min_bin_count else None
                    for name in ("mu", "sigma", "mu_error", "sigma_error"):
                        row[f"{prefix}_gaussian_{name}_MeV"] = fit[name] if fit else None
                a, b = row["observed_gaussian_sigma_MeV"], row["fm_gaussian_sigma_MeV"]
                row["gaussian_sigma_ratio"] = b/a if a and b else None
                resolution.append(row)
    return {"response_summary": summary, "momentum_resolution": resolution,
            "pid_closure": pid_rows, "pid_channels": channels}, raw, corrected


def evaluate_response(checkpoint_path, output_dir, native_parquet=None, parquet_glob=None,
                      device="cpu", max_rows_per_species=100000, sample_seed=20261101,
                      trigger="associated_fd", split="test", min_bin_count=200):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    frame, metadata = load_response_frame(checkpoint_path, native_parquet, parquet_glob,
        max_rows_per_species, sample_seed, trigger, split)
    sampled, probabilities, checkpoint, timing = predict_response(
        checkpoint_path, frame, torch.device(device), sample_seed)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-6):
        raise FloatingPointError("PID softmax probabilities must be finite and normalized")
    tables, raw, corrected = response_tables(frame, sampled, probabilities,
        checkpoint["rec_pid_vocabulary"], min_bin_count)
    for name, table in tables.items():
        pd.DataFrame(table).to_csv(output_dir/f"{name}.csv", index=False)
    species = frame.gen_pid.to_numpy(dtype=int)
    p = frame.gen_p.to_numpy(dtype=float)
    theta_deg = np.rad2deg(frame.gen_theta.to_numpy(dtype=float))
    # This is a coarse generator-production bounding box, not an empirical
    # learned-density support guarantee for the selected FD checkpoint.
    inside_box = ((p >= .3) & (p <= np.where(species == 2212, 7., 9.))
                  & (theta_deg >= 5.) & (theta_deg <= 130.))
    np.savez_compressed(output_dir/"response_events.npz", species=species, gen_p=p,
        gen_theta=frame.gen_theta.to_numpy(), gen_phi=frame.gen_phi.to_numpy(),
        file_id=frame.source_file_id.to_numpy(), event_id=frame.event_id.to_numpy(),
        mcindex=frame.mcindex.to_numpy(), raw_truth=raw, corrected_truth=corrected,
        sampled_response=sampled, pid_rec=frame.rec_pid.to_numpy(),
        pid_probabilities=probabilities,
        pid_vocabulary=np.array([str(v) for v in [*checkpoint["rec_pid_vocabulary"], "OTHER"]]),
        inside_aug_generated_box=inside_box, was_corrected=frame.was_corrected.to_numpy())
    pid_frame = pd.DataFrame(tables["pid_closure"])
    aggregate = {}
    for pid, label in SPECIES_LABELS.items():
        rows = pid_frame[(pid_frame.generated_pid == pid) & pid_frame.eligible]
        sel = species == pid
        aggregate[label] = {
            "n": int(sel.sum()), "eligible_momentum_bins": int(len(rows)),
            "correct_id_mae_pp": float(rows.absolute_error_pp.mean()) if len(rows) else None,
            "pid_tv_particle_weighted": float(np.average(rows.total_variation, weights=rows.n)) if len(rows) else None,
            "outside_coarse_aug_generated_box_fraction": float(np.mean(~inside_box[sel])),
            "fm_nonfinite_response_fraction": float(np.mean(~np.isfinite(sampled[sel]).all(axis=1))),
            "fm_abs_delta_p_gt_10_fraction": float(np.mean(np.abs(sampled[sel, 0]) > 10)),
            "fm_negative_reconstructed_p_fraction": float(np.mean(p[sel]+sampled[sel, 0] <= 0)),
        }
    metrics = {**metadata, **timing, "species": aggregate,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "feature_names": checkpoint["feature_names"], "target_names": checkpoint["target_names"],
        "min_bin_count": int(min_bin_count), "weighting": "unweighted particle counts",
        "sampling": "1 draw/particle; CPU RNG, float64 physical inverse; no tail clipping",
        "selection_note": "Both teacher views use the same raw-selected particles; PID and beta are unchanged by XGBoost",
        "environment": {"platform": platform.platform(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "device": str(device),
                        "gpu": torch.cuda.get_device_name(torch.device(device)) if str(device).startswith("cuda") else None}}
    (output_dir/"metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False)+"\n")
    return metrics
