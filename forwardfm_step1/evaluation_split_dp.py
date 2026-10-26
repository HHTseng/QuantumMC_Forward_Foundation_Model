"""Physical-unit closure for split-asinh checkpoints.

Central summaries condition on |delta_p| <= 100 MeV, matching the poster.
Gaussian fits retain Dr. Joo's two-stage histogram-fit prescription. They
measure a core, not the full non-Gaussian response width.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

from .data_asinh_dp import DELTA_P_ASINH_SCALE_GEV, TARGET_COLUMNS, inverse_targets_from_model
from .evaluation import conditional_pid_response_rows, _wasserstein_equal_sample
from .model_split_dp import sample_split_standardized_residuals
from .training_split_dp import make_loader, run_epoch

SPECIES_LABELS = {-211: "pi-", 211: "pi+", 2212: "proton"}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x-mean)/sigma)**2)


def fit_gaussian_core(values):
    """Author's 1-MeV histogram fit in [-100,100] MeV; None records failure."""
    values = 1000 * np.asarray(values)
    selected = values[np.isfinite(values) & (values >= -100) & (values <= 100)]
    if len(selected) < 100:
        return None
    counts, edges = np.histogram(selected, bins=np.arange(-100., 101., 1.))
    centers = (edges[:-1] + edges[1:]) / 2
    mean0 = centers[np.argmax(counts)]
    mask = (centers >= mean0-40) & (centers <= mean0+40) & (counts > 0)
    try:
        params, _ = curve_fit(gaussian, centers[mask], counts[mask],
            p0=[counts.max(), mean0, 15.], bounds=([0,-100,1], [np.inf,100,100]), maxfev=20000)
        half_width = np.clip(2.5*abs(params[2]), 10, 80)
        mask = (centers >= params[1]-half_width) & (centers <= params[1]+half_width) & (counts > 0)
        params, covariance = curve_fit(gaussian, centers[mask], counts[mask],
            p0=params, bounds=([0,-100,.5], [np.inf,100,100]), maxfev=20000)
    except (RuntimeError, ValueError, TypeError):
        return None
    errors = np.sqrt(np.diag(covariance))
    return {"mu": float(params[1]), "sigma": float(abs(params[2])),
            "mu_error": float(errors[1]), "sigma_error": float(errors[2])}


def central_statistics(values):
    values = np.asarray(values)
    selected = values[(values >= -100) & (values <= 100)]
    if len(selected) < 100:
        raise ValueError("Not enough observations in the displayed core window")
    q16, median, q84 = np.quantile(selected, [.16,.5,.84])
    return {"median_MeV": float(median), "sigma68_MeV": float((q84-q16)/2),
            "window_fraction": float(len(selected)/len(values))}


@torch.no_grad()
def evaluate_split(model, split, feature_scaler, target_scaler, vocabulary, config, output_dir):
    """Draw once with the author's CPU sampling seed, then evaluate all cells."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    generator = torch.Generator().manual_seed(int(config["evaluation"].get("sampling_seed", 20261003)))
    samples, probabilities = [], []
    for start in range(0, len(split), 8192):
        sl = slice(start, start+8192)
        output = model(torch.from_numpy(split.continuous[sl]), torch.from_numpy(split.species_index[sl]))
        samples.append(sample_split_standardized_residuals(output, generator).numpy())
        probabilities.append(torch.softmax(output.pid_logits, -1).numpy())
    observed = inverse_targets_from_model(target_scaler.inverse(split.targets))
    sampled = inverse_targets_from_model(target_scaler.inverse(np.concatenate(samples)))
    probabilities = np.concatenate(probabilities)
    if not np.isfinite(sampled).all():
        raise FloatingPointError("Nonfinite sampled physical residuals; no clipping applied")
    # Dr. Joo's fitted delta-p distribution is unchanged by the azimuth wrap.
    sampled[:,2] = (sampled[:,2]+np.pi) % (2*np.pi) - np.pi
    features = feature_scaler.inverse(split.continuous)
    momentum = np.expm1(features[:,0])
    edges = np.asarray(config["evaluation"]["momentum_edges_gev"], dtype=float)
    core_rows, binned, histograms, response_rows, beta_rows = [], [], [], [], []
    for pid, name in SPECIES_LABELS.items():
        mask = split.raw_species == pid
        core = {"generated_pid":pid, "species":name, "n":int(mask.sum())}
        for prefix, array in (("coat",observed),("fm",sampled)):
            core.update({prefix+"_"+k:v for k,v in central_statistics(1000*array[mask,0]).items()})
            fit = fit_gaussian_core(array[mask,0])
            for field in ("mu","sigma"):
                core[f"{prefix}_gaussian_{field}_MeV"] = fit[field] if fit else float("nan")
        core["median_bias_MeV"] = core["fm_median_MeV"]-core["coat_median_MeV"]
        core["sigma68_ratio"] = core["fm_sigma68_MeV"]/core["coat_sigma68_MeV"]
        core_rows.append(core)
        hist_edges = np.arange(-100.,102.,2.)
        density = {}
        for prefix, array in (("coat",observed),("fm",sampled)):
            values = 1000*array[mask,0]
            density[prefix], _ = np.histogram(values, bins=hist_edges, density=True)
        for i, center in enumerate((hist_edges[:-1]+hist_edges[1:])/2):
            histograms.append({"species":name,"delta_p_MeV":center,
                               "coat_density":density["coat"][i],"fm_density":density["fm"][i]})
        for index, (low,high) in enumerate(zip(edges[:-1],edges[1:])):
            cell = mask & (momentum >= low) & ((momentum<=high) if index==len(edges)-2 else (momentum<high))
            if cell.sum()<100:
                continue
            cj, fm = fit_gaussian_core(observed[cell,0]), fit_gaussian_core(sampled[cell,0])
            row = {"generated_pid":pid,"species":name,"p_low_GeV":low,"p_high_GeV":high,
                   "n":int(cell.sum()),"fit_ok":cj is not None and fm is not None}
            for prefix, fit in (("coat",cj),("fm",fm)):
                for field in ("mu","sigma","mu_error","sigma_error"):
                    row[f"{prefix}_{field}_MeV"] = fit[field] if fit else float("nan")
            row["sigma_ratio"] = fm["sigma"]/cj["sigma"] if row["fit_ok"] else float("nan")
            row["mu_difference_MeV"] = fm["mu"]-cj["mu"] if row["fit_ok"] else float("nan")
            binned.append(row)
        for index, target in enumerate(TARGET_COLUMNS):
            truth, draw = observed[mask,index],sampled[mask,index]
            response_rows.append({"species":name,"target":target,"wasserstein_1d":_wasserstein_equal_sample(truth,draw),
                                  "observed_mean":float(truth.mean()),"sampled_mean":float(draw.mean()),
                                  "std_ratio":float(draw.std()/truth.std())})
        beta_gen = features[mask,4]
        beta_rows.append({"species":name,"wasserstein_1d":_wasserstein_equal_sample(
            beta_gen+observed[mask,3], beta_gen+sampled[mask,3])})
    pid_rows, pid_bins = conditional_pid_response_rows(split.raw_species, momentum,
        split.rec_pid_index, probabilities, [*vocabulary,"OTHER"], edges)
    loader = make_loader(split, int(config["training"]["batch_size"]), False, int(config["project"]["seed"]), 0)
    loss = run_epoch(model, loader, torch.device("cpu"), float(config["training"]["pid_loss_weight"]))
    high = [row for row in binned if row["p_low_GeV"]>=6 and row["generated_pid"] in (-211,211)]
    high_available = bool(high) and all(row["fit_ok"] for row in high)
    beta_rec = features[:,4]+observed[:,3]
    metrics = {
        "n":len(split), "loss_coordinates":"standardized asinh-dp and affine other residuals",
        "loss":loss.as_dict(),
        "pid_tv_particle_weighted":float(np.average([r["total_variation_distance"] for r in pid_bins],weights=[r["n"] for r in pid_bins])),
        "beta_w1_macro":float(np.mean([r["wasserstein_1d"] for r in beta_rows])),
        "high_pion_width_mae":float(np.mean([abs(r["sigma_ratio"]-1) for r in high])) if high_available else None,
        "high_pion_cells":[[r["species"],r["p_low_GeV"],r["p_high_GeV"]] for r in high],
        "failed_gaussian_fits":sum(not r["fit_ok"] for r in binned),
        "negative_reconstructed_p_fraction":float(np.mean(momentum+sampled[:,0]<0)),
        "sampled_abs_delta_p_gt_10_fraction":float(np.mean(np.abs(sampled[:,0])>10)),
        "teacher_rows_outside_later_beta_validity_domain":int(np.sum((beta_rec<=0)|(beta_rec>1.2))),
        "target_correlation_frobenius":{
            name:float(np.linalg.norm(np.corrcoef(observed[split.raw_species==pid].T)-np.corrcoef(sampled[split.raw_species==pid].T)))
            for pid,name in SPECIES_LABELS.items()
        },
    }
    for filename, rows in (("core_summary.csv",core_rows),("delta_p_gaussian_vs_momentum.csv",binned),
        ("core_histograms.csv",histograms),("pid_response_fixed_bins.csv",pid_rows),
        ("pid_bin_closure_summary.csv",pid_bins),("response_closure.csv",response_rows),("beta_closure.csv",beta_rows)):
        if rows:
            write_csv(output_dir/filename,rows)
    (output_dir/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    return metrics
