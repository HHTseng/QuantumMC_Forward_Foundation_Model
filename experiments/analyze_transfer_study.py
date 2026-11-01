#!/usr/bin/env python3
"""Publication-style summaries of frozen response and reconstruction transfer.

Response-skim and mutual-match phase-space references have different label/data
contracts. Their curves describe process/dataset transfer, not a controlled
retraining improvement. MC errors are binomial/fit errors; FM bands are sample
SD across independent network seeds, not detector-systematic uncertainties.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPECIES = {-211: r"$\pi^-$", 211: r"$\pi^+$", 2212: r"$p$"}
LABELS = {-211: "pi-", 211: "pi+", 2212: "proton"}
COLORS = {"august": "#737373", "phase_test": "#737373", "rho": "#2166ac", "rho_test": "#2166ac"}
DATASET_LABELS = {"august": "Response phase space", "phase_test": "Mutual-match phase space",
                  "rho": r"Exclusive $\rho^0$", "rho_test": r"Exclusive $\rho^0$"}


def save_figure(fig, output, name):
    fig.savefig(output/f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output/f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_runs(frame, group_columns, metric_columns):
    records = []
    if frame.empty:
        return pd.DataFrame()
    for key, group in frame.groupby(group_columns, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns, key))
        row["seeds"] = int(group.seed.nunique())
        row["n_min"], row["n_max"] = int(group.n.min()), int(group.n.max())
        for column in metric_columns:
            values = group[column].dropna().to_numpy(float)
            row[f"{column}_mean"] = float(values.mean()) if len(values) else None
            row[f"{column}_sd"] = float(values.std(ddof=1)) if len(values)>1 else None
        records.append(row)
    return pd.DataFrame(records)


def read_results(runs):
    response, gates, response_records, gate_records = [], [], [], []
    for path in sorted(runs.glob("*/response/seed_*/*/metrics.json")):
        relative = path.relative_to(runs)
        host, seed, dataset = relative.parts[0], int(relative.parts[2].removeprefix("seed_")), relative.parts[3]
        metrics = json.loads(path.read_text())
        resolution = pd.read_csv(path.parent/"momentum_resolution.csv")
        response_records.append(dict(host=host, seed=seed, dataset=dataset, path=path.parent, metrics=metrics))
        for label, values in metrics["species"].items():
            pid = next(pid for pid, name in LABELS.items() if name == label)
            fits = resolution[(resolution.generated_pid==pid) & resolution.eligible &
                              (resolution.teacher=="raw")].gaussian_sigma_ratio.dropna()
            response.append(dict(host=host, seed=seed, dataset=dataset, species=label,
                n=values["n"], correct_id_mae_pp=values["correct_id_mae_pp"],
                pid_tv_pp=100*values["pid_tv_particle_weighted"] if values["pid_tv_particle_weighted"] is not None else None,
                gaussian_width_error_percent=100*float(np.mean(np.abs(fits-1))) if len(fits) else None,
                valid_width_bins=len(fits),
                fm_abs_dp_gt_10_percent=100*values["fm_abs_delta_p_gt_10_fraction"],
                fm_negative_p_percent=100*values["fm_negative_reconstructed_p_fraction"],
                outside_coarse_aug_box_percent=100*values["outside_coarse_aug_generated_box_fraction"]))
    for path in sorted(runs.glob("*/efficiency/seed_*/pid_*/*_test/metrics.json")):
        relative = path.relative_to(runs)
        host = relative.parts[0]
        seed = int(relative.parts[2].removeprefix("seed_"))
        pid = int(relative.parts[3].removeprefix("pid_"))
        dataset = relative.parts[4]
        metrics = json.loads(path.read_text())
        gate_records.append(dict(host=host, seed=seed, dataset=dataset, pid=pid, path=path.parent, metrics=metrics))
        for target in ("fd", "any"):
            value = metrics[target]
            closure = metrics["kinematic_closure"][target]["gen_p_theta"]
            gates.append(dict(host=host, seed=seed, dataset=dataset, species=LABELS[pid],
                target=target, n=value["n"], mc_rate_percent=100*value["observed_rate"],
                fm_rate_percent=100*value["mean_predicted_probability"],
                bias_pp=100*value["mean_bias"], ece_pp=100*value["expected_calibration_error"],
                p_theta_weighted_mae_pp=100*closure["weighted_mae"] if closure["weighted_mae"] is not None else None,
                brier=value["brier_score"]))
    return pd.DataFrame(response), pd.DataFrame(gates), response_records, gate_records


def common_bin_response(records):
    """Compare datasets over identical eligible bins, with explicit coverage.

    The common momentum intervals remove a trivial MAE-support difference;
    their within-bin populations/trigger definitions are still different.
    """
    rows = []
    lookup = {(r["host"],r["seed"],r["dataset"]):r for r in records}
    for record in records:
        if record["dataset"] not in ("rho", "rho_identified"):
            continue
        reference = lookup.get((record["host"],record["seed"],"august"))
        if not reference:
            continue
        pid_tables = {"august":pd.read_csv(reference["path"]/"pid_closure.csv"),
                      record["dataset"]:pd.read_csv(record["path"]/"pid_closure.csv")}
        width_tables = {"august":pd.read_csv(reference["path"]/"momentum_resolution.csv"),
                        record["dataset"]:pd.read_csv(record["path"]/"momentum_resolution.csv")}
        for pid, species in LABELS.items():
            pid_cells, width_cells = {}, {}
            for dataset in pid_tables:
                selected = pid_tables[dataset]
                selected = selected[(selected.generated_pid==pid) & selected.eligible].copy()
                pid_cells[dataset] = selected.set_index(["p_low_GeV","p_high_GeV"])
                selected = width_tables[dataset]
                selected = selected[(selected.generated_pid==pid) & selected.eligible &
                                    (selected.teacher=="raw") & selected.gaussian_sigma_ratio.notna()].copy()
                width_cells[dataset] = selected.set_index(["p_low_GeV","p_high_GeV"])
            common_pid = sorted(set(pid_cells["august"].index) & set(pid_cells[record["dataset"]].index))
            common_width = sorted(set(width_cells["august"].index) & set(width_cells[record["dataset"]].index))
            for dataset in pid_tables:
                chosen = pid_cells[dataset].loc[common_pid] if common_pid else pid_cells[dataset].iloc[:0]
                widths = width_cells[dataset].loc[common_width] if common_width else width_cells[dataset].iloc[:0]
                rows.append(dict(host=record["host"],seed=record["seed"],comparison=record["dataset"],
                    dataset=dataset,species=species,n=int(chosen.n.sum()),
                    common_pid_bins=";".join(f"{lo:g}-{hi:g}" for lo,hi in common_pid),
                    common_pid_bin_count=len(common_pid),
                    correct_id_mae_pp=float(chosen.absolute_error_pp.mean()) if len(chosen) else None,
                    pid_tv_pp=100*float(np.average(chosen.total_variation,weights=chosen.n)) if len(chosen) else None,
                    common_width_bins=";".join(f"{lo:g}-{hi:g}" for lo,hi in common_width),
                    common_width_bin_count=len(common_width),
                    gaussian_width_error_percent=100*float(np.mean(np.abs(widths.gaussian_sigma_ratio-1))) if len(widths) else None))
    return pd.DataFrame(rows)


def efficiency_support(records):
    """Retain calibration inside/outside the training coordinate box.

    This diagnostic identifies extrapolation; it does not certify joint
    kinematic coverage or remove the need to model event correlations.
    """
    rows = []
    for record in records:
        for target in ("fd", "any"):
            for subset, metrics in record["metrics"].get("support", {}).get(target, {}).items():
                rows.append(dict(host=record["host"], seed=record["seed"],
                    dataset=record["dataset"], species=LABELS[record["pid"]],
                    target=target, subset=subset, n=metrics["n"],
                    mc_rate_percent=100*metrics["observed_rate"],
                    fm_rate_percent=100*metrics["mean_predicted_probability"],
                    bias_pp=100*metrics["mean_bias"],
                    ece_pp=100*metrics["expected_calibration_error"]))
    return pd.DataFrame(rows)


def check_seed_populations(response_records, gate_records):
    """Seed bands require identical teacher particles, not independent data draws."""
    rows = []
    groups = {}
    for record in response_records:
        groups.setdefault((record["host"],"response",record["dataset"],None),[]).append(record)
    for record in gate_records:
        groups.setdefault((record["host"],"efficiency",record["dataset"],record["pid"]),[]).append(record)
    for (host,task,dataset,pid),records in groups.items():
        records = sorted(records,key=lambda r:r["seed"])
        reference = records[0]
        filename = "response_events.npz" if task=="response" else "predictions.npz"
        if not (reference["path"]/filename).exists():
            continue
        with np.load(reference["path"]/filename,allow_pickle=False) as first:
            for record in records:
                path = record["path"]/filename
                if not path.exists():
                    continue
                with np.load(path,allow_pickle=False) as second:
                    keys = ("file_id","event_id","mcindex","species") if task=="response" else ("event_keys",)
                    truth = ("raw_truth","corrected_truth","gen_p","gen_theta") if task=="response" else ("outcome_target","gen_p","gen_theta")
                    same_keys = all(np.array_equal(first[k],second[k]) for k in keys)
                    same_truth = same_keys and all(np.array_equal(first[k],second[k],equal_nan=True) for k in truth)
                    rows.append(dict(host=host,task=task,dataset=dataset,species=LABELS.get(pid,"all"),
                        reference_seed=reference["seed"],seed=record["seed"],
                        identical_particle_keys=same_keys,identical_teacher=same_truth))
                    if not same_truth:
                        raise ValueError(f"Cannot aggregate seeds with different teacher particles: {host}/{task}/{dataset}/{pid}")
    return pd.DataFrame(rows)


def pid_figure(records, host, output):
    frames = []
    for record in records:
        if record["host"] == host:
            frame = pd.read_csv(record["path"]/"pid_closure.csv")
            frame["dataset"], frame["seed"] = record["dataset"], record["seed"]
            frames.append(frame)
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), sharey=True)
    for axis, (pid, label) in zip(axes, SPECIES.items()):
        for dataset in ("august", "rho"):
            selected = frame[(frame.generated_pid==pid) & (frame.dataset==dataset) & frame.eligible]
            if selected.empty:
                continue
            grouped = selected.groupby(["p_low_GeV", "p_high_GeV"], sort=True)
            mean = grouped[["observed_fraction", "fm_fraction", "observed_binomial_se"]].mean()
            sd = grouped.fm_fraction.std(ddof=1).fillna(0)
            x = np.array([(lo+hi)/2 for lo, hi in mean.index])
            color = COLORS[dataset]
            axis.errorbar(x, mean.observed_fraction, yerr=mean.observed_binomial_se,
                          fmt="o", ms=3.5, color=color, capsize=2,
                          label=DATASET_LABELS[dataset]+" MC")
            axis.plot(x, mean.fm_fraction, "s-", ms=3, color=color,
                      label=DATASET_LABELS[dataset]+" FM")
            axis.fill_between(x, mean.fm_fraction-sd, mean.fm_fraction+sd, color=color, alpha=.17)
        axis.set(title=label, xlabel=r"$p_{\rm gen}$ [GeV/$c$]", xlim=(0,9), ylim=(0,1.03))
        axis.grid(alpha=.2)
    axes[0].set_ylabel(r"Correct-ID fraction / mean $P_{\rm FM}$")
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("Frozen response transfer; FM bands: seed SD; MC bars: binomial SE", fontsize=11)
    fig.tight_layout()
    save_figure(fig, output, "pid_correct_id_transfer")


def mae_bar_figure(summary, common_runs, host, output):
    if summary.empty:
        return
    selected = summary[(summary.host==host) & (summary.comparison=="rho")]
    if selected.empty:
        return
    fig, axis = plt.subplots(figsize=(7.2,4.1))
    x = np.arange(3)
    for index,dataset in enumerate(("august","rho")):
        means, errors = [], []
        for species in LABELS.values():
            row = selected[(selected.dataset==dataset) & (selected.species==species)]
            means.append(float(row.correct_id_mae_pp_mean.iloc[0]) if len(row) else np.nan)
            sd = row.correct_id_mae_pp_sd.iloc[0] if len(row) else np.nan
            errors.append(float(sd) if pd.notna(sd) else 0.)
        axis.bar(x+(index-.5)*.34,means,.34,yerr=errors,capsize=3,
                 color=COLORS[dataset],label=DATASET_LABELS[dataset])
    labels = []
    for pid,species in LABELS.items():
        counts = common_runs[(common_runs.host==host) & (common_runs.comparison=="rho") &
                             (common_runs.species==species)].common_pid_bin_count.unique()
        suffix = str(int(counts[0])) if len(counts)==1 else "/".join(str(int(v)) for v in counts)
        labels.append(SPECIES[pid]+f"\n{suffix} common bins")
    axis.set(xticks=x,xticklabels=labels,ylabel="Correct-ID closure MAE [pp]",ylim=(0,None))
    axis.grid(axis="y",alpha=.2)
    axis.set_axisbelow(True)
    axis.legend(fontsize=8)
    axis.set_title("Common momentum-bin comparison; network-seed mean +/- SD",fontsize=11)
    fig.tight_layout()
    save_figure(fig,output,"pid_common_bin_mae")


def resolution_figure(records, host, output):
    frames = []
    for record in records:
        if record["host"] == host:
            frame = pd.read_csv(record["path"]/"momentum_resolution.csv")
            frame["dataset"], frame["seed"] = record["dataset"], record["seed"]
            frames.append(frame)
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.1), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1]})
    variants = [("august", "raw", "#737373", "Response phase-space raw"),
                ("rho", "raw", "#2166ac", r"$\rho^0$ raw"),
                ("rho", "corrected", "#1b7837", r"$\rho^0$ corrected")]
    for column, (pid, label) in enumerate(SPECIES.items()):
        for dataset, teacher, color, title in variants:
            selected = frame[(frame.generated_pid==pid) & (frame.dataset==dataset) &
                             (frame.teacher==teacher) & frame.eligible]
            if selected.empty:
                continue
            grouped = selected.groupby(["p_low_GeV", "p_high_GeV"], sort=True)
            mean = grouped[["observed_gaussian_sigma_MeV", "observed_gaussian_sigma_error_MeV",
                            "fm_gaussian_sigma_MeV", "gaussian_sigma_ratio"]].mean()
            x = np.array([(lo+hi)/2 for lo, hi in mean.index])
            axes[0,column].errorbar(x, mean.observed_gaussian_sigma_MeV,
                yerr=mean.observed_gaussian_sigma_error_MeV, color=color, fmt="o", ms=3,
                capsize=2, label=title+" MC")
            # The FM is one raw-response prediction. Repeat it only once per
            # generated population, rather than implying a corrected model.
            if teacher == "raw":
                sd = grouped.fm_gaussian_sigma_MeV.std(ddof=1).fillna(0)
                axes[0,column].plot(x, mean.fm_gaussian_sigma_MeV, "s-", color=color, ms=3, label=title+" FM")
                axes[0,column].fill_between(x, mean.fm_gaussian_sigma_MeV-sd,
                    mean.fm_gaussian_sigma_MeV+sd, color=color, alpha=.15)
            ratio_sd = grouped.gaussian_sigma_ratio.std(ddof=1).fillna(0)
            axes[1,column].plot(x, mean.gaussian_sigma_ratio, "o-", color=color, ms=2.5, label=title)
            axes[1,column].fill_between(x, mean.gaussian_sigma_ratio-ratio_sd,
                mean.gaussian_sigma_ratio+ratio_sd, color=color, alpha=.13)
        axes[0,column].set_title(label)
        axes[1,column].axhline(1, ls="--", color=".4", lw=1)
        axes[1,column].set(xlabel=r"$p_{\rm gen}$ [GeV/$c$]", xlim=(0,9))
        for axis in axes[:,column]:
            axis.grid(alpha=.2)
    axes[0,0].set_ylabel(r"Gaussian core $\sigma_{\Delta p}$ [MeV/$c$]")
    axes[1,0].set_ylabel(r"$\sigma_{\rm FM}/\sigma_{\rm MC}$")
    axes[0,1].legend(fontsize=7, ncol=2)
    fig.suptitle("Core resolution; corrected MC is a separate reference, not the frozen FM target", fontsize=11)
    fig.tight_layout()
    save_figure(fig, output, "momentum_resolution_transfer")


def core_histogram_figure(records, host, output):
    selected = sorted([r for r in records if r["host"]==host and r["dataset"]=="rho"], key=lambda r:r["seed"])
    if not selected or not (selected[0]["path"]/"response_events.npz").exists():
        return
    record = selected[0]
    with np.load(record["path"]/"response_events.npz", allow_pickle=False) as arrays:
        fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.9))
        edges = np.arange(-100., 102., 2.)
        for axis, (pid, label) in zip(axes, SPECIES.items()):
            mask = arrays["species"]==pid
            notes = []
            for key, color, title in (("raw_truth", "#2166ac", "Raw MC"),
                                       ("corrected_truth", "#1b7837", "Corrected MC"),
                                       ("sampled_response", "#d6604d", "Frozen FM")):
                values = 1000*arrays[key][mask,0]
                counts, _ = np.histogram(values[np.isfinite(values)], bins=edges)
                # Normalize to ALL selected particles, preserving tail mass
                # outside the displayed +/-100 MeV interval.
                density = counts/(len(values)*np.diff(edges))
                axis.stairs(density, edges, color=color, lw=1.4, label=title)
                tail = 100*np.mean(~np.isfinite(values) | (np.abs(values)>100))
                notes.append(f"{title}: {tail:.1f}% outside")
            axis.text(.03,.97,"\n".join(notes), transform=axis.transAxes, va="top", fontsize=7)
            axis.set(title=label, xlabel=r"$\Delta p=p_{\rm rec}-p_{\rm gen}$ [MeV/$c$]", xlim=(-100,100))
            axis.grid(alpha=.15)
        axes[0].set_ylabel(r"Probability density [(MeV/$c$)$^{-1}$]")
        axes[1].legend(fontsize=8, loc="upper right")
        fig.suptitle(f"Exclusive rho central momentum response; representative network seed {record['seed']}", fontsize=11)
        fig.tight_layout()
        save_figure(fig, output, "central_momentum_response")


def efficiency_figures(records, host, output):
    selected = [r for r in records if r["host"]==host]
    if not selected:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.4), sharex=True, sharey=True)
    for row, target in enumerate(("fd", "any")):
        for column, (pid, label) in enumerate(SPECIES.items()):
            axis = axes[row,column]
            for dataset in ("phase_test", "rho_test"):
                tables = []
                for record in selected:
                    if record["pid"]==pid and record["dataset"]==dataset:
                        table = pd.read_csv(record["path"]/f"efficiency_{target}_vs_gen_p.csv")
                        table["seed"] = record["seed"]
                        tables.append(table)
                if not tables:
                    continue
                table = pd.concat(tables, ignore_index=True)
                table = table[table.eligible]
                grouped = table.groupby(["low", "high"], sort=True)
                mean = grouped[["observed_efficiency", "fm_mean_probability", "wilson_low", "wilson_high"]].mean()
                sd = grouped.fm_mean_probability.std(ddof=1).fillna(0)
                x = np.array([(lo+hi)/2 for lo,hi in mean.index])
                y = mean.observed_efficiency.to_numpy()
                error = np.maximum(0, np.array([y-mean.wilson_low, mean.wilson_high-y]))
                color = COLORS[dataset]
                axis.errorbar(x, y, yerr=error, fmt="o", color=color, ms=3, capsize=2,
                              label=DATASET_LABELS[dataset]+" MC")
                axis.plot(x, mean.fm_mean_probability, "s-", ms=3, color=color,
                          label=DATASET_LABELS[dataset]+" FM")
                axis.fill_between(x, mean.fm_mean_probability-sd, mean.fm_mean_probability+sd,
                                  color=color, alpha=.16)
            axis.set(title=label if row==0 else None, xlim=(0,10), ylim=(0,1.03))
            axis.grid(alpha=.2)
            if row==1:
                axis.set_xlabel(r"$p_{\rm gen}$ [GeV/$c$]")
        axes[row,0].set_ylabel(("FD" if target=="fd" else "Any-region")+r" efficiency $\mid T_e=1$")
    axes[0,1].legend(fontsize=7)
    fig.suptitle("Phase-space-trained reciprocal reconstruction gates; FM bands: seed SD", fontsize=11)
    fig.tight_layout()
    save_figure(fig, output, "hadron_efficiency_transfer")

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.2), sharex=True, sharey=True)
    for row, target in enumerate(("fd", "any")):
        for column, (pid, label) in enumerate(SPECIES.items()):
            axis = axes[row,column]
            axis.plot([0,1], [0,1], "--", color=".4", lw=1)
            for dataset in ("phase_test", "rho_test"):
                first = True
                for record in selected:
                    if record["pid"]!=pid or record["dataset"]!=dataset:
                        continue
                    table = pd.read_csv(record["path"]/f"calibration_{target}.csv")
                    table = table[table.n>=200]
                    axis.plot(table.mean_predicted_probability, table.observed_efficiency,
                              "o-", color=COLORS[dataset], ms=2, lw=.8, alpha=.65,
                              label=DATASET_LABELS[dataset] if first else None)
                    first = False
            axis.set(title=label if row==0 else None, xlim=(0,1), ylim=(0,1))
            axis.grid(alpha=.2)
            if row==1:
                axis.set_xlabel("Mean predicted probability")
        axes[row,0].set_ylabel(("FD" if target=="fd" else "Any-region")+" observed fraction")
    axes[0,1].legend(fontsize=7)
    fig.suptitle("Reconstruction calibration; one curve/network seed; bins with N >= 200", fontsize=11)
    fig.tight_layout()
    save_figure(fig, output, "hadron_efficiency_calibration")

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6), sharex=True, sharey=True)
    matrices = []
    grid = []
    for row, target in enumerate(("fd", "any")):
        for column, (pid, label) in enumerate(SPECIES.items()):
            tables = [pd.read_csv(r["path"]/f"efficiency_{target}_p_theta.csv")
                      for r in selected if r["pid"]==pid and r["dataset"]=="rho_test"]
            if not tables:
                continue
            table = pd.concat(tables, ignore_index=True)
            grouped = table.groupby(["p_bin_index", "bin_index", "p_low", "p_high", "low", "high"], sort=True)
            cells = grouped[["n", "observed_efficiency", "fm_mean_probability"]].mean().reset_index()
            p_edges = np.unique(np.r_[cells.p_low, cells.p_high])
            theta_edges = np.unique(np.r_[cells.low, cells.high])
            matrix = np.full((len(theta_edges)-1, len(p_edges)-1), np.nan)
            for cell in cells.itertuples():
                if cell.n>=200 and np.isfinite(cell.fm_mean_probability):
                    matrix[int(cell.bin_index),int(cell.p_bin_index)] = 100*(cell.fm_mean_probability-cell.observed_efficiency)
            matrices.append(matrix)
            grid.append((row, column, label, p_edges, theta_edges, matrix))
    if matrices:
        maximum = max((float(np.nanmax(np.abs(m))) for m in matrices if np.isfinite(m).any()), default=0.)
        limit = max(1., np.ceil(maximum))
        for row, column, label, p_edges, theta_edges, matrix in grid:
            image = axes[row,column].pcolormesh(p_edges,theta_edges,matrix,
                cmap="RdBu_r", vmin=-limit, vmax=limit, shading="flat")
            axes[row,column].set_title(label if row==0 else "")
        for row, target in enumerate(("FD", "Any")):
            axes[row,0].set_ylabel(target+r": $\theta_{\rm gen}$ [deg]")
        for axis in axes.ravel():
            axis.set_ylim(0, 80)
            axis.axhline(5, color=".35", ls=":", lw=.7)
        for axis in axes[1]:
            axis.set_xlabel(r"$p_{\rm gen}$ [GeV/$c$]")
        fig.suptitle(r"Exclusive $\rho^0$: FM - MC [pp]; blank: N < 200; dotted: phase-space $\theta_{\rm gen}=5^\circ$ boundary", fontsize=10)
        fig.subplots_adjust(right=.88, hspace=.12, wspace=.12)
        fig.colorbar(image, ax=axes.ravel().tolist(), label="FM - MC [pp]", fraction=.025, pad=.025)
        save_figure(fig, output, "rho_efficiency_p_theta_closure")
    else:
        plt.close(fig)


def cross_platform(records, gate_records, runs):
    response_rows, gate_rows = [], []
    hosts = sorted({r["host"] for r in [*records, *gate_records]})
    if len(hosts)<2:
        return pd.DataFrame(), pd.DataFrame()
    reference = "tara" if "tara" in hosts else hosts[0]
    lookup = {(r["host"],r["seed"],r["dataset"]):r for r in records}
    for record in records:
        host = record["host"]
        if host==reference:
            continue
        paired = lookup.get((reference, record["seed"], record["dataset"]))
        if not paired:
            continue
        first_path, second_path = paired["path"]/"response_events.npz", record["path"]/"response_events.npz"
        if not first_path.exists() or not second_path.exists():
            continue
        with np.load(first_path, allow_pickle=False) as first, np.load(second_path, allow_pickle=False) as second:
            key_equal = all(np.array_equal(first[k],second[k]) for k in ("file_id","event_id","mcindex","species"))
            teacher_equal = key_equal and all(np.array_equal(first[k],second[k],equal_nan=True)
                                              for k in ("raw_truth","corrected_truth","gen_p","gen_theta"))
            row = dict(reference_host=reference, comparison_host=host, seed=record["seed"],
                dataset=record["dataset"], n=len(first["species"]),
                identical_checkpoint=paired["metrics"]["checkpoint_sha256"]==record["metrics"]["checkpoint_sha256"],
                identical_particle_keys=key_equal, identical_teacher=teacher_equal)
            if key_equal:
                diff = np.abs(first["pid_probabilities"]-second["pid_probabilities"])
                row["max_abs_pid_probability_difference"] = float(diff.max())
                row["mean_abs_pid_probability_difference"] = float(diff.mean())
                draw_diff = np.abs(first["sampled_response"][:,0]-second["sampled_response"][:,0])
                finite = draw_diff[np.isfinite(draw_diff)]
                row["max_abs_sampled_dp_difference_GeV"] = float(finite.max()) if len(finite) else None
                row["median_abs_sampled_dp_difference_GeV"] = float(np.median(finite)) if len(finite) else None
            response_rows.append(row)
    gate_lookup = {(r["host"],r["seed"],r["dataset"],r["pid"]):r for r in gate_records}
    for record in gate_records:
        host = record["host"]
        if host==reference:
            continue
        paired = gate_lookup.get((reference, record["seed"],record["dataset"],record["pid"]))
        if not paired:
            continue
        first_path, second_path = paired["path"]/"predictions.npz", record["path"]/"predictions.npz"
        key_equal = teacher_equal = None
        max_difference = None
        if first_path.exists() and second_path.exists():
            with np.load(first_path, allow_pickle=False) as first, np.load(second_path, allow_pickle=False) as second:
                key_equal = bool(np.array_equal(first["event_keys"],second["event_keys"]))
                teacher_equal = key_equal and bool(np.array_equal(first["outcome_target"],second["outcome_target"]))
                if key_equal:
                    max_difference = float(np.max(np.abs(first["outcome_probability"]-second["outcome_probability"])))
        for target in ("fd", "any"):
            a, b = paired["metrics"][target], record["metrics"][target]
            gate_rows.append(dict(reference_host=reference,comparison_host=host,
                seed=record["seed"],dataset=record["dataset"],species=LABELS[record["pid"]],target=target,
                n=a["n"],identical_particle_keys=key_equal,identical_teacher=teacher_equal,
                ece_difference_pp=100*(b["expected_calibration_error"]-a["expected_calibration_error"]),
                mean_prediction_difference_pp=100*(b["mean_predicted_probability"]-a["mean_predicted_probability"]),
                max_abs_probability_difference=max_difference,
                interpretation="paired seeds trained independently on each GPU"))
    return pd.DataFrame(response_rows), pd.DataFrame(gate_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot-host", default="tara")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 12,
                         "legend.frameon": False, "savefig.facecolor": "white"})
    response, gates, response_records, gate_records = read_results(args.runs)
    if response.empty and gates.empty:
        parser.error("No completed response or efficiency metrics found")
    response_summary = aggregate_runs(response, ["host","dataset","species"],
        ["correct_id_mae_pp","pid_tv_pp","gaussian_width_error_percent",
         "fm_abs_dp_gt_10_percent","fm_negative_p_percent","outside_coarse_aug_box_percent"])
    gate_summary = aggregate_runs(gates, ["host","dataset","species","target"],
        ["mc_rate_percent","fm_rate_percent","bias_pp","ece_pp","p_theta_weighted_mae_pp","brier"])
    support_runs = efficiency_support(gate_records)
    support_summary = aggregate_runs(support_runs, ["host","dataset","species","target","subset"],
        ["mc_rate_percent","fm_rate_percent","bias_pp","ece_pp"])
    population_checks = check_seed_populations(response_records,gate_records)
    common_response = common_bin_response(response_records)
    common_summary = aggregate_runs(common_response,["host","comparison","dataset","species"],
        ["correct_id_mae_pp","pid_tv_pp","gaussian_width_error_percent"])
    response_parity, gate_parity = cross_platform(response_records, gate_records, args.runs)
    for name, frame in (("response_runs",response),("response_summary",response_summary),
                        ("efficiency_runs",gates),("efficiency_summary",gate_summary),
                        ("efficiency_support_runs",support_runs),("efficiency_support_summary",support_summary),
                        ("response_common_bins_runs",common_response),("response_common_bins_summary",common_summary),
                        ("seed_population_checks",population_checks),
                        ("response_cross_platform",response_parity),("efficiency_cross_platform",gate_parity)):
        if not frame.empty:
            frame.to_csv(args.output/f"{name}.csv", index=False)
    hosts = sorted({r["host"] for r in [*response_records,*gate_records]})
    host = args.plot_host if args.plot_host in hosts else hosts[0]
    pid_figure(response_records,host,args.output)
    mae_bar_figure(common_summary,common_response,host,args.output)
    resolution_figure(response_records,host,args.output)
    core_histogram_figure(response_records,host,args.output)
    efficiency_figures(gate_records,host,args.output)
    summary = {"plot_host":host,"hosts":hosts,"response_runs":len(response_records),
        "efficiency_runs":len(gate_records),"fm_band":"sample SD across network seeds",
        "mc_errors":"binomial/Wilson or Gaussian-fit sampling errors; unweighted",
        "response_reference":"Response phase-space skim: original checkpoint-selected held-out particles",
        "efficiency_reference":"Mutual-match phase-space export: generated-hadron denominator given T_e",
        "interpretation":"Dataset/process transfer; differing reference contracts prohibit attributing changes solely to physics generation",
        "response_summary":json.loads(response_summary.to_json(orient="records")),
        "response_common_bins_summary":json.loads(common_summary.to_json(orient="records")),
        "efficiency_summary":json.loads(gate_summary.to_json(orient="records")),
        "efficiency_support_summary":json.loads(support_summary.to_json(orient="records")),
        "response_cross_platform":json.loads(response_parity.to_json(orient="records")),
        "efficiency_cross_platform":json.loads(gate_parity.to_json(orient="records"))}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"hosts":hosts,"response_runs":len(response_records),"efficiency_runs":len(gate_records),
                      "output":str(args.output)},indent=2))


if __name__=="__main__":
    main()
