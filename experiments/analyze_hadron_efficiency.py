#!/usr/bin/env python3
"""Combine 3 fixed-split training seeds and the original electron calibration.

The electron panel estimates P(T=1|x_e) using all generated events. Hadron
panels estimate P(C_h=FD|x_h,s,T=1) using all generated hadrons in triggered
events. These are different denominators, not four hardware trigger models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forwardfm_hadron.evaluation import wilson_interval


SPECIES = (-211, 211, 2212)
LABELS = {-211: r"$\pi^-$", 211: r"$\pi^+$", 2212: "proton"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def _seed_independent_recipe(config: dict[str, Any]) -> str:
    training = {key: value for key, value in config["training"].items() if key not in ("device", "seed")}
    return json.dumps({"model": config["model"], "training": training,
                       "evaluation": config["evaluation"]}, sort_keys=True)


def _collect(root: Path, seeds: list[int]) -> tuple[pd.DataFrame, dict[tuple[int, int], Path]]:
    rows = []
    directories = {}
    reference_audit = {}
    reference_hash = {}
    reference_recipe = {}
    source_hashes = None
    for seed in seeds:
        for pid in SPECIES:
            directory = root / f"seed_{seed}" / f"pid_{pid}"
            metrics = _load_json(directory / "metrics.json")
            audit = _load_json(directory / "data_audit.json")
            with (directory / "resolved_config.yaml").open() as stream:
                config = yaml.safe_load(stream)
            environment = _load_json(directory / "environment.json")
            if int(config["project"]["seed"]) != seed or int(audit["species"]) != pid:
                raise ValueError(f"Seed/species metadata disagrees with run directory: {directory}")
            if source_hashes is not None and environment["source_hashes"] != source_hashes:
                raise ValueError("Study runs used different training/evaluation source files")
            source_hashes = environment["source_hashes"]
            with np.load(directory / "predictions.npz") as prediction:
                digest = hashlib.sha256()
                for key in ("event_keys", "outcome_target", "gen_p", "gen_theta", "gen_phi", "gen_vz"):
                    digest.update(np.ascontiguousarray(prediction[key]).tobytes())
                truth_hash = digest.hexdigest()
            comparison = {
                "dataset_metadata_sha256": audit["dataset_metadata_sha256"],
                "split_seed": audit["split_seed"],
                "active_feature_names": audit["active_feature_names"],
                "particle_order_sha256": {name: audit["split_counts"][name]["particle_order_sha256"]
                                           for name in ("train", "validation", "test")},
            }
            recipe = _seed_independent_recipe(config)
            if pid in reference_audit:
                if comparison != reference_audit[pid] or truth_hash != reference_hash[pid]:
                    raise ValueError(f"Different data, test-row order, or features for species {pid}")
                if recipe != reference_recipe[pid]:
                    raise ValueError(f"Different training/evaluation recipe for species {pid}")
            reference_audit[pid] = comparison
            reference_hash[pid] = truth_hash
            reference_recipe[pid] = recipe
            directories[(seed, pid)] = directory
            for target in ("fd", "any"):
                binary = metrics[target]
                row = {"seed": seed, "gen_pid": pid, "target": target, **binary,
                       "test_truth_sha256": truth_hash,
                       "checkpoint_sha256": hashlib.sha256((directory / "model.pt").read_bytes()).hexdigest()}
                for variable, summary in metrics["kinematic_closure"][target].items():
                    row[f"{variable}_weighted_mae"] = summary["weighted_mae"]
                    row[f"{variable}_unweighted_mae"] = summary["unweighted_mae"]
                    row[f"{variable}_maximum_absolute_difference"] = summary["maximum_absolute_difference"]
                rows.append(row)
    return pd.DataFrame(rows), directories


def _plot_calibration(root: Path, directories: dict[tuple[int, int], Path],
                      representative: int, electron: Path, target: str) -> dict[str, Any]:
    electron_metrics = _load_json(electron / "metrics.json")["trigger"]
    electron_curve = pd.read_csv(electron / "calibration_curve.csv")
    shutil.copy2(electron / "calibration_curve.csv", root / "electron_calibration_reference.csv")
    shutil.copy2(electron / "metrics.json", root / "electron_metrics_reference.json")
    panels = [("Electron: all generated events", electron_curve, electron_metrics)]
    for pid in SPECIES:
        directory = directories[(representative, pid)]
        panels.append((f"{LABELS[pid]}: given electron trigger",
                       pd.read_csv(directory / f"calibration_{target}.csv"),
                       _load_json(directory / "metrics.json")[target]))
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 9.4), sharex=True, sharey=True)
    for index, (axis, (title, frame, metrics)) in enumerate(zip(axes.flat, panels)):
        frame = frame.loc[frame.n > 0].copy()
        x = frame.mean_predicted_probability.to_numpy()
        observed_key = "observed_trigger_rate" if index == 0 else "observed_efficiency"
        observed = frame[observed_key].to_numpy()
        intervals = np.array([wilson_interval(float(rate * n), int(n))
                              for rate, n in zip(observed, frame.n)])
        counts = axis.twinx()
        center = (frame.probability_low + frame.probability_high) / 2
        counts.bar(center, frame.n, width=.045, color=".6", alpha=.18, label="MC bin count")
        counts.tick_params(axis="y", labelsize=7, colors=".5")
        counts.set_ylabel("Bin count", fontsize=8, color=".5")
        counts.set_ylim(0, max(frame.n) * 1.25)
        axis.set_zorder(counts.get_zorder() + 1)
        axis.patch.set_visible(False)
        axis.plot([0, 1], [0, 1], "--", color=".35", label="Ideal calibration")
        axis.errorbar(x, observed, yerr=np.maximum(0, np.array([
            observed - intervals[:, 0], intervals[:, 1] - observed
        ])), fmt="o-", capsize=2, markersize=4, label="Held-out MC (68% Wilson)")
        axis.set(xlim=(0, 1), ylim=(0, 1), title=title)
        axis.grid(alpha=.2)
        n = int(frame.n.sum())
        ece = metrics["expected_calibration_error"]
        axis.text(.04, .94, f"N = {n:,}\nECE = {100 * ece:.2f} pp\nOccupied bins: {len(frame)}/20",
                  transform=axis.transAxes, va="top", fontsize=9)
        if index in (0, 2):
            axis.set_ylabel("Observed trigger rate" if index == 0 else
                            f"Observed {'FD' if target == 'fd' else 'any-region'} reconstruction rate")
        if index in (2, 3):
            axis.set_xlabel("Mean predicted probability")
        if index == 0:
            axis.legend(loc="lower right", fontsize=8)
    figure.suptitle(f"Probability closure: electron trigger and 3 {'FD' if target == 'fd' else 'any-region'} hadron gates\n"
                    f"Hadron representative seed {representative}; electron baseline unchanged", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .95))
    name = "probability_closure_four_species.png" if target == "fd" else "probability_closure_four_species_any.png"
    figure.savefig(root / name, dpi=180)
    plt.close(figure)
    return {"n": int(electron_curve.n.sum()), **electron_metrics,
            "reference_run": str(electron.resolve()), "retrained": False}


def _plot_kinematic(root: Path, directories: dict[tuple[int, int], Path],
                     seeds: list[int], representative: int, variable: str) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 6.6), sharex="col", sharey="row",
                                gridspec_kw={"height_ratios": [3, 1]})
    for column, pid in enumerate(SPECIES):
        frames = [pd.read_csv(directories[(seed, pid)] / f"efficiency_fd_vs_{variable}.csv") for seed in seeds]
        frame = frames[seeds.index(representative)]
        valid = frame.eligible.astype(bool)
        x = ((frame.low + frame.high) / 2).to_numpy()[valid]
        observed = frame.observed_efficiency.to_numpy()[valid]
        prediction = np.stack([current.fm_mean_probability.to_numpy()[valid] for current in frames])
        mean = prediction.mean(axis=0)
        sd = prediction.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(x))
        axes[0, column].errorbar(x, observed, yerr=np.maximum(0, np.array([
            observed - frame.wilson_low.to_numpy()[valid], frame.wilson_high.to_numpy()[valid] - observed
        ])), fmt="o", markersize=4, capsize=2, label="Full MC (68% Wilson)")
        axes[0, column].plot(x, mean, "s-", markersize=4, label=f"FM: mean of {len(seeds)} seeds")
        axes[0, column].fill_between(x, mean - sd, mean + sd, alpha=.22, label="FM seed SD")
        axes[0, column].set(title=LABELS[pid], ylim=(-.025, 1.025))
        axes[1, column].plot(x, 100 * (mean - observed), "o-")
        axes[1, column].fill_between(x, 100 * (mean - observed - sd),
                                     100 * (mean - observed + sd), alpha=.22)
        axes[1, column].axhline(0, color=".4", linestyle="--")
        axes[1, column].set_xlabel("Generated momentum [GeV]" if variable == "gen_p" else "Generated polar angle [deg]")
        for axis in axes[:, column]:
            axis.grid(alpha=.25)
    axes[0, 0].set_ylabel("FD reconstruction efficiency given electron trigger")
    axes[1, 0].set_ylabel("FM-MC [pp]")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Hadron reconstruction closure: all generated hadrons in triggered events")
    figure.tight_layout()
    figure.savefig(root / f"hadron_efficiency_vs_{variable}.png", dpi=180)
    plt.close(figure)


def _plot_two_dimensional(root: Path, directories: dict[tuple[int, int], Path],
                          seeds: list[int], representative: int) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(12.2, 10), sharex=True, sharey=True)
    differences = []
    matrices = []
    for pid in SPECIES:
        frames = [pd.read_csv(directories[(seed, pid)] / "efficiency_fd_p_theta.csv") for seed in seeds]
        frame = frames[seeds.index(representative)]
        p_edges = np.unique(np.r_[frame.p_low, frame.p_high])
        theta_edges = np.unique(np.r_[frame.low, frame.high])
        observed = np.full((len(theta_edges) - 1, len(p_edges) - 1), np.nan)
        predicted = observed.copy()
        average = np.mean(np.stack([current.fm_mean_probability.to_numpy() for current in frames]), axis=0)
        for index, row in frame.iterrows():
            if bool(row.eligible):
                position = (int(row.bin_index), int(row.p_bin_index))
                observed[position] = row.observed_efficiency
                predicted[position] = average[index]
        matrices.append((observed, predicted, p_edges, theta_edges))
        differences.append(predicted - observed)
    limit = max(.01, max(float(np.nanmax(np.abs(difference))) for difference in differences))
    for column, (pid, (observed, predicted, p_edges, theta_edges)) in enumerate(zip(SPECIES, matrices)):
        for row, matrix in enumerate((observed, predicted)):
            mesh = axes[row, column].pcolormesh(p_edges, theta_edges, matrix, vmin=0, vmax=1, cmap="viridis")
            figure.colorbar(mesh, ax=axes[row, column], label="Efficiency", fraction=.05)
        mesh = axes[2, column].pcolormesh(p_edges, theta_edges, 100 * (predicted - observed),
                                        vmin=-100 * limit, vmax=100 * limit, cmap="RdBu_r")
        figure.colorbar(mesh, ax=axes[2, column], label="FM-MC [pp]", fraction=.05)
        axes[0, column].set_title(LABELS[pid])
        axes[2, column].set_xlabel("Generated momentum [GeV]")
    for row, label in enumerate(("Full MC", f"FM: {len(seeds)}-seed mean", "Closure difference")):
        axes[row, 0].set_ylabel(f"{label}\nGenerated polar angle [deg]")
    figure.suptitle("FD reconstruction given electron trigger: momentum-angle closure\n"
                    "Blank cells have fewer than the configured minimum MC count")
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(root / "hadron_efficiency_p_theta.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/hadron_efficiency"))
    parser.add_argument("--electron-run", type=Path, default=Path("runs/gpu_electron_efficiency_full"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260822, 20260823, 20260824])
    parser.add_argument("--representative-seed", type=int, default=20260822)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.representative_seed not in args.seeds:
        raise ValueError("Seeds must be distinct and contain the representative seed")
    rows, directories = _collect(args.root, args.seeds)
    summary_dir = args.root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(summary_dir / "per_run_metrics.csv", index=False)
    measured = ["observed_rate", "mean_predicted_probability", "mean_bias", "log_loss", "brier_score",
                "expected_calibration_error", "brier_skill_vs_constant", "gen_p_weighted_mae",
                "gen_theta_weighted_mae", "gen_phi_weighted_mae", "gen_vz_weighted_mae",
                "gen_p_theta_weighted_mae"]
    summary_rows = []
    for (pid, target), group in rows.groupby(["gen_pid", "target"], sort=False):
        row = {"gen_pid": int(pid), "target": target, "seeds": len(group), "test_rows": int(group.n.iloc[0])}
        for name in measured:
            row[f"{name}_mean"] = float(group[name].mean())
            row[f"{name}_std"] = float(group[name].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(summary_dir / "condition_summary.csv", index=False)
    electron = _plot_calibration(summary_dir, directories, args.representative_seed, args.electron_run, "fd")
    _plot_calibration(summary_dir, directories, args.representative_seed, args.electron_run, "any")
    for variable in ("gen_p", "gen_theta"):
        _plot_kinematic(summary_dir, directories, args.seeds, args.representative_seed, variable)
    _plot_two_dimensional(summary_dir, directories, args.seeds, args.representative_seed)
    summary = {
        "network_seeds": args.seeds, "representative_seed": args.representative_seed,
        "seed_variability": "sample SD over network seeds; same held-out events, not independent MC samples",
        "hadron_conditioning": "all generated hadrons of each species in T=1 events; no response-quality cuts",
        "electron": electron, "hadron_results": summary_rows,
        "limitations": ["Calibration is not event-level correlation closure.",
                         "Acceptance times reconstruction, not intrinsic tracking efficiency.",
                         "Single phase-space production; no run-condition or physics-generator transfer claim."],
    }
    with (summary_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Summary and figures: {summary_dir}")


if __name__ == "__main__":
    main()
