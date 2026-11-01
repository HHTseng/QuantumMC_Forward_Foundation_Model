#!/usr/bin/env python3
"""Train phase-space reconstruction gates; test unchanged on exclusive rho.

Each gate estimates P(C_h | generated hadron coordinates, T_e=1), where
C_h in {unreconstructed,FD,FT,CD,other}. Inputs are truth quantities only.
Neither reconstructed momentum corrections nor PID labels define success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from forwardfm_hadron.data import SPECIES, OUTCOME_LABELS, load_frame, prepare_splits
from forwardfm_hadron.evaluation import (
    binary_metrics, binned_rows, closure_summary, predict, reliability_rows,
    write_rows, _plot_closure, _two_dimensional_rows, _plot_two_dimensional,
)
from forwardfm_hadron.model import HadronEfficiencyNet
from forwardfm_hadron.training import fit
from forwardfm_step1.training import choose_device, seed_everything
from forwardfm_transfer.data import connect_native


def evaluate_dataset(model, split, train_rates, device, output: Path, min_count: int, feature_bounds):
    output.mkdir(parents=True, exist_ok=True)
    probability = predict(model, split, device, 8192)
    y = split.outcome_target
    one_hot = np.eye(len(OUTCOME_LABELS))[y]
    lower, upper = feature_bounds
    inside_box = np.all((split.continuous >= lower-1e-6) &
                        (split.continuous <= upper+1e-6), axis=1)
    metrics = {
        "n": len(y),
        "outcome_cross_entropy": float(-np.log(np.clip(probability[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "multiclass_brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "counts": {label: int(np.count_nonzero(y == j)) for j, label in enumerate(OUTCOME_LABELS)},
        "kinematic_closure": {},
        "support": {"inside_training_feature_box": int(inside_box.sum()),
                    "outside_training_feature_box": int((~inside_box).sum()),
                    "note": "Coordinate box only; not a joint-density support guarantee."},
    }
    p_edges = np.arange(0, 11, dtype=float)
    theta_edges = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 180], float)
    variables = {
        "gen_p": (split.raw_gen_p, p_edges, r"Generated momentum $p_{\mathrm{gen}}$ [GeV/$c$]"),
        "gen_theta": (np.rad2deg(split.raw_gen_theta), theta_edges, r"Generated polar angle $\theta_{\mathrm{gen}}$ [deg]"),
    }
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "--", color=".5")
    for name, target, prediction, train_rate in (
        ("fd", y == 1, probability[:, 1], train_rates[1]),
        ("any", y != 0, 1 - probability[:, 0], 1 - train_rates[0]),
    ):
        metrics[name] = binary_metrics(target, prediction, float(train_rate))
        metrics["support"][name] = {
            subset: binary_metrics(target[mask], prediction[mask], float(train_rate))
            for subset, mask in (("inside_box", inside_box), ("outside_box", ~inside_box))
            if mask.any()
        }
        calibration = reliability_rows(target, prediction)
        write_rows(calibration, output / f"calibration_{name}.csv")
        occupied = [r for r in calibration if r["n"]]
        axis.plot([r["mean_predicted_probability"] for r in occupied],
                  [r["observed_efficiency"] for r in occupied], "o-", label=name.upper())
        metrics["kinematic_closure"][name] = {}
        for variable, (values, edges, xlabel) in variables.items():
            rows = binned_rows(values, target, prediction, edges, variable, min_count)
            write_rows(rows, output / f"efficiency_{name}_vs_{variable}.csv")
            metrics["kinematic_closure"][name][variable] = closure_summary(rows, len(split))
            _plot_closure(rows, output / f"efficiency_{name}_vs_{variable}.png", xlabel, name.upper())
        cells = _two_dimensional_rows(split, target, prediction, p_edges, theta_edges, min_count)
        write_rows(cells, output / f"efficiency_{name}_p_theta.csv")
        metrics["kinematic_closure"][name]["gen_p_theta"] = closure_summary(cells, len(split))
        _plot_two_dimensional(cells, p_edges, theta_edges, output / f"efficiency_{name}_p_theta.png", name.upper())
    axis.set(xlabel="Mean predicted probability", ylabel="Observed reconstruction fraction",
             xlim=(0, 1), ylim=(0, 1), title="Reconstruction calibration given electron trigger")
    axis.legend()
    axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "calibration.png", dpi=180)
    plt.close(fig)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "predictions.npz", outcome_target=y,
                        outcome_probability=probability, event_keys=split.event_keys,
                        gen_p=split.raw_gen_p, gen_theta=split.raw_gen_theta,
                        gen_phi=split.raw_gen_phi, gen_vz=split.raw_gen_vz,
                        inside_training_feature_box=inside_box)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-parquet", required=True)
    parser.add_argument("--rho-parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--split-seed", type=int, default=20261101)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-train", type=int, default=300000)
    parser.add_argument("--max-val", type=int, default=50000)
    parser.add_argument("--max-test", type=int, default=100000)
    parser.add_argument("--min-bin-count", type=int, default=200)
    parser.add_argument("--species", nargs="+", type=int, choices=SPECIES, default=SPECIES)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for species in args.species:
        if (output / f"pid_{species}" / "model.pt").exists():
            raise FileExistsError(f"Refusing to overwrite trained gate for {species}")
    config = {
        "project": {"seed": args.seed},
        "training": dict(epochs=args.epochs, batch_size=8192, learning_rate=1e-3,
                         weight_decay=1e-5, gradient_clip_norm=5.0, early_stopping_patience=8),
    }
    torch.set_num_threads(4)
    device = choose_device(args.device)
    sources = [Path(__file__), ROOT / "forwardfm_transfer/data.py", ROOT / "forwardfm_electron/data.py",
               ROOT / "forwardfm_step1/data.py", ROOT / "forwardfm_step1/training.py",
               *sorted((ROOT / "forwardfm_hadron").glob("*.py"))]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    environment = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                       duckdb=duckdb.__version__, numpy=np.__version__, device=str(device),
                       gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       source_hashes=source_hashes)
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (output / "resolved_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    con_phase = connect_native(args.phase_parquet)
    con_rho = connect_native(args.rho_parquet)
    summary = []
    try:
        for species in args.species:
            started = time.perf_counter()
            print(f"species={species} seed={args.seed} device={device}: loading", flush=True)
            frames = {
                name: load_frame(con_phase, species, name, limit, args.split_seed)
                for name, limit in (("train", args.max_train), ("validation", args.max_val), ("test", args.max_test))
            }
            frames["rho"] = load_frame(con_rho, species, "rho", args.max_test, args.split_seed)
            splits, scaler, audit = prepare_splits(frames)
            directory = output / f"pid_{species}"
            directory.mkdir(parents=True, exist_ok=True)
            audit.update(species=species, split_seed=args.split_seed,
                         conditioning="has_valid_trigger_electron = true",
                         trigger_definition="EXISTS generated electron mutually associated with first negative-status REC row in FD; no reconstructed-PID requirement",
                         label_definition="mutual reconstruction, then FD/FT/CD/other; PID mistakes remain successes",
                         unweighted=True, phase_parquet=args.phase_parquet, rho_parquet=args.rho_parquet)
            (directory / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
            print({name: len(split) for name, split in splits.items()}, flush=True)
            seed_everything(args.seed)
            model = HadronEfficiencyNet(len(audit["feature_names"]), hidden_width=args.hidden_width,
                                        hidden_layers=args.hidden_layers, dropout=.03)
            model, history, best_epoch = fit(model, splits, config, device)
            checkpoint = dict(format_version=1, task="trigger_conditioned_hadron_efficiency", species=species,
                              model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                              architecture=model.architecture_dict(), feature_scaler=scaler.as_dict(),
                              feature_names=audit["feature_names"],
                              dropped_constant_feature_values=audit["dropped_constant_feature_values"],
                              outcome_labels=list(OUTCOME_LABELS), conditioning=audit["conditioning"],
                              seed=args.seed, split_seed=args.split_seed, best_epoch=best_epoch,
                              label_definition=audit["label_definition"], trigger_definition=audit["trigger_definition"],
                              source_hashes=source_hashes, data_audit=audit)
            torch.save(checkpoint, directory / "model.pt")
            (directory / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            rates = np.bincount(splits["train"].outcome_target, minlength=5) / len(splits["train"])
            bounds = (splits["train"].continuous.min(axis=0), splits["train"].continuous.max(axis=0))
            for name, split_name in (("phase_test", "test"), ("rho_test", "rho")):
                metrics = evaluate_dataset(model, splits[split_name], rates, device, directory / name, args.min_bin_count, bounds)
                for target in ("fd", "any"):
                    summary.append({"species": species, "dataset": name, "target": target,
                                    "seed": args.seed, "best_epoch": best_epoch,
                                    **{key: metrics[target][key] for key in (
                                        "n", "observed_rate", "mean_predicted_probability", "mean_bias",
                                        "expected_calibration_error", "brier_score")},
                                    "p_theta_weighted_mae": metrics["kinematic_closure"][target]["gen_p_theta"]["weighted_mae"],
                                    "outside_training_feature_box_rows": audit["splits"][split_name]["outside_training_feature_box_rows"]})
            print(f"complete species={species} seconds={time.perf_counter()-started:.1f}", flush=True)
            write_rows(summary, output / "summary.csv")
    finally:
        con_phase.close()
        con_rho.close()


if __name__ == "__main__":
    main()
