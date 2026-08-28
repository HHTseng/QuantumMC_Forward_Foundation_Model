from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def write_history(history: list[dict[str, Any]], path: Path) -> None:
    write_json(history, path)


def write_model_card(
    path: Path,
    config: dict[str, Any],
    audit: dict[str, Any],
    metrics: dict[str, Any],
    best_epoch: int,
    parameter_count: int,
) -> None:
    selected = audit["selected_population"]
    selected_summary = "\n".join(
        f"- PID {row['gen_pid']}: {row['rows']:,} selected rows across {row['events']:,} events"
        for row in selected
    )
    closure = metrics["closure"]
    closure_summary = "\n".join(
        f"- {row['species']} {row['target']}: W1={row['wasserstein_1d']:.5g}, "
        f"mean difference={row['absolute_mean_difference']:.5g}, std ratio={row['std_ratio']:.4f}"
        for row in closure
    )
    test = metrics["test"]
    worst_pid_bin = max(
        metrics["pid_conditional_closure"]["bin_summary"],
        key=lambda row: row["total_variation_distance"],
    )
    text = f"""# Model card: CLAS12 Forward FM step-one FD response seed

## Intended use

This checkpoint samples raw Forward Detector residuals `(delta_p, delta_theta,
delta_phi)` and reconstructed PID for a hadron that is already known to be in
the selected, triggered, FD-reconstructed population. It is a seed component,
not yet a complete detector surrogate.

It must not be used to estimate the electron trigger probability or hadron
reconstruction efficiency: the supplied FD-cuts view conditions those failures
away.

## Data and selection

- Dataset: `{audit['dataset_glob']}`
- Dataset metadata fingerprint: `{audit['dataset_metadata_sha256']}`
- Parquet files: {audit['dataset_file_count']} ({audit['dataset_total_bytes']:,} bytes)
- Split: deterministic 80/10/10 hash of `(source_file_id, event_id)`; verified event-disjoint
- Raw residual targets: no corrected response columns exist in this skim
- Quality policy: reciprocal match, `rec_pid != 0`, `rec_beta > -99`, and configurable `|delta_p| <= 10 GeV`
- The quality policy is an explicit modeling choice and does not alter the source Parquet.

Selected population before deterministic training subsampling:

{selected_summary}

## Model

- Conditional diagonal Gaussian mixture-density network with a shared particle backbone
- Shared across generated pi-, pi+, and proton via a learned species embedding
- Periodic generated phi encoded as sine/cosine
- Joint reconstructed-PID categorical head
- Trainable parameters: {parameter_count:,}
- Best validation epoch: {best_epoch}

## Held-out test metrics

- Residual negative log likelihood (standardized coordinates): {test['residual_nll']:.6f}
- Reconstructed-PID cross entropy: {test['pid_cross_entropy']:.6f}
- Reconstructed-PID top-1 accuracy: {test['pid_accuracy']:.4%}
- PID maximum marginal probability discrepancy: {metrics['pid']['max_absolute_fraction_difference']:.4%}
- Worst fixed-bin conditional PID total-variation distance: {worst_pid_bin['total_variation_distance']:.4f} ({worst_pid_bin['generated_species']}, {worst_pid_bin['p_low_gev']:g}-{worst_pid_bin['p_high_gev']:g} GeV)
- Physical sampled `(p, theta)` fraction: {metrics['joint_and_physical']['physical_sample_fraction']:.4%}
- Sampled reconstructed-theta fraction inside the conditioned 33-degree selection: {metrics['joint_and_physical']['sampled_rec_theta_below_33deg_fraction']:.4%}

One-dimensional sampled closure:

{closure_summary}

The fixed-bin PID closure compares observed reconstructed-class fractions with
the mean PID-head softmax probabilities for the same held-out particles. It is
not top-1 classification accuracy. Full class-by-class values and uncertainties
are saved in `pid_response_fixed_bins.csv`.

## Known limitations

- This is a classical baseline trained on a deterministic subset, not the final foundation model.
- It models raw residuals because versioned energy-loss/swum-back-phi corrected columns were not delivered.
- It does not model trigger, unreconstructed/FD/CD outcome probabilities, electron response, event correlations, run-condition variation, or CD residuals.
- Gaussian components are diagonal; mixture membership captures some joint dependence, but a flow/full-covariance model may improve correlations and tails.
- The `|delta_p| <= 10 GeV` policy removes a tiny pathological population and needs a documented ablation before a physics release.
- Release tolerances for closure have not yet been approved by the analysis group. These metrics are diagnostics, not a physics sign-off.
"""
    path.write_text(text, encoding="utf-8")
