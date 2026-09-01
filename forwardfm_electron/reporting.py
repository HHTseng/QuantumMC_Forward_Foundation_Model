"""Human-readable model card for the efficiency checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_model_card(
    path: Path,
    audit: dict[str, Any],
    metrics: dict[str, Any],
    best_epoch: int,
    parameter_count: int,
) -> None:
    trigger = metrics["trigger"]
    outcome = metrics["outcome"]
    counts = audit["global_counts"]
    text = f"""# Model card: generated-electron trigger efficiency

## Intended use

This checkpoint estimates the event-level probability

$$
eta_T(x_e)=P(T=1 | x_e),
$$

where `T=1` means that the event's generated electron yields the valid trigger
electron. Inputs contain generated truth only. The model does not generate
reconstructed electron kinematics.

## Denominator and labels

- Denominator: `{audit['denominator_sql']}` (one PID-11 truth row per event)
- Events/generated electrons: {int(counts['events']):,}/{int(counts['generated_electrons']):,}
- Trigger successes: {int(counts['trigger_successes']):,}
- Trigger failures: {int(counts['trigger_failures']):,}
- Label: `{audit['trigger_label']}`; positive rows satisfy `{audit['positive_association_invariant']}`
- Dataset fingerprint: `{audit['dataset_metadata_sha256']}`
- Split: deterministic event-disjoint 80/10/10 hash

`is_generated_trigger_electron` is not the denominator in this production: it
equals trigger success and would remove every failure.

## Architecture and objective

- Active truth features: {audit['active_feature_names']}
- Constant audited features removed before training: {audit['dropped_constant_feature_names']}
- Shared MLP with Bernoulli trigger and categorical reconstruction-outcome heads
- Unweighted binary cross entropy preserves the efficiency probability target
- Trainable parameters: {parameter_count:,}
- Best validation epoch: {best_epoch}

## Held-out test results

- Observed / mean predicted trigger rate: {trigger['observed_rate']:.6f} / {trigger['mean_predicted_probability']:.6f}
- Trigger log loss: {trigger['log_loss']:.6f}
- Brier score: {trigger['brier_score']:.6f}
- Expected calibration error: {trigger['expected_calibration_error']:.6f}
- ROC AUC (secondary discrimination metric): {trigger['roc_auc']:.6f}
- Average precision: {trigger['average_precision']:.6f}
- Maximum absolute reported phase-space-bin difference: {trigger['maximum_absolute_binned_difference']:.6f}
- Outcome cross entropy / argmax accuracy: {outcome['cross_entropy']:.6f} / {outcome['argmax_accuracy']:.4%}

## Scope limitation

The present data encode all failures as `unreconstructed` and all successes as
`FD`. The outcome head is therefore equivalent to the trigger decision rather
than evidence of a learned general FD/FT/CD reconstruction model. A future
dataset with independent region outcomes is required for that factor.
"""
    path.write_text(text, encoding="utf-8")
