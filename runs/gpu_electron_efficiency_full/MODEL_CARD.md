# Model card: generated-electron trigger efficiency

## Intended use

This checkpoint estimates the event-level probability

$$
eta_T(x_e)=P(T=1 | x_e),
$$

where `T=1` means that the event's generated electron yields the valid trigger
electron. Inputs contain generated truth only. The model does not generate
reconstructed electron kinematics.

## Denominator and labels

- Denominator: `gen_pid = 11` (one PID-11 truth row per event)
- Events/generated electrons: 5,000,000/5,000,000
- Trigger successes: 2,471,543
- Trigger failures: 2,528,457
- Label: `has_valid_trigger_electron`; positive rows satisfy `trigger_mcindex = mcindex`
- Dataset fingerprint: `6a7245cb0ec4125610b9dcd8c1635d70a7773eeb2b29d146dd80d5f149eb43ab`
- Split: deterministic event-disjoint 80/10/10 hash

`is_generated_trigger_electron` is not the denominator in this production: it
equals trigger success and would remove every failure.

## Architecture and objective

- Active truth features: ['log1p_gen_p', 'gen_theta', 'sin_gen_phi', 'cos_gen_phi', 'gen_vz']
- Constant audited features removed before training: ['gen_vx', 'gen_vy']
- Shared MLP with Bernoulli trigger and categorical reconstruction-outcome heads
- Unweighted binary cross entropy preserves the efficiency probability target
- Trainable parameters: 202,502
- Best validation epoch: 12

## Held-out test results

- Observed / mean predicted trigger rate: 0.494381 / 0.492075
- Trigger log loss: 0.228603
- Brier score: 0.066686
- Expected calibration error: 0.004085
- ROC AUC (secondary discrimination metric): 0.946356
- Average precision: 0.908242
- Maximum absolute reported phase-space-bin difference: 0.010502
- Outcome cross entropy / argmax accuracy: 0.228671 / 91.8166%

## Scope limitation

The present data encode all failures as `unreconstructed` and all successes as
`FD`. The outcome head is therefore equivalent to the trigger decision rather
than evidence of a learned general FD/FT/CD reconstruction model. A future
dataset with independent region outcomes is required for that factor.
