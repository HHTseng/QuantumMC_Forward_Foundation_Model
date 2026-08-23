# Model card: CLAS12 Forward FM step-one FD response seed

## Intended use

This checkpoint samples raw Forward Detector residuals `(delta_p, delta_theta,
delta_phi)` and reconstructed PID for a hadron that is already known to be in
the selected, triggered, FD-reconstructed population. It is a seed component,
not yet a complete detector surrogate.

It must not be used to estimate the electron trigger probability or hadron
reconstruction efficiency: the supplied FD-cuts view conditions those failures
away.

## Data and selection

- Dataset: `/Users/huan-hsintseng/Downloads/QuantumMC_Simulations/phase-space_parquet-Aug17-26/particle_responses/*.parquet`
- Dataset metadata fingerprint: `6a7245cb0ec4125610b9dcd8c1635d70a7773eeb2b29d146dd80d5f149eb43ab`
- Parquet files: 40 (2,061,164,285 bytes)
- Split: deterministic 80/10/10 hash of `(source_file_id, event_id)`; verified event-disjoint
- Raw residual targets: no corrected response columns exist in this skim
- Quality policy: reciprocal match, `rec_pid != 0`, `rec_beta > -99`, and configurable `|delta_p| <= 10 GeV`
- The quality policy is an explicit modeling choice and does not alter the source Parquet.

Selected population before deterministic training subsampling:

- PID -211: 458,373 selected rows across 458,373 events
- PID 211: 571,241 selected rows across 571,241 events
- PID 2212: 559,627 selected rows across 559,627 events

## Model

- Conditional diagonal Gaussian mixture-density network with a shared particle backbone
- Shared across generated pi-, pi+, and proton via a learned species embedding
- Periodic generated phi encoded as sine/cosine
- Joint reconstructed-PID categorical head
- Trainable parameters: 41,543
- Best validation epoch: 12

## Held-out test metrics

- Residual negative log likelihood (standardized coordinates): -4.078913
- Reconstructed-PID cross entropy: 1.279365
- Reconstructed-PID top-1 accuracy: 66.6156%
- PID maximum marginal probability discrepancy: 0.6915%
- Physical sampled `(p, theta)` fraction: 95.6667%
- Sampled reconstructed-theta fraction inside the conditioned 33-degree selection: 95.3178%

One-dimensional sampled closure:

- pi- delta_p: W1=0.075442, mean difference=0.020727, std ratio=0.9678
- pi- delta_theta: W1=0.008825, mean difference=0.0037103, std ratio=0.8107
- pi- delta_phi: W1=0.039931, mean difference=0.00068714, std ratio=0.8511
- pi+ delta_p: W1=0.078959, mean difference=0.016787, std ratio=0.9343
- pi+ delta_theta: W1=0.0083501, mean difference=4.7993e-05, std ratio=0.7613
- pi+ delta_phi: W1=0.021371, mean difference=0.00016311, std ratio=0.9352
- proton delta_p: W1=0.072003, mean difference=0.020772, std ratio=0.8970
- proton delta_theta: W1=0.0058182, mean difference=0.00057551, std ratio=0.9391
- proton delta_phi: W1=0.0099976, mean difference=0.0038283, std ratio=0.9548

## Known limitations

- This is a classical baseline trained on a deterministic subset, not the final foundation model.
- It models raw residuals because versioned energy-loss/swum-back-phi corrected columns were not delivered.
- It does not model trigger, unreconstructed/FD/CD outcome probabilities, electron response, event correlations, run-condition variation, or CD residuals.
- Gaussian components are diagonal; mixture membership captures some joint dependence, but a flow/full-covariance model may improve correlations and tails.
- The `|delta_p| <= 10 GeV` policy removes a tiny pathological population and needs a documented ablation before a physics release.
- Release tolerances for closure have not yet been approved by the analysis group. These metrics are diagnostics, not a physics sign-off.
