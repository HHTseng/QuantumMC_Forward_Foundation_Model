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

- Dataset: `/home/htseng/QuantumMC_Simulations/phase-space_parquet-Aug17-26/particle_responses/*.parquet`
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
- Trainable parameters: 222,324
- Best validation epoch: 16

## Held-out test metrics

- Residual negative log likelihood (standardized coordinates): -4.758140
- Reconstructed-PID cross entropy: 1.177356
- Reconstructed-PID top-1 accuracy: 67.2170%
- PID maximum marginal probability discrepancy: 0.4264%
- Physical sampled `(p, theta)` fraction: 97.1293%
- Sampled reconstructed-theta fraction inside the conditioned 33-degree selection: 96.7047%

One-dimensional sampled closure:

- pi- delta_p: W1=0.067056, mean difference=0.020924, std ratio=0.9776
- pi- delta_theta: W1=0.0034563, mean difference=0.00010279, std ratio=0.9811
- pi- delta_phi: W1=0.02627, mean difference=0.012425, std ratio=0.9450
- pi+ delta_p: W1=0.052865, mean difference=0.022589, std ratio=0.9402
- pi+ delta_theta: W1=0.00289, mean difference=0.0026293, std ratio=0.9773
- pi+ delta_phi: W1=0.0093093, mean difference=0.00055545, std ratio=0.9716
- proton delta_p: W1=0.050173, mean difference=0.048249, std ratio=1.0646
- proton delta_theta: W1=0.0024164, mean difference=0.0018076, std ratio=1.0029
- proton delta_phi: W1=0.0089714, mean difference=0.0050861, std ratio=1.0461

## Known limitations

- This is a classical baseline trained on a deterministic subset, not the final foundation model.
- It models raw residuals because versioned energy-loss/swum-back-phi corrected columns were not delivered.
- It does not model trigger, unreconstructed/FD/CD outcome probabilities, electron response, event correlations, run-condition variation, or CD residuals.
- Gaussian components are diagonal; mixture membership captures some joint dependence, but a flow/full-covariance model may improve correlations and tails.
- The `|delta_p| <= 10 GeV` policy removes a tiny pathological population and needs a documented ablation before a physics release.
- Release tolerances for closure have not yet been approved by the analysis group. These metrics are diagnostics, not a physics sign-off.
