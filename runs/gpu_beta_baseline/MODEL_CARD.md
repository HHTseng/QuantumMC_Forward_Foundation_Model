# Model card: CLAS12 Forward FM step-one FD response seed

## Intended use

This checkpoint samples the configured Forward Detector response vector
`(delta_p, delta_theta, delta_phi, delta_beta)` and reconstructed PID for a hadron that is already known to be in
the selected, triggered, FD-reconstructed population. It is a seed component,
not yet a complete detector surrogate.

It must not be used to estimate the electron trigger probability or hadron
reconstruction efficiency: the supplied FD-cuts view conditions those failures
away.

## Data and selection

- Dataset: `../phase-space_parquet-Aug17-26/particle_responses/*.parquet`
- Dataset metadata fingerprint: `6a7245cb0ec4125610b9dcd8c1635d70a7773eeb2b29d146dd80d5f149eb43ab`
- Parquet files: 40 (2,061,164,285 bytes)
- Split: deterministic 80/10/10 hash of `(source_file_id, event_id)`; verified event-disjoint
- Raw residual targets: no corrected response columns exist in this skim
- Quality policy: reciprocal match, `rec_pid != 0`, `rec_beta > -99`, and configurable `|delta_p| <= 10 GeV`; beta-response domain `0.0 < rec_beta <= 1.2` (no clipping)
- The quality policy is an explicit modeling choice and does not alter the source Parquet.

Selected population before deterministic training subsampling:

- PID -211: 456,635 selected rows across 456,635 events
- PID 211: 569,068 selected rows across 569,068 events
- PID 2212: 558,454 selected rows across 558,454 events

## Model

- Conditional diagonal Gaussian mixture-density network with a shared particle backbone
- Shared across generated pi-, pi+, and proton via a learned species embedding
- Periodic generated phi encoded as sine/cosine
- Joint reconstructed-PID categorical head
- Trainable parameters: 226,436
- Best validation epoch: 14

## Held-out test metrics

- Residual negative log likelihood (standardized coordinates): -5.260970
- Reconstructed-PID cross entropy: 1.020072
- Reconstructed-PID top-1 accuracy: 67.5244%
- PID maximum marginal probability discrepancy: 0.3235%
- Worst fixed-bin conditional PID total-variation distance: 0.1216 (pi-, 0-1 GeV)
- Physical sampled `(p, theta)` fraction: 97.7329%
- Sampled reconstructed-theta fraction inside the conditioned 33-degree selection: 97.7600%

One-dimensional sampled closure:

- pi- delta_p: W1=0.042668, mean difference=0.029651, std ratio=1.0035
- pi- delta_theta: W1=0.0031035, mean difference=0.00060933, std ratio=0.9482
- pi- delta_phi: W1=0.020666, mean difference=0.00995, std ratio=0.9203
- pi- delta_beta: W1=0.0065762, mean difference=0.00035645, std ratio=0.9893
- pi+ delta_p: W1=0.07916, mean difference=0.06527, std ratio=1.0270
- pi+ delta_theta: W1=0.0028124, mean difference=8.4913e-06, std ratio=0.9319
- pi+ delta_phi: W1=0.014303, mean difference=0.0079954, std ratio=1.0177
- pi+ delta_beta: W1=0.0066156, mean difference=0.0011915, std ratio=1.0174
- proton delta_p: W1=0.028803, mean difference=0.0042124, std ratio=0.9811
- proton delta_theta: W1=0.0019649, mean difference=0.0010241, std ratio=0.9769
- proton delta_phi: W1=0.0068964, mean difference=0.0017105, std ratio=0.9728
- proton delta_beta: W1=0.0041589, mean difference=3.2077e-05, std ratio=0.9858

The fixed-bin PID closure compares observed reconstructed-class fractions with
the mean PID-head softmax probabilities for the same held-out particles. It is
not top-1 classification accuracy. Full class-by-class values and uncertainties
are saved in `pid_response_fixed_bins.csv`.


## Beta-response closure

The model learns `delta_beta = rec_beta - beta_gen`, where
`beta_gen = p_gen / sqrt(p_gen^2 + m_s^2)`, jointly with the three kinematic
residuals. It retains the categorical reconstructed-PID head.

- pi-: W1=0.006462, mean difference=0.00035645, std ratio=0.9884, sampled in-domain=99.9061%
- pi+: W1=0.0066238, mean difference=0.0011914, std ratio=1.0181, sampled in-domain=99.8697%
- proton: W1=0.0025785, mean difference=3.2074e-05, std ratio=1.0096, sampled in-domain=99.8909%

Fixed-momentum-bin values are saved in `beta_closure_vs_gen_p.csv`. The beta
prediction is a detector-response diagnostic; this baseline does not implement
or claim a COATJAVA-equivalent beta-derived PID rule.


## Known limitations

- This is a classical baseline trained on a deterministic subset, not the final foundation model.
- It models raw residuals because versioned energy-loss/swum-back-phi corrected columns were not delivered.
- It does not model trigger, unreconstructed/FD/CD outcome probabilities, electron response, event correlations, run-condition variation, or CD residuals.
- Gaussian components are diagonal; mixture membership captures some joint dependence, but a flow/full-covariance model may improve correlations and tails.
- The `|delta_p| <= 10 GeV` policy removes a tiny pathological population and needs a documented ablation before a physics release.
- Release tolerances for closure have not yet been approved by the analysis group. These metrics are diagnostics, not a physics sign-off.
