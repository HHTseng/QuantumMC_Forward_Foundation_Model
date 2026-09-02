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

- Dataset: `../phase-space_parquet-Aug17-26/particle_responses/*.parquet`
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
- Trainable parameters: 3,024,476
- Best validation epoch: 68

## Held-out test metrics

- Residual negative log likelihood (standardized coordinates): -5.371373
- Reconstructed-PID cross entropy: 0.978258
- Reconstructed-PID top-1 accuracy: 67.9624%
- PID maximum marginal probability discrepancy: 0.2828%
- Worst fixed-bin conditional PID total-variation distance: 0.0398 (pi-, 8-9 GeV)
- Physical sampled `(p, theta)` fraction: 99.1366%
- Sampled reconstructed-theta fraction inside the conditioned 33-degree selection: 99.1013%

One-dimensional sampled closure:

- pi- delta_p: W1=0.022418, mean difference=0.012394, std ratio=1.0042
- pi- delta_theta: W1=0.0017342, mean difference=0.0013017, std ratio=1.0109
- pi- delta_phi: W1=0.014391, mean difference=0.0030889, std ratio=0.9586
- pi+ delta_p: W1=0.022036, mean difference=0.01368, std ratio=1.0246
- pi+ delta_theta: W1=0.0013503, mean difference=0.00038235, std ratio=1.0193
- pi+ delta_phi: W1=0.013674, mean difference=0.0020025, std ratio=1.0053
- proton delta_p: W1=0.0070211, mean difference=0.0017998, std ratio=1.0020
- proton delta_theta: W1=0.00097126, mean difference=0.00019954, std ratio=1.0252
- proton delta_phi: W1=0.0084407, mean difference=0.00059063, std ratio=1.0108

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
