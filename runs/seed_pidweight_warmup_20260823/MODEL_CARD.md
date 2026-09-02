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
- Best validation epoch: 70

## Held-out test metrics

- Residual negative log likelihood (standardized coordinates): -5.274901
- Reconstructed-PID cross entropy: 0.977862
- Reconstructed-PID top-1 accuracy: 67.9876%
- PID maximum marginal probability discrepancy: 0.1526%
- Worst fixed-bin conditional PID total-variation distance: 0.0279 (pi-, 8-9 GeV)
- Physical sampled `(p, theta)` fraction: 99.0659%
- Sampled reconstructed-theta fraction inside the conditioned 33-degree selection: 99.0312%

One-dimensional sampled closure:

- pi- delta_p: W1=0.010367, mean difference=0.0019539, std ratio=1.0061
- pi- delta_theta: W1=0.0013145, mean difference=0.00035701, std ratio=1.0026
- pi- delta_phi: W1=0.016972, mean difference=0.0027165, std ratio=0.9570
- pi+ delta_p: W1=0.021094, mean difference=0.015183, std ratio=1.0168
- pi+ delta_theta: W1=0.00094172, mean difference=8.8185e-05, std ratio=1.0023
- pi+ delta_phi: W1=0.011811, mean difference=0.002346, std ratio=0.9960
- proton delta_p: W1=0.027125, mean difference=0.014491, std ratio=0.9945
- proton delta_theta: W1=0.0014256, mean difference=0.00080152, std ratio=1.0166
- proton delta_phi: W1=0.012303, mean difference=0.0024412, std ratio=1.0051

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
