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

- Residual negative log likelihood (standardized coordinates): -5.314118
- Reconstructed-PID cross entropy: 0.979566
- Reconstructed-PID top-1 accuracy: 67.9580%
- PID maximum marginal probability discrepancy: 0.0490%
- Worst fixed-bin conditional PID total-variation distance: 0.0278 (pi+, 8-9 GeV)
- Physical sampled `(p, theta)` fraction: 99.0838%
- Sampled reconstructed-theta fraction inside the conditioned 33-degree selection: 99.1059%

One-dimensional sampled closure:

- pi- delta_p: W1=0.027148, mean difference=0.022421, std ratio=1.0164
- pi- delta_theta: W1=0.0017773, mean difference=0.0010623, std ratio=1.0142
- pi- delta_phi: W1=0.012521, mean difference=0.0053896, std ratio=0.9689
- pi+ delta_p: W1=0.014757, mean difference=0.0083905, std ratio=1.0096
- pi+ delta_theta: W1=0.0010076, mean difference=0.00026919, std ratio=1.0038
- pi+ delta_phi: W1=0.0070159, mean difference=0.00036803, std ratio=0.9808
- proton delta_p: W1=0.02238, mean difference=0.019767, std ratio=0.9720
- proton delta_theta: W1=0.00072898, mean difference=0.00022221, std ratio=0.9953
- proton delta_phi: W1=0.0077284, mean difference=0.0020174, std ratio=0.9795

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
