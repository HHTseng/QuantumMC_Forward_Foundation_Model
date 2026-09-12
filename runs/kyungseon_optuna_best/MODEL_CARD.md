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
- Trainable parameters: 3,043,716
- Best validation epoch: 70

## Held-out test metrics

- Residual negative log likelihood (standardized coordinates): -5.828562
- Reconstructed-PID cross entropy: 0.976924
- Reconstructed-PID top-1 accuracy: 67.9945%
- PID maximum marginal probability discrepancy: 0.2062%
- Worst fixed-bin conditional PID total-variation distance: 0.0467 (pi-, 8-9 GeV)
- Physical sampled `(p, theta)` fraction: 98.8573%
- Sampled reconstructed-theta fraction inside the conditioned 33-degree selection: 98.9955%

One-dimensional sampled closure:

- pi- delta_p: W1=0.016424, mean difference=0.0033774, std ratio=1.0030
- pi- delta_theta: W1=0.0017563, mean difference=0.00036781, std ratio=1.0054
- pi- delta_phi: W1=0.014675, mean difference=0.0020265, std ratio=0.9439
- pi- delta_beta: W1=0.0024413, mean difference=0.00018469, std ratio=0.9998
- pi+ delta_p: W1=0.016351, mean difference=0.0058059, std ratio=1.0065
- pi+ delta_theta: W1=0.0011528, mean difference=0.00057693, std ratio=0.9983
- pi+ delta_phi: W1=0.0047974, mean difference=0.0014204, std ratio=0.9806
- pi+ delta_beta: W1=0.0030893, mean difference=0.00019457, std ratio=1.0097
- proton delta_p: W1=0.012299, mean difference=0.00091633, std ratio=0.9956
- proton delta_theta: W1=0.00077423, mean difference=9.7238e-05, std ratio=1.0016
- proton delta_phi: W1=0.0046465, mean difference=0.00068076, std ratio=0.9846
- proton delta_beta: W1=0.001897, mean difference=0.00095569, std ratio=0.9721

The fixed-bin PID closure compares observed reconstructed-class fractions with
the mean PID-head softmax probabilities for the same held-out particles. It is
not top-1 classification accuracy. Full class-by-class values and uncertainties
are saved in `pid_response_fixed_bins.csv`.


## Beta-response closure

The model learns `delta_beta = rec_beta - beta_gen`, where
`beta_gen = p_gen / sqrt(p_gen^2 + m_s^2)`, jointly with the three kinematic
residuals. It retains the categorical reconstructed-PID head.

- pi-: W1=0.0024288, mean difference=0.00018469, std ratio=0.9995, sampled in-domain=99.9236%
- pi+: W1=0.0031338, mean difference=0.00019457, std ratio=1.0100, sampled in-domain=99.8873%
- proton: W1=0.0014385, mean difference=0.00095568, std ratio=0.9950, sampled in-domain=99.9159%

Fixed-momentum-bin values are saved in `beta_closure_vs_gen_p.csv`. The beta
prediction is a detector-response diagnostic; this baseline does not implement
or claim a COATJAVA-equivalent beta-derived PID rule.


## Known limitations

- This is a classical baseline trained on the full selected sample, not the final foundation model.
- It models raw residuals because versioned energy-loss/swum-back-phi corrected columns were not delivered.
- It does not model trigger, unreconstructed/FD/CD outcome probabilities, electron response, event correlations, run-condition variation, or CD residuals.
- Gaussian components are diagonal; mixture membership captures some joint dependence, but a flow/full-covariance model may improve correlations and tails.
- The `|delta_p| <= 10 GeV` policy removes a tiny pathological population and needs a documented ablation before a physics release.
- Release tolerances for closure have not yet been approved by the analysis group. These metrics are diagnostics, not a physics sign-off.
