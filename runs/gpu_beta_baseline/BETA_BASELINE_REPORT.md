# Beta-response baseline report

## Outcome

The opt-in beta-response model was trained on one GPU from source commit
`be39ff1`. The selected checkpoint is epoch 14 and has SHA-256
`31e2c65ac417081123c87edf3fc7d874e618739b8b7cdd061c2ef3f92a102078`.
It uses the same Parquet dataset fingerprint as the original full run:
`6a7245cb0ec4125610b9dcd8c1635d70a7773eeb2b29d146dd80d5f149eb43ab`.

The continuous target is

$$
\Delta_4=(\Delta p,\Delta\theta,\Delta\phi,\Delta\beta),
\qquad
\Delta\beta=\beta_{\rm rec}
-\frac{p_{\rm gen}}{\sqrt{p_{\rm gen}^2+m_s^2}}.
$$

The reconstructed-PID categorical head is retained. Beta is an additional
continuous output, not an input and not a replacement for direct PID.

## Audited beta domain

The old `rec_beta > -99` sentinel rule still admitted rare pathological values
as low as about -95 and as high as 749. This run used the explicit teacher
domain $0<\beta_{\rm rec}\le1.2$ without clipping.

| generated species | rows before beta rule | at/below 0 | above 1.2 | rows after rule |
|---|---:|---:|---:|---:|
| $\pi^-$ | 458,373 | 154 | 1,584 | 456,635 |
| $\pi^+$ | 571,241 | 234 | 1,939 | 569,068 |
| proton | 559,627 | 184 | 989 | 558,454 |

The deterministic event-disjoint partitions contained 1,266,603 training,
159,072 validation, and 158,482 test particles after this rule.

## Training

- Architecture: 8-component conditional MDN, width 256, 4 hidden layers,
  shared generated-species embedding, and categorical PID head.
- Trainable parameters: 226,436.
- Seed: 20260822.
- Early stopping: epoch 19; restored best epoch 14.
- Held-out residual NLL: -5.260970 in standardized four-target coordinates.
- Held-out PID cross entropy: 1.020072.
- Held-out PID top-1 accuracy: 0.675244.

The NLL is not directly comparable to the original three-dimensional NLL
because this likelihood contains a fourth standardized response dimension.

## Beta closure

| generated species | test rows | beta W1 | mean difference | sampled/observed width | sampled inside fitted domain |
|---|---:|---:|---:|---:|---:|
| $\pi^-$ | 45,817 | 0.006462 | 0.000356 | 0.9884 | 99.906% |
| $\pi^+$ | 56,774 | 0.006624 | 0.001191 | 1.0181 | 99.870% |
| proton | 55,891 | 0.002578 | 0.000032 | 1.0096 | 99.891% |

The largest fixed-bin beta W1 is 0.012664 for generated $\pi^-$ at 0--1 GeV.
The largest fixed-bin mean difference is 0.007230 for generated $\pi^-$ at
1--2 GeV. The model reproduces global means and widths well, while the
two-dimensional response plot shows that low-momentum tails remain an area for
more detailed study.

Across the original nine kinematic residual/species marginals, mean W1 changed
from 0.02482 to 0.02226. This average hides mixed behavior: most marginals
improved, while $\pi^+$ $\Delta p$ and $\Delta\phi$ worsened and should remain
visible in model selection.

## Conditional PID closure

Both checkpoints were also evaluated on the beta branch's exact 158,482 test
particles, so the small beta-validity selection change cannot explain the PID
difference.

| generated species | COATJAVA correct fraction | original FM mean probability | beta FM mean probability |
|---|---:|---:|---:|
| $\pi^-$ | 0.549971 | 0.546963 | 0.550227 |
| $\pi^+$ | 0.590640 | 0.406701 | 0.582493 |
| proton | 0.814299 | 0.577392 | 0.813916 |

The worst fixed-bin PID total-variation distance changed from 0.466220 in the
original checkpoint ($\pi^+$, 0--1 GeV) to 0.121610 in the beta checkpoint
($\pi^-$, 0--1 GeV). This is encouraging evidence that beta multitask learning
regularized the shared representation. It is one seeded baseline, not yet proof
that beta alone caused the improvement; repeat seeds and a same-selection
three-target ablation are needed.

## Reproduce

From the repository root in an environment containing the dependencies:

```bash
python train.py --config configs/gpu_beta_baseline.yaml
python sample.py \
  --checkpoint runs/gpu_beta_baseline/model.pt \
  --input example_generated_hadrons.csv \
  --output runs/gpu_beta_baseline/example_beta_samples.csv
python runs/gpu_beta_baseline/compare_original_pid_same_test.py
```

The comparison script requires the published original checkpoint at
`runs/tara_gpu_full/model.pt`.

## Interpretation boundary

No beta-derived PID is implemented here. Reproducing COATJAVA's decision from
sampled beta and reconstructed momentum requires an explicit, versioned PID
rule or calibration. The current beta-versus-momentum plots are response
closure tests, and the softmax head remains the direct PID model.
