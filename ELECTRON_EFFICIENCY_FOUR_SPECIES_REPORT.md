# Electron efficiency and four-species response experiment

## Executive summary

This branch implements and validates two separate probability factors:

$$
\widehat\eta_T(x_e)\approx P(T=1\mid x_e)
$$

from one generated electron in every event, and

$$
q_\theta(\Delta,\widehat s
\mid x,s,T=1,C=\mathrm{FD},F=1)
$$

for generated $e^-$, $\pi^-$, $\pi^+$, and protons in the selected conditional
FD population.

The efficiency model is well calibrated on the held-out test split: Brier
score 0.066686, expected calibration error 0.004085, and observed/predicted
integrated trigger rates of 49.4381%/49.2075%. The four-species response model
shows good one-dimensional electron residual closure and useful hadron closure,
but retains electron residual-correlation and hadron PID discrepancies that
should guide the next architecture study.

## 1. Reproducibility contract

| Item | Value |
|---|---|
| Branch | `feature/trigger-electron-efficiency-four-species` |
| Dataset files | 40 Parquet files |
| Dataset rows | 20,000,000 particle rows / 5,000,000 events |
| Dataset metadata SHA-256 | `6a7245cb0ec4125610b9dcd8c1635d70a7773eeb2b29d146dd80d5f149eb43ab` |
| Split | deterministic 80/10/10 hash of `(source_file_id,event_id)` |
| Training accelerator | one NVIDIA H100 PCIe GPU |
| Random seed | 20260822 |
| Unit tests | 15 passed |

Checkpoint fingerprints:

| Checkpoint | Size | SHA-256 |
|---|---:|---|
| `runs/gpu_electron_efficiency_full/model.pt` | 816,371 bytes | `bc26edd3d28211e7814b91bfd15714969f40ddfcebf5032e3b65f454cc10f9e1` |
| `runs/gpu_four_species_full/model.pt` | 897,306 bytes | `0a9a722fb8b532ea6a6d07a06cfed63d7fe3e15ccdbb7d0bb18a0506413c0771` |

## 2. Blocking data audit

The original candidate selector `is_generated_trigger_electron` cannot be the
efficiency denominator. It is false for all trigger failures and equals
`has_valid_trigger_electron` in this production.

The accepted denominator is therefore the truth-only selector

```sql
gen_pid = 11
```

The audit verified:

| Check | Result |
|---|---:|
| One PID-11 row per event | 5,000,000 / 5,000,000 |
| PID-11 row `mcindex` | always 0 |
| Trigger successes | 2,471,543 |
| Trigger failures | 2,528,457 |
| Positive association mismatches | 0 |
| Role-flag/trigger-label mismatches | 0 |
| Event split overlaps | 0 |

The failure encoding is unambiguous in these Parquet files: all 2,528,457
negative electron rows are unreconstructed, while all 2,471,543 positive rows
are reconstructed in FD. Consequently, the outcome head and trigger head learn
the same present binary partition.

## 3. Implementation summary

### 3.1 Electron efficiency

The new `forwardfm_electron` package provides:

- all-event denominator construction;
- truth-only feature scaling;
- Bernoulli trigger and categorical outcome heads;
- unweighted BCE/CE training;
- log-loss, Brier, ECE, ROC-AUC, average precision, and binned probability
  closure;
- calibrated inference through `sample_trigger_electron.py`.

The initial audit retained all seven candidate features. It then found
`gen_vx` and `gen_vy` exactly constant and removed them before training. The
checkpoint's active ordered feature contract is

```text
log1p_gen_p, gen_theta, sin_gen_phi, cos_gen_phi, gen_vz
```

### 3.2 Four-species conditional response

The prior response code is now configuration-driven. Existing configurations
still default to `(-211, 211, 2212)`. The new checkpoint explicitly saves

```text
[11, -211, 211, 2212]
```

as its embedding order. The electron response population is selected
independently of the hadron-only teacher flag, while all species share the
common FD/fiducial/residual-quality policy.

The evaluation code now derives subplot dimensions and species loops from the
checkpoint/configuration. The sampler also reports supported species from the
checkpoint rather than a module constant.

### 3.3 Throughput correction

The first full attempt exposed row-by-row `DataLoader` collation as the
bottleneck: GPU utilization was only 0–2%. The final implementation optionally
preloads the compact train/validation tensors once and performs vectorized GPU
index batching. This preserves the identical rows, objective, splits, and
early-stopping rule while making the full experiments practical on a large
accelerator.

## 4. Efficiency results

### 4.1 Data and optimization

| Split | Rows | Successes | Failures | Trigger rate |
|---|---:|---:|---:|---:|
| Train | 3,999,385 | 1,976,265 | 2,023,120 | 49.4142% |
| Validation | 500,177 | 247,871 | 252,306 | 49.5567% |
| Test | 500,438 | 247,407 | 253,031 | 49.4381% |

The model has 202,502 trainable parameters. Early stopping selected epoch 12
and stopped after epoch 17.

### 4.2 Integrated calibration

| Metric | Value |
|---|---:|
| Observed trigger rate | 0.494381 |
| Mean predicted probability | 0.492075 |
| Signed difference | -0.002306 |
| Binary log loss | 0.228603 |
| Brier score | 0.066686 |
| Expected calibration error | 0.004085 |
| ROC AUC | 0.946356 |
| Average precision | 0.908242 |
| Argmax trigger accuracy | 91.82% |

The ROC metrics show useful separation, while the Brier/ECE and probability
closure are the primary efficiency criteria.

### 4.3 Phase-space closure

| Variable | Maximum absolute bin gap | Bin giving maximum | Rows in bin |
|---|---:|---|---:|
| $p_{e,\rm gen}$ | 0.004815 | 7–8 GeV | 41,945 |
| $\theta_{e,\rm gen}$ | 0.005290 | 20–25 degrees | 74,592 |
| $\phi_{e,\rm gen}$ | 0.008679 | 120–180 degrees | 83,376 |
| $v_{z,e}^{\rm gen}$ | 0.010502 | 4–6 cm | 1,220 |

The last gap is statistically compatible with the sparse bin's observed
standard error (0.01431). More detailed values and uncertainties are saved in
the four `efficiency_vs_*.csv` files.

![Efficiency versus generated momentum](runs/gpu_electron_efficiency_full/efficiency_vs_gen_p.png)

![Efficiency versus generated polar angle](runs/gpu_electron_efficiency_full/efficiency_vs_gen_theta.png)

![Calibration curve](runs/gpu_electron_efficiency_full/calibration_curve.png)

## 5. Four-species response results

### 5.1 Data and optimization

| Split | Rows |
|---|---:|
| Train | 2,610,583 |
| Validation | 327,804 |
| Test | 326,644 |

The model has 222,340 trainable parameters. Early stopping selected epoch 17
and stopped after epoch 22.

| Aggregate metric | Value |
|---|---:|
| Test residual NLL | -5.162882 |
| Test PID cross entropy | 0.489958 |
| Test PID top-1 accuracy | 0.842498 |
| Physical sampled fraction | 0.993381 |
| Sampled reconstructed-theta conditioned fraction | 0.989184 |

The negative NLL is valid for a continuous density in standardized coordinates
and is meaningful only under the same transformation. PID top-1 accuracy is
not a balanced metric because electrons constitute 51.3% of the test rows and
are almost definitionally PID 11 after trigger selection.

### 5.2 Residual closure

| Species | $W_1(\Delta p)$ | $W_1(\Delta\theta)$ | $W_1(\Delta\phi)$ | Width-ratio range |
|---|---:|---:|---:|---:|
| electron | 0.014755 | 0.000252 | 0.001362 | 1.006–1.029 |
| $\pi^-$ | 0.064631 | 0.003255 | 0.017914 | 0.942–1.013 |
| $\pi^+$ | 0.071976 | 0.003733 | 0.018882 | 1.002–1.054 |
| proton | 0.027427 | 0.002238 | 0.017991 | 1.002–1.085 |

![Four-species residual closure](runs/gpu_four_species_full/residual_closure.png)

The electron marginal distributions close well. Joint closure is weaker:
the Frobenius difference between teacher and sampled residual correlation
matrices is 0.404 for electrons, compared with 0.073, 0.139, and 0.084 for
$\pi^-$, $\pi^+$, and protons. A full-covariance mixture or conditional flow
is a motivated next ablation.

### 5.3 PID probability closure

| Species | Teacher correct fraction | FM correct probability | Difference | Worst bin TV |
|---|---:|---:|---:|---:|
| electron | 0.999869 | 0.995822 | -0.004046 | 0.005945 |
| $\pi^-$ | 0.550184 | 0.534365 | -0.015819 | 0.058665 |
| $\pi^+$ | 0.591098 | 0.586805 | -0.004294 | 0.060524 |
| proton | 0.812642 | 0.832070 | +0.019429 | 0.065988 |

![Conditional correct-PID response](runs/gpu_four_species_full/pid_correct_response_vs_gen_p.png)

The electron conditional PID head is close but should be interpreted as a
closure of a highly selected response, not a measurement of electron-ID
efficiency. The pion/proton differences are consistent with the earlier PID
closure concern and remain a priority after the true reconstruction-outcome
factor is implemented.

## 6. Conclusions

1. The generated-electron denominator is now correct and includes all failures.
2. The trigger-efficiency probability is well calibrated globally and across
   the reported generated phase space.
3. The code can train and evaluate electron plus three hadron species without
   changing the legacy three-species configuration.
4. The present outcome head must not be described as a general reconstruction
   efficiency model because its labels are exactly trigger-equivalent.
5. The next model-development targets are particle reconstruction outcomes,
   electron residual correlations, and conditional hadron PID closure.
