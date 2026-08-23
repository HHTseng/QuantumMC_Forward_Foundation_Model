# Tara one-GPU large-scale training report

## Outcome

The enlarged CLAS12 Forward FM step-one model trained successfully on Tara and
saved a complete, reproducible result bundle at:

```text
/home/htseng/QuantumMC_Simulations/step1/runs/tara_gpu_full
```

The selected checkpoint is epoch 16. Training stopped at epoch 21 after five
epochs without a better validation objective.

This remains a conditional FD residual/PID baseline,

\[
P(\Delta p,\Delta\theta,\Delta\phi,\widehat s
\mid x,T=1,C=\mathrm{FD}),
\]

not yet the complete trigger/outcome/response foundation model.

## Transfer and remote layout

- Source: local `/Users/huan-hsintseng/Downloads/QuantumMC_Simulations`
- Destination: Tara `/home/htseng/QuantumMC_Simulations`
- Uploaded size reported by `rsync`: 2,080,140,348 bytes
- Parquet verification: 40 shards, 2,061,164,285 data bytes
- Dataset metadata fingerprint recorded by the run:
  `6a7245cb0ec4125610b9dcd8c1635d70a7773eeb2b29d146dd80d5f149eb43ab`

The transfer used archival `rsync` with partial-transfer recovery. The full
local folder—including design notes, physics guide, dataset, annotated code,
local seed result, and tests—was uploaded.

## Tara environment and GPU isolation

A new Conda environment named `QuantumMC` was created at
`/home/htseng/.conda/envs/QuantumMC` with Python 3.12.13.

Runtime versions:

| Component | Version |
|---|---:|
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| DuckDB | 1.5.5 |
| NumPy | 2.5.2 |
| pandas | 2.3.3 |
| Matplotlib | 3.11.1 |
| PyYAML | 6.0.3 |

Tara has four NVIDIA H100 PCIe GPUs (81,559 MiB each; driver 580.82.07).
The training process was launched with `CUDA_VISIBLE_DEVICES=0`; PyTorch
confirmed `visible_cuda_device_count = 1` and `visible_gpu = NVIDIA H100 PCIe`.

The exact Conda export and explicit package list are saved as
`conda_environment.yml` and `conda_explicit.txt` in the result directory.

## Preflight validation

Before training:

1. all 40 Parquet files were present remotely;
2. PyTorch reported CUDA available and exactly one visible H100;
3. all four uploaded unit tests passed;
4. event split disjointness, composite particle-key uniqueness, and wrapped
   delta-phi range were checked by the training data audit.

## Data used

No row-count cap was applied after the deterministic event-level split. The
complete quality-selected population was used:

| Split | pi- | pi+ | proton | Total |
|---|---:|---:|---:|---:|
| Train | 366,317 | 456,997 | 447,384 | 1,270,698 |
| Validation | 46,075 | 57,245 | 56,238 | 159,558 |
| Test | 45,981 | 56,999 | 56,005 | 158,985 |
| All | 458,373 | 571,241 | 559,627 | 1,589,241 |

The split is a deterministic 80/10/10 hash of the true event identity
`(source_file_id, event_id)`, so particles from one Monte Carlo event cannot
cross partitions.

Selection:

\[
C=\mathrm{FD},\quad \theta_{rec}<33^\circ,\quad
-5.5<z_{gen}<-0.5\ \mathrm{cm},\quad T=1,
\]

plus the explicit residual-density policy: reciprocal match, `rec_pid != 0`,
`rec_beta > -99`, and `|delta_p| <= 10 GeV`. Source Parquet was not modified.

## Enlarged model and optimization

| Setting | Local seed | Tara large run |
|---|---:|---:|
| Training rows | 180,000 | 1,270,698 |
| Hidden width/layers | 128 / 3 | 256 / 4 |
| PID embedding | 8 | 16 |
| Mixture components | 5 | 8 |
| Trainable parameters | 41,543 | 222,324 |
| Batch size | 2,048 | 8,192 |
| Maximum epochs | 12 | 30 |

The optimized likelihood was

\[
\mathcal L=-\mathbb E[\log p_\theta(\Delta\mid x)]
-0.2\,\mathbb E[\log P_\theta(\widehat s\mid x)].
\]

AdamW, gradient-norm clipping at 5, and validation early stopping with patience
5 were used. Data loading and preprocessing took 3.57 seconds. End-to-end wall
time was 5 minutes 15 seconds, including evaluation and artifact generation.
Peak host resident memory was approximately 2.50 GiB.

## Held-out results

| Metric | Local seed | Tara large run |
|---|---:|---:|
| Residual NLL | -4.078913 | **-4.758140** |
| REC-PID cross entropy | 1.279365 | **1.177356** |
| REC-PID top-1 accuracy | 66.62% | **67.22%** |
| PID maximum marginal discrepancy | 0.6915% | **0.4264%** |
| Physical sampled `(p, theta)` fraction | 95.67% | **97.13%** |
| Test evaluation throughput | — | 112,434 examples/s |

Aggregate residual-width ratios, sampled standard deviation divided by the
full-simulation value, lie between 0.940 and 1.065. Residual-correlation matrix
Frobenius discrepancies are 0.037 for pi-, 0.047 for pi+, and 0.076 for
protons—substantially better than the local seed values of roughly
0.098–0.149.

## Remaining closure gaps

Aggregate closure improved, but binned metrics expose important low-angle and
low-momentum deficiencies:

- pi+ lowest generated-theta bin: delta-p Wasserstein distance 0.418 GeV,
  mean discrepancy 0.410 GeV, and width ratio 0.775;
- proton lowest generated-theta bin: delta-theta Wasserstein distance
  0.0295 rad and width ratio 0.665;
- pi- lowest generated-momentum bin: delta-phi Wasserstein distance 0.0970 rad
  and mean discrepancy 0.0903 rad.

These are diagnostics against the held-out simulation teacher. No
collaboration-approved numerical release gate has yet been applied, so this
checkpoint should not be labeled physics-ready.

## Launch record and one corrected pre-training failure

Successful run:

```bash
ssh tara
cd /home/htseng/QuantumMC_Simulations/step1
conda activate QuantumMC
CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/tara_gpu_full.yaml
```

- Start: 2026-08-22T22:22:32Z
- Finish: 2026-08-22T22:27:47Z
- Exit status: 0

The first launch stopped before training because the configuration expressed
“use all rows” as a Top-N value of exactly 1,000,000, while DuckDB requires
that internal N to be smaller than 1,000,000. The loader was corrected to
accept `null` as an explicit unlimited split, local and remote tests were
rerun, and the full population then loaded successfully. The original failure
is preserved in `training_attempt1_duckdb_limit.log`.

## Saved artifacts

- `model.pt` — checkpoint and preprocessing state;
- `model.sha256` — checkpoint checksum
  `22dde8fe78c5bec337e5014be46e4c8037673015bc88d4fbe812c05bebcffe11`;
- `MODEL_CARD.md` — scope, held-out summary, and prohibited uses;
- `metrics.json`, `closure_metrics.csv`, `kinematic_closure_metrics.csv`;
- `residual_closure.png`, `training_history.png`;
- `data_audit.json`, `resolved_config.yaml`, `history.json`, `training.log`;
- `runtime_gpu_environment.json`, `gpu_hardware.csv`;
- `conda_environment.yml`, `conda_explicit.txt`;
- UTC start/finish records and the preserved first-attempt log.

## Recommended next work

1. Investigate the three reported conditional closure gaps with finer
   low-angle/low-momentum occupancy and match-quality plots.
2. Compare diagonal-mixture, full-covariance mixture, and conditional-flow
   response heads using the same frozen split.
3. Run the documented `|delta_p|` and match-policy ablations.
4. Freeze and version energy-loss/swum-back-phi corrections before training a
   corrected target.
5. Return to the all-event dataset to implement `P(T|x_e)` and `P(C|x,T)`;
   only then compose a complete detector-event sampler.

