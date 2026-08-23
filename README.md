# QuantumMC Forward Foundation Model

A physics-aware, stochastic detector-response model for CLAS12 Forward
Detector hadrons. This repository is the first classical seed toward a reusable
forward foundation model: it learns a joint residual distribution and
reconstructed particle identification from generated-particle truth.

## Physics scope

The planned detector surrogate factorizes as

\[
P(Y\mid X)=P(T\mid x_e)\prod_i P(C_i\mid x_i,T)
P(\Delta_i,\widehat{s}_i\mid x_i,T,C_i),
\]

where `T` is the event trigger-electron outcome, `C` is the reconstruction
region/failure outcome, and

\[
\Delta=(p_{\mathrm{rec}}-p_{\mathrm{gen}},\,
\theta_{\mathrm{rec}}-\theta_{\mathrm{gen}},\,
\mathrm{wrap}(\phi_{\mathrm{rec}}-\phi_{\mathrm{gen}})).
\]

The implemented step learns the conditional factor

\[
P(\Delta p,\Delta\theta,\Delta\phi,\widehat{s}\mid
p,\theta,\phi,s,T=1,C=\mathrm{FD})
\]

for generated pi-, pi+, and proton hadrons. Trigger efficiency and the
unreconstructed/FT/FD/CD outcome model are intentionally separate future
components because their denominators require the full all-event dataset.

See [PHYSICS_TO_CODE.md](PHYSICS_TO_CODE.md) for the equation-to-function map
and [PSEUDOCODE.md](PSEUDOCODE.md) for the staged implementation plan.

## Model

- Shared multilayer particle backbone with a learned generated-species
  embedding.
- Periodic generated azimuth represented as `(sin(phi), cos(phi))`.
- Conditional Gaussian mixture-density head for the joint
  `(delta_p, delta_theta, delta_phi)` response.
- Categorical reconstructed-PID head that retains physical misidentification.
- Event-disjoint splitting on the composite key
  `(source_file_id, event_id)`.
- Train-only normalization, seeded stochastic sampling, data-contract audits,
  aggregate and kinematically binned closure tests.

## Dataset and selection

The canonical Aug17-26 phase-space sample contains 20,000,000 particle rows
from 5,000,000 generated four-particle events. The current residual model uses
the following conditional population:

\[
C=\mathrm{FD},\quad \theta_{\mathrm{rec}}<33^\circ,\quad
-5.5<z_{\mathrm{gen}}<-0.5\ \mathrm{cm},\quad T=1.
\]

For the residual-density baseline, reciprocal matching is required and known
PID/beta sentinels plus `|delta_p| > 10 GeV` pathologies are excluded. These
are recorded modeling choices; the source Parquet is never modified.

| Generated species | Selected rows |
|---|---:|
| pi- | 458,373 |
| pi+ | 571,241 |
| proton | 559,627 |
| **Total** | **1,589,241** |

The Parquet data are not stored in this Git repository. The supplied portable
configs expect this layout when commands are run from the repository root:

```text
QuantumMC_Simulations/
├── QuantumMC_Forward_Foundation_Model/
└── phase-space_parquet-Aug17-26/
    └── particle_responses/*.parquet
```

## Held-out results

The full selected-population experiment used one NVIDIA H100 GPU, an
eight-component mixture, a four-layer width-256 backbone, and 222,324 trainable
parameters.

| Split | Rows |
|---|---:|
| Train | 1,270,698 |
| Validation | 159,558 |
| Test | 158,985 |

| Metric | Development seed | Full training |
|---|---:|---:|
| Residual negative log likelihood | -4.0789 | **-4.7581** |
| Reconstructed-PID cross entropy | 1.2794 | **1.1774** |
| Reconstructed-PID top-1 accuracy | 66.62% | **67.22%** |
| Maximum PID marginal discrepancy | 0.692% | **0.426%** |
| Physical sampled `(p, theta)` fraction | 95.67% | **97.13%** |
| Test evaluation throughput | — | **112,434 examples/s** |

### Residual-distribution closure

Blue curves are held-out full simulation and orange curves are stochastic
samples from the learned mixture model. The central response and long tails
are compared independently for every generated species and residual.

![Held-out residual distributions from full simulation and model samples](docs/figures/full_training_residual_closure.png)

### Resolution closure

The heatmap reports
`sampled standard deviation / full-simulation standard deviation`; the ideal
value is one. Aggregate ratios span 0.940–1.065.

![Residual-width closure heatmap](docs/figures/full_training_width_closure.png)

Aggregate agreement does not guarantee uniform conditional fidelity. The
largest remaining discrepancies occur for pi+ momentum response at the lowest
generated polar angles, proton angular response at the lowest generated polar
angles, and pi- azimuth response at low generated momentum. Detailed aggregate
and binned metrics are versioned under `runs/`.

## Installation

```bash
git clone https://github.com/HHTseng/QuantumMC_Forward_Foundation_Model.git
cd QuantumMC_Forward_Foundation_Model
conda create -n QuantumMC python=3.12 -y
conda activate QuantumMC
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Python 3.11–3.13 is supported. DuckDB is used instead of PyArrow to avoid the
page-index metadata issue present in the source Parquet files.

## Training

Fast end-to-end smoke test:

```bash
python train.py --config configs/fd_response_seed.yaml --smoke
```

Development-scale run with automatic CPU/CUDA/MPS selection:

```bash
python train.py --config configs/fd_response_seed.yaml
```

Complete selected-population run on one visible CUDA GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/gpu_full.yaml
```

Every run writes a checkpoint, resolved configuration, dataset audit, model
card, metrics, closure tables, plots, and environment record under `runs/`.
Checkpoints are intentionally ignored by Git because they are reproducible
binary artifacts.

## Sampling

Given a CSV containing `gen_pid`, `gen_p`, `gen_theta`, and `gen_phi` in
GeV/radians:

```bash
python sample.py \
  --checkpoint runs/fd_response_seed/model.pt \
  --input example_generated_hadrons.csv \
  --output sampled_fd_response.csv
```

Inference draws a mixture component and Gaussian noise, samples reconstructed
PID, then applies

\[
p_{\mathrm{rec}}=p_{\mathrm{gen}}+\Delta p,\quad
\theta_{\mathrm{rec}}=\theta_{\mathrm{gen}}+\Delta\theta,\quad
\phi_{\mathrm{rec}}=\mathrm{wrap}(\phi_{\mathrm{gen}}+\Delta\phi).
\]

The sampler flags nonphysical draws instead of silently clipping them.

## Repository map

```text
configs/                 Reproducible development and full-training settings
forwardfm_step1/         Data contract, MDN, training, evaluation, reporting
tests/                   Scaling, leakage, likelihood, and sampling tests
docs/figures/            README-ready held-out result visualizations
runs/                    Versioned metrics, plots, audits, and model cards
train.py                 Training entry point
sample.py                Conditional stochastic inference entry point
PHYSICS_TO_CODE.md       Physics-equation to implementation mapping
PSEUDOCODE.md            Staged modeling plan
```

## Limitations and next milestones

- The current targets are raw residuals because versioned energy-loss and
  swum-back-phi corrected targets were not available.
- The model is conditional on a triggered, selected, reconstructed FD hadron;
  it is not an efficiency model.
- Event correlations, electron response, CD response, condition tokens, and
  analysis-level physics closure are not yet implemented.
- The checkpoint learns its simulation/reconstruction teacher; agreement with
  held-out simulation is not evidence of agreement with real detector data.
- Physics release requires predefined closure gates and event-level validation
  of invariant masses, missing quantities, and target analysis observables.
