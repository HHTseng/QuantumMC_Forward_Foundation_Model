# QuantumMC Forward Foundation Model

A stochastic neural surrogate for the CLAS12 detector and reconstruction
chain. Given a generated particle's identity and momentum, the model samples
what CLAS12 reconstruction would report after detector resolution, tails, and
particle misidentification.

This repository is **Step 1** toward a broader forward foundation model. The
current network covers reconstructed hadrons in the CLAS12 Forward Detector
(FD); it does not yet replace the complete event simulation.

## The idea in one picture

Physics analyses normally compare real detector data with a detailed Monte
Carlo path: generate an event, propagate every particle through GEANT4/GEMC,
simulate detector signals, and run reconstruction. That path is accurate but
computationally expensive. A trained forward model is intended to provide a
much faster, statistically faithful shortcut from truth particles to
reconstructed-like particles.

![Real experiment, full computer simulation, and the learned forward-model shortcut](docs/figures/real_simulation_forward_model.png)

The detector bends charged particles in magnetic fields and measures their
tracks, timing, Cherenkov light, and calorimeter deposits. These measurements
set the momentum/angular resolution and the probability of assigning a
particle the wrong identity. The right panel shows the long-term model scope;
the current Step 1 implementation is the **FD hadron residual + reconstructed
PID** portion only.

![CLAS12 particle paths and the intended placement of the forward model](docs/figures/clas12_detector_forward_model_scope.png)

## What goes in and what comes out

Each Parquet row describes one generated particle and, when successfully
matched, its reconstructed counterpart. Events contain an electron, a
$\pi^+$, a $\pi^-$, and a proton. Step 1 trains on the three hadron species:

| Meaning | Data columns | Role in Step 1 | Units |
|---|---|---|---|
| Generated identity | `gen_pid` | Input species $s_{\mathrm{gen}}\in\{-211,211,2212\}$ | PDG code |
| Generated kinematics | `gen_p`, `gen_theta`, `gen_phi` | Continuous inputs $(p_{\mathrm{gen}},\theta_{\mathrm{gen}},\phi_{\mathrm{gen}})$ | GeV, rad, rad |
| Reconstructed-minus-generated response | `delta_p`, `delta_theta`, `delta_phi` | Continuous training labels $\Delta$ | GeV, rad, rad |
| Reconstructed identity | `rec_pid` | Categorical training label $\widehat{s}$; wrong-ID cases are retained | PDG code |
| Vertex and detector state | `gen_vz`, `rec_theta`, `rec_detector_region`, `usable_for_hadron_response_training` | Define the selected conditional population; not network inputs | cm, rad, category, Boolean |
| Match/quality information | `match_reciprocal`, `rec_beta` | Data-quality checks; not network inputs | Boolean, dimensionless |
| Event identity | `source_file_id`, `event_id`, `mcindex` | Prevent event leakage and identify rows; not network inputs | identifiers |

In short, one supervised example is

$$
(s_{\mathrm{gen}},p_{\mathrm{gen}},\theta_{\mathrm{gen}},\phi_{\mathrm{gen}})
\longrightarrow
(\Delta p,\Delta\theta,\Delta\phi,\widehat{s}).
$$

At inference time there are no reconstructed labels. The network receives only
generated truth, samples the residuals and reconstructed PID, and produces a
new reconstructed-like particle.

## Physics and mathematics handled by the code

### 1. Particle state and detector-response factorization

A truth particle is represented in spherical momentum coordinates as

$$
x_i=(p_i,\theta_i,\phi_i,s_i),
\qquad
p_i=\sqrt{p_{x,i}^2+p_{y,i}^2+p_{z,i}^2},
$$

$$
\theta_i=\cos^{-1}\!\left(\frac{p_{z,i}}{p_i}\right),
\qquad
\phi_i=\operatorname{atan2}(p_{y,i},p_{x,i}),
$$

where $s_i$ is the PDG particle identity. For a complete generated event
$X=(x_e,x_1,\ldots,x_n)$, the planned detector surrogate is factorized as

$$
P(Y\mid X)
=P(T\mid x_e)
\prod_i
P(C_i\mid x_i,T)
P(\Delta_i,\widehat{s}_i\mid x_i,T,C_i).
$$

Here:

- $T$ is the event trigger-electron outcome;
- $C_i$ is the particle outcome (not reconstructed, or reconstructed in
  FT/FD/CD);
- $\Delta_i$ is the detector/reconstruction kinematic response;
- $\widehat{s}_i$ is reconstructed PID.

The present data have already been conditioned on a triggered event and an FD
hadron. Therefore, Step 1 learns only

$$
P(\Delta p,\Delta\theta,\Delta\phi,\widehat{s}
\mid p,\theta,\phi,s,T=1,C=\mathrm{FD},\text{fiducial}).
$$

This distinction matters: Step 1 models **how a selected FD particle is
reconstructed**, not whether an arbitrary generated particle triggers or is
reconstructed.

### 2. Residual labels

The continuous labels quantify the change from truth to reconstruction:

$$
\Delta p=p_{\mathrm{rec}}-p_{\mathrm{gen}},
\qquad
\Delta\theta=\theta_{\mathrm{rec}}-\theta_{\mathrm{gen}},
$$

$$
\Delta\phi
=\operatorname{wrap}(\phi_{\mathrm{rec}}-\phi_{\mathrm{gen}})
=\bigl((\phi_{\mathrm{rec}}-\phi_{\mathrm{gen}}+\pi)\bmod 2\pi\bigr)-\pi.
$$

Wrapping keeps angular differences in $[-\pi,\pi)$, so particles close to the
$-\pi/\pi$ boundary remain close numerically.

### 3. Physics-aware input coordinates

The generated variables are transformed before entering the neural network:

$$
f(x)=\left[
\log\!\left(1+\frac{p_{\mathrm{gen}}}{1\,\mathrm{GeV}}\right),
\theta_{\mathrm{gen}},
\sin\phi_{\mathrm{gen}},
\cos\phi_{\mathrm{gen}}
\right].
$$

The logarithm compresses the momentum range. The sine/cosine pair represents
azimuth on a circle without introducing an artificial discontinuity at
$\phi=\pm\pi$. Generated species enters through a learned embedding. Features
and targets are standardized with training-set statistics only,

$$
z_j=\frac{q_j-\mu_j^{\mathrm{train}}}{\sigma_j^{\mathrm{train}}},
$$

and are converted back to GeV/radians for evaluation. This is a numerical
coordinate change, not a detector correction.

### 4. A distribution, not a single correction

Detector response is stochastic and often non-Gaussian. A deterministic
regression would average away resolution and tails, so the model uses a
conditional mixture-density network (MDN):

$$
p_\vartheta(\Delta\mid x)
=\sum_{k=1}^{K}\pi_k(x)
\prod_{j\in\{p,\theta,\phi\}}
\mathcal{N}\!\left(
\Delta_j;\mu_{kj}(x),\sigma_{kj}^2(x)
\right).
$$

The current full model has $K=8$ Gaussian components. A separate softmax head
models particle identification,

$$
P_\vartheta(\widehat{s}=c\mid x)
=\operatorname{softmax}(a(x))_c.
$$

Training minimizes residual negative log likelihood plus reconstructed-PID
cross entropy,

$$
\mathcal{L}
=-\mathbb{E}_{\mathrm{data}}\!\left[\log p_\vartheta(\Delta\mid x)\right]
-\lambda_{\mathrm{PID}}
\mathbb{E}_{\mathrm{data}}\!\left[
\log P_\vartheta(\widehat{s}\mid x)
\right],
\qquad \lambda_{\mathrm{PID}}=0.20.
$$

### 5. Sampling reconstructed-like particles

For each generated particle, inference draws a mixture component and random
noise,

$$
k\sim\operatorname{Categorical}(\pi(x)),
\qquad
\epsilon\sim\mathcal{N}(0,I),
\qquad
\Delta=\mu_k(x)+\sigma_k(x)\odot\epsilon,
$$

then restores reconstructed kinematics:

$$
p_{\mathrm{rec}}=p_{\mathrm{gen}}+\Delta p,
\qquad
\theta_{\mathrm{rec}}=\theta_{\mathrm{gen}}+\Delta\theta,
$$

$$
\phi_{\mathrm{rec}}
=\operatorname{wrap}(\phi_{\mathrm{gen}}+\Delta\phi).
$$

The sampled reconstructed PID comes from the categorical head. The sampler
flags nonphysical draws such as $p_{\mathrm{rec}}\leq0$ or
$\theta_{\mathrm{rec}}\notin[0,\pi]$ instead of silently clipping them.

For the exact equation-to-function mapping, see
[PHYSICS_TO_CODE.md](PHYSICS_TO_CODE.md). The staged roadmap is in
[PSEUDOCODE.md](PSEUDOCODE.md).

## Dataset and selection

The Aug17-26 phase-space sample contains **20,000,000 particle rows** from
**5,000,000 generated four-particle events**. The immutable source Parquet is
not stored in this repository.

The current response model selects

$$
C=\mathrm{FD},
\qquad
\theta_{\mathrm{rec}}<33^\circ,
\qquad
-5.5<z_{\mathrm{gen}}<-0.5\ \mathrm{cm},
\qquad
T=1.
$$

Reciprocal truth/reconstruction matching is required. Known PID and beta
sentinels and pathological $|\Delta p|>10\ \mathrm{GeV}$ rows are excluded.
These are explicit modeling choices; the source data are never modified.

| Generated species | PDG code | Selected rows |
|---|---:|---:|
| $\pi^-$ | -211 | 458,373 |
| $\pi^+$ | 211 | 571,241 |
| Proton | 2212 | 559,627 |
| **Total** |  | **1,589,241** |

All particles sharing `(source_file_id, event_id)` remain in the same split,
which prevents correlated particles from one generated event leaking between
training and evaluation.

| Split | Rows |
|---|---:|
| Train | 1,270,698 |
| Validation | 159,558 |
| Test | 158,985 |

The portable configs expect this layout when commands are run from the
repository root:

```text
QuantumMC_Simulations/
├── QuantumMC_Forward_Foundation_Model/
└── phase-space_parquet-Aug17-26/
    └── particle_responses/*.parquet
```

## Held-out results

The complete selected-population experiment used one NVIDIA H100 GPU, an
eight-component MDN, a four-layer width-256 backbone, and **222,324 trainable
parameters**. Metrics below are evaluated on events excluded from training.

| Metric | Development seed | Full training |
|---|---:|---:|
| Residual negative log likelihood | -4.0789 | **-4.7581** |
| Reconstructed-PID cross entropy | 1.2794 | **1.1774** |
| Reconstructed-PID top-1 accuracy | 66.62% | **67.22%** |
| Maximum PID marginal discrepancy | 0.692% | **0.426%** |
| Physical sampled $(p,\theta)$ fraction | 95.67% | **97.13%** |
| Test evaluation throughput | — | **112,434 examples/s** |

Negative log likelihood evaluates the entire predicted residual distribution;
lower is better. PID marginal discrepancy compares the generated fractions of
each reconstructed PID class with full simulation. The physical sampled
fraction is the fraction of draws with valid momentum and polar angle.

### Residual-distribution closure

Blue curves are held-out full simulation and orange curves are stochastic
samples from the learned mixture. Each row is a generated species and each
column is one residual. Agreement in the core and tails tests more than a mean
prediction would.

![Held-out residual distributions from full simulation and model samples](docs/figures/full_training_residual_closure.png)

### Resolution closure

Each heatmap cell is

$$
\text{width ratio}
=\frac{\operatorname{Std}(\Delta)_{\mathrm{model}}}
{\operatorname{Std}(\Delta)_{\mathrm{full\ simulation}}}.
$$

The ideal value is one; aggregate ratios span 0.940–1.065.

![Residual-width closure heatmap](docs/figures/full_training_width_closure.png)

Aggregate agreement does not guarantee uniform conditional fidelity. The
largest remaining discrepancies occur for $\pi^+$ momentum response at the
lowest generated polar angles, proton angular response at the lowest generated
polar angles, and $\pi^-$ azimuth response at low generated momentum. Detailed
aggregate and binned metrics are versioned under `runs/`.

## Installation

```bash
git clone https://github.com/HHTseng/QuantumMC_Forward_Foundation_Model.git
cd QuantumMC_Forward_Foundation_Model
conda create -n QuantumMC python=3.12 -y
conda activate QuantumMC
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Python 3.11–3.13 is supported. DuckDB is used instead of PyArrow to avoid a
page-index metadata issue in the source Parquet files.

## Train the model

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
Checkpoints are reproducible binary artifacts and are intentionally ignored by
Git.

## Sample detector response

Given a CSV containing `gen_pid`, `gen_p`, `gen_theta`, and `gen_phi` in
GeV/radians:

```bash
python sample.py \
  --checkpoint runs/fd_response_seed/model.pt \
  --input example_generated_hadrons.csv \
  --output sampled_fd_response.csv
```

## Repository map

```text
configs/                 Development and full-training configurations
forwardfm_step1/         Data contract, MDN, training, evaluation, reporting
tests/                   Scaling, leakage, likelihood, and sampling tests
docs/figures/            Process diagrams and held-out result figures
runs/                    Versioned metrics, plots, audits, and model cards
train.py                 Training entry point
sample.py                Conditional stochastic inference entry point
PHYSICS_TO_CODE.md       Physics-equation to implementation mapping
PSEUDOCODE.md            Staged modeling plan
```

## Scope, limitations, and next milestones

- The current targets are raw residuals because versioned energy-loss and
  swum-back-$\phi$ corrected targets were not available.
- The model is conditional on a triggered, selected, reconstructed FD hadron;
  it is not yet a trigger or reconstruction-efficiency model.
- Event correlations, electron response, CD response, detector-condition
  tokens, and analysis-level physics closure are not yet implemented.
- The checkpoint learns from its simulation/reconstruction teacher. Agreement
  with held-out simulation is not evidence of agreement with real detector
  data.
- Physics release requires predefined closure gates and event-level validation
  of invariant masses, missing quantities, and target analysis observables.

The next foundation-model stages add $P(T\mid x_e)$, then
$P(C\mid x,T)$, followed by event-level generation and analysis-observable
closure.
