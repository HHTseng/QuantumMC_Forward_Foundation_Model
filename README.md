# QuantumMC Forward Foundation Model

This branch, `Kyungseon_beta_response`, is a physics-informed baseline for the
conditional CLAS12 Forward Detector (FD) response. The primary model uses
**five continuous generated-particle inputs**, including relativistic truth
velocity $\beta_{\rm gen}$, plus a categorical generated-species input. It
learns four continuous reconstruction residuals and the stochastic
reconstructed-PID response.

The model is trained only on generated hadrons from events with a valid trigger
electron that also have a matched, selected FD reconstruction. It models
response *conditional on reconstruction*; it does not model trigger or
reconstruction efficiency.

## Physics scope

For one generated hadron, define

$$
x=(p_{\rm gen},\theta_{\rm gen},\phi_{\rm gen},s_{\rm gen}),
\qquad
s_{\rm gen}\in\{-211,211,2212\}=\{\pi^-,\pi^+,p\}.
$$

Here $p_{\rm gen}$ is momentum in GeV, and $\theta_{\rm gen}$ and
$\phi_{\rm gen}$ are angles in radians. The implemented conditional model
is factorized into continuous-response and PID heads:

$$
q_\vartheta(\Delta,s_{\rm rec}\mid x,T=1,C={\rm FD},F=1)
=q_\vartheta(\Delta\mid x)\,q_\vartheta(s_{\rm rec}\mid x).
$$

The fixed conditions $T=1$, $C={\rm FD}$, and $F=1$ are suppressed on the
right-hand side for compactness.

where $T$ is the trigger outcome, $C$ is the reconstruction region, $F$ is the
current fiducial and quality selection, and $s_{\rm rec}$ is the COATJAVA
reconstructed PID. The continuous response is

$$
\Delta=(\Delta p,\Delta\theta,\Delta\phi,\Delta\beta),
$$

with

$$
\Delta p=p_{\rm rec}-p_{\rm gen},\qquad
\Delta\theta=\theta_{\rm rec}-\theta_{\rm gen},
$$

$$
\Delta\phi=
\mathrm{wrap}_{[-\pi,\pi)}
(\phi_{\rm rec}-\phi_{\rm gen}),
$$

and

$$
\beta_{\rm gen}
=\frac{p_{\rm gen}}
{\sqrt{p_{\rm gen}^{2}+m_s^{2}}},
\qquad
\Delta\beta=\beta_{\rm rec}-\beta_{\rm gen}.
$$

The generated-species mass $m_s$ is:

| Species | PDG code | $m_s$ [GeV] |
|---|---:|---:|
| $\pi^-$ | -211 | 0.13957039 |
| $\pi^+$ | 211 | 0.13957039 |
| proton | 2212 | 0.93827208816 |

This is the final conditional factor in a future event-level decomposition,

$$
P(Y\mid X)=P(T\mid x_e)
\prod_i P(C_i\mid x_i,T)
P(\Delta_i,s_{{\rm rec},i}\mid x_i,T,C_i).
$$

The trigger and reconstruction-outcome factors are not trained in this branch.

![Full simulation and learned conditional-response shortcut](docs/figures/real_simulation_forward_model.png)

## Inputs: five continuous features plus species

The primary physics-informed configuration maps each truth particle to

$$
\Phi_\beta(x)=
\left[
\log(1+p_{\rm gen}),
\theta_{\rm gen},
\sin\phi_{\rm gen},
\cos\phi_{\rm gen},
\beta_{\rm gen}
\right]\in\mathbb{R}^{5}.
$$

For minibatch size $B$, the model receives

$$
z_x\in\mathbb{R}^{B\times5},
\qquad
s_{\rm index}\in\{0,1,2\}^{B}.
$$

The five continuous columns are standardized with training-split statistics,

$$
z_{x,j}=\frac{\Phi_{\beta,j}-\mu_j^{\rm train}}
{\sigma_j^{\rm train}},
$$

while `species_index` is passed separately through a learned 16-dimensional
embedding. Therefore the interface is **five continuous features plus one
categorical species input**; species is not a sixth continuous feature.

| Input | Purpose |
|---|---|
| `log1p_gen_p` | Compresses the generated-momentum range. |
| `gen_theta` | Retains the generated polar angle. |
| `sin_gen_phi`, `cos_gen_phi` | Represents periodic azimuth without a discontinuity at $\pm\pi$. |
| `beta_gen` | Supplies nonlinear relativistic mass dependence. |
| `species_index` | Selects a learned embedding for $\pi^-$, $\pi^+$, or proton. |

Although $\beta_{\rm gen}$ is determined by $(p_{\rm gen},s_{\rm gen})$ and
adds no new truth information, it exposes a useful physics coordinate directly
to the optimizer.

Four-input configurations remain only as controlled ablations. They
intentionally omit `beta_gen`; they are not the primary interface of this
branch.

## Labels

Each selected particle supplies two kinds of labels:

1. The standardized continuous target
   $\Delta=(\Delta p,\Delta\theta,\Delta\phi,\Delta\beta)\in\mathbb{R}^4$.
2. `rec_pid_index`, the categorical reconstructed-PID outcome. Misidentified
   particles are retained because PID migration is part of the response.

Thus the primary target tensors have domains

$$
z_\Delta\in\mathbb{R}^{B\times4},
\qquad
y_{\rm PID}\in\{0,\ldots,C-1\}^{B}.
$$

The reconstructed-PID vocabulary is discovered from the training split and an
`OTHER` class handles values absent from that vocabulary. The ordered
vocabulary, feature and target names, and both standardizers are saved in every
checkpoint; inference reads this metadata instead of hard-coding class
indices.

## Network and loss

The full configuration uses a shared multilayer perceptron with four 256-unit
SiLU/LayerNorm layers, dropout 0.03, a 16-dimensional species embedding, and
two heads.

The continuous head is an eight-component mixture-density network (MDN):

$$
q_\vartheta(z_\Delta\mid z_x,s)
=\sum_{k=1}^{8}\pi_k(z_x,s)
\prod_{j=1}^{4}
\mathcal N\!\left(z_{\Delta,j};\mu_{kj},\sigma_{kj}^{2}\right).
$$

Each Gaussian component has diagonal covariance. Shared mixture membership can
still represent dependence among the four residual coordinates. The PID head
predicts

$$
q_\vartheta(s_{\rm rec}=c\mid z_x,s)
=\mathrm{softmax}(\ell^{\rm PID})_c.
$$

Its returned tensors have shapes

| Tensor | Shape | Meaning |
|---|---|---|
| `mixture_logits` | $B\times8$ | Mixture-component logits. |
| `means` | $B\times8\times4$ | Component means for standardized residuals. |
| `log_scales` | $B\times8\times4$ | Component log standard deviations. |
| `pid_logits` | $B\times C$ | Reconstructed-PID class logits. |

Training minimizes

$$
\mathcal L
=-\mathbb E_{\rm train}\!\left[\log q_\vartheta(z_\Delta\mid z_x,s)\right]
-\lambda_{\rm PID}
\mathbb E_{\rm train}\!\left[\log q_\vartheta(s_{\rm rec}\mid z_x,s)\right].
$$

The recommended configuration uses $\lambda_{\rm PID}=1.0$. The primary
checkpoint is selected by validation PID cross-entropy; held-out test data are
never used for selection.

| Checkpoint | Selection rule |
|---|---|
| `model.pt` | Primary metric declared in the configuration. |
| `model_min_validation_pid_cross_entropy.pt` | Lowest validation PID cross-entropy. |
| `model_min_validation_total_loss.pt` | Lowest validation total loss. |

Checkpoints also contain architecture, feature and target names, species and
PID vocabularies, scalers, selection SQL, dataset fingerprint, seeds, masses,
and selection provenance. Model dimensions are reconstructed from this
metadata, so earlier four-input or three-target checkpoints remain loadable.

## Data selection and split

One row is one generated hadron matched to one reconstructed FD candidate, not
a complete event. The common factorial selection requires:

- generated $\pi^-$, $\pi^+$, or proton;
- `usable_for_hadron_response_training` and reconstructed region `FD`;
- $\theta_{\rm rec}<33^\circ$ and $-5.5<z_{\rm gen}<-0.5$ cm;
- reciprocal truth/reconstruction matching;
- finite residuals and $|\Delta p|\leq10$ GeV;
- nonzero reconstructed PID and no beta sentinel;
- $0<\beta_{\rm rec}\leq1.2$.

The physical event key `(source_file_id, event_id)`, not the particle row, is
assigned by a fixed seeded hash to approximately 80%/10%/10%
train/validation/test partitions. This keeps all particles from one event in
one partition.

All 50 models in the 10-seed factorial study used the same selected particles:

| Split | $\pi^-$ | $\pi^+$ | proton | Total |
|---|---:|---:|---:|---:|
| Train | 364,925 | 455,246 | 446,432 | 1,266,603 |
| Validation | 45,893 | 57,048 | 56,131 | 159,072 |
| Test | 45,817 | 56,774 | 55,891 | 158,482 |

Scalers and the reconstructed-PID vocabulary are fit on the training split
only.

## Closure metrics

PID closure compares conditional probabilities, not top-1 accuracy. For
generated species $s$, momentum bin $b$, and reconstructed class $c$,

$$
P_{\rm CJ}(c\mid s,b)
=\frac{1}{N_{s,b}}
\sum_{i\in(s,b)}\mathbf 1(y_i=c),
$$

while the Forward FM response is the mean softmax probability

$$
P_{\rm FM}(c\mid s,b)
=\frac{1}{N_{s,b}}
\sum_{i\in(s,b)}q_\vartheta(c\mid x_i).
$$

The correct-ID mean absolute error and full-row total variation are

$$
{\rm MAE}_s
=\frac{1}{|\mathcal B_s|}
\sum_{b\in\mathcal B_s}
\left|P_{\rm FM}(s\mid s,b)-P_{\rm CJ}(s\mid s,b)\right|,
$$

$$
{\rm TV}_{s,b}
=\frac{1}{2}\sum_c
\left|P_{\rm FM}(c\mid s,b)-P_{\rm CJ}(c\mid s,b)\right|.
$$

Both are lower-is-better. Continuous beta closure uses MDN samples and reports
one-dimensional Wasserstein distance plus binned means, widths, and quantiles
of $\beta_{\rm rec}=\beta_{\rm gen}+\Delta\beta$.

## Ten-seed factorial results

Five matched conditions were trained for 30 epochs with seeds 20260822 through
20260831. Every condition used the same beta-valid rows, event split, training
budget, and validation-PID checkpoint rule. Beta-input models were nested at
initialization inside their four-input controls by setting the new input
column's first-layer weights to zero.

| Condition | Inputs | Targets | $\lambda_{\rm PID}$ | Macro TV | Correct-ID MAE |
|---|---:|---:|---:|---:|---:|
| A: original control | 4 | 3 | 0.2 | 0.03440 $\pm$ 0.00541 | 0.01851 $\pm$ 0.00405 |
| B: $\beta_{\rm gen}$ input only | 5 | 3 | 0.2 | 0.03413 $\pm$ 0.00445 | 0.01813 $\pm$ 0.00279 |
| C: $\Delta\beta$ target only | 4 | 4 | 0.2 | 0.03509 $\pm$ 0.00682 | 0.01880 $\pm$ 0.00307 |
| D: input + target | 5 | 4 | 0.2 | 0.03593 $\pm$ 0.00738 | 0.01941 $\pm$ 0.00630 |
| D: input + target, stronger PID loss | **5** | **4** | **1.0** | **0.02418 $\pm$ 0.00160** | **0.01202 $\pm$ 0.00102** |

Values are mean $\pm$ sample standard deviation across 10 seeds. At
$\lambda_{\rm PID}=0.2$, adding $\beta_{\rm gen}$ was not a seed-stable PID
improvement. With $\Delta\beta$ present, its macro-TV improvement was
$-0.00085$ with 95% confidence interval $[-0.00490,0.00321]$.

Increasing $\lambda_{\rm PID}$ from 0.2 to 1.0 was robust. Macro TV improved
by $0.01175$ with 95% confidence interval $[0.00688,0.01662]$ in 10/10 paired
seeds (exact sign-flip $p=0.001953$). Correct-ID MAE improved from 1.85% to
1.19% for generated $\pi^+$ and from 2.46% to 1.19% for generated protons.

![Ten-seed correct-PID closure summary](runs/gpu_beta_gen_factorial/summary/pid_closure_mae_comparison_our_10seed.png)

![Momentum-dependent correct-PID closure](runs/gpu_beta_gen_factorial/summary/pid_closure_beta_comparison_our_10seed.png)

The beta-informed endpoints are close to the collaborator-supplied figure, but
the no-beta control is not. For $\pi^+$, supplied versus current results are
18.4% versus $2.06\pm0.65$% without beta and 1.3% versus
$1.19\pm0.25$% for the five-input model at $\lambda_{\rm PID}=1.0$. For
protons they are 23.3% versus $2.09\pm0.59$% and 1.5% versus
$1.19\pm0.28$%, respectively. The beta-informed endpoints differ by only
0.11--0.31 percentage points, whereas no-beta controls differ by 16.34--21.21
points. This cannot be attributed to $\beta_{\rm gen}$ until the exact
checkpoint, selected population, and binning used for the supplied no-beta
curve are matched.

The beta-target conditions give mean beta Wasserstein distances across species
of $0.00475\pm0.00065$ (C), $0.00499\pm0.00112$ (D, 0.2), and
$0.00534\pm0.00067$ (D, 1.0). A stronger PID weight improves PID closure but
does not improve this beta-distribution metric.

![Continuous beta-response closure](runs/gpu_beta_gen_factorial/summary/beta_response_vs_gen_p_factorial.png)

The complete report and machine-readable tables are in
[`runs/gpu_beta_gen_factorial/summary/`](runs/gpu_beta_gen_factorial/summary/).

## Reproduce training and evaluation

Use Python 3.11 or another version supported by the dependency ranges.

```bash
git clone https://github.com/HHTseng/QuantumMC_Forward_Foundation_Model.git
cd QuantumMC_Forward_Foundation_Model
git switch Kyungseon_beta_response

conda create -n QuantumMC python=3.11 -y
conda activate QuantumMC
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Full-scale configs expect Parquet files at
`../phase-space_parquet-Aug17-26/particle_responses/*.parquet`. Change that
path in a copied YAML file or pass `--parquet-glob` to the factorial runner.

Run a two-epoch smoke test of the recommended five-input, four-target model:

```bash
python train.py \
  --config configs/gpu_beta_factorial_D_beta_input_target_pid1.yaml \
  --smoke \
  --device cpu \
  --run-dir runs/smoke_beta_informed
```

Train one full recommended model on one GPU:

```bash
python train.py \
  --config configs/gpu_beta_factorial_D_beta_input_target_pid1.yaml \
  --device cuda:0 \
  --run-dir runs/beta_informed_pid1
```

Run all five matched conditions for the default 10 seeds, then rebuild the
tables and figures:

```bash
python experiments/run_beta_gen_factorial.py \
  --device cuda:0 \
  --parquet-glob '/path/to/particle_responses/*.parquet'

python experiments/analyze_beta_gen_factorial.py
```

The runner writes each model to
`runs/gpu_beta_gen_factorial/seed_<seed>/<condition>/model.pt`, skips a
completed condition unless `--force` is supplied, and locks each run
directory to prevent concurrent writers.

## Sample reconstructed-like particles

`sample.py` accepts a CSV with
`gen_pid,gen_p,gen_theta,gen_phi`. It constructs the checkpoint's exact
ordered features, including $\beta_{\rm gen}$ for the current model, then
samples the MDN and categorical PID head.

```bash
python sample.py \
  --checkpoint runs/beta_informed_pid1/model.pt \
  --input example_generated_hadrons.csv \
  --output artifacts/example_beta_informed_response.csv \
  --device cuda:0
```

For a four-target checkpoint, output includes `sampled_delta_p`,
`sampled_delta_theta`, `sampled_delta_phi`, `sampled_delta_beta`,
`sampled_rec_p`, `sampled_rec_theta`, `sampled_rec_phi`,
`sampled_rec_beta`, and `sampled_rec_pid`. The script writes physical-domain
flags but does not silently clip samples.

## Repository guide

| Path | Purpose |
|---|---|
| `forwardfm_step1/data.py` | Selection, event split, five-feature construction, $\beta_{\rm gen}$, $\Delta\beta$, scaling, and PID vocabulary. |
| `forwardfm_step1/model.py` | Conditional MDN, species embedding, and PID head. |
| `forwardfm_step1/training.py` | Likelihood training and validation-only checkpoint selection. |
| `forwardfm_step1/evaluation.py` | Residual, PID, and beta closure metrics and plots. |
| `train.py` | End-to-end training, evaluation, and checkpoint provenance. |
| `sample.py` | Checkpoint-driven stochastic inference. |
| `configs/gpu_beta_factorial_*.yaml` | Matched A/B/C/D definitions. |
| `experiments/run_beta_gen_factorial.py` | Locked 10-seed factorial runner. |
| `experiments/analyze_beta_gen_factorial.py` | Paired statistics, tables, and figures. |
| `runs/gpu_beta_gen_factorial/summary/` | Committed aggregate results for 50 models. |
| `tests/test_core.py` | Data, model, sampling, initialization, metric, and locking tests. |

## Current limitations

- The data already require a valid trigger electron and selected FD match, so
  this model cannot estimate trigger or reconstruction efficiency.
- Generated electrons and $K^-$ are not modeled species.
- Rows are modeled independently; event-level correlations and conservation
  laws are not enforced.
- The categorical PID head and continuous MDN share a backbone, but their
  output likelihoods are factorized conditional on generated inputs.
- MDN covariance is diagonal within each mixture component.
- The selection covers restricted FD phase space and a beta-valid teacher
  domain, not full detector acceptance.
- Closure against the training simulation tests emulation of that simulation;
  it does not establish agreement with experimental data.

These boundaries are intentional. A larger forward model should add explicit
trigger and reconstruction-outcome factors before using this conditional
response as a complete event simulator.
