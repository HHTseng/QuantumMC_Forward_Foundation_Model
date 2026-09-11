# QuantumMC Forward Foundation Model

<code>experiment/beta-pid-weight-completion</code> isolates the effects of
$\beta_{\rm gen}$, $\Delta\beta$, and $\lambda_{\rm PID}$ on the conditional
CLAS12 Forward Detector (FD) hadron response.

## 1. Mathematical object

Let

$$
\mathcal S:=\{-211,211,2212\}=\{\pi^-,\pi^+,p\}
$$

be the generated-species set. Let $\mathcal V_{\rm train}$ be the sorted
training PID vocabulary and

$$
\mathcal R:=\mathcal V_{\rm train}\cup\{{\rm OTHER}\}.
$$

Define

$$
\Sigma_{n-1}:=
\left\lbrace u\in[0,1]^n:\sum_{j=1}^{n}u_j=1\right\rbrace
$$

the probability simplex. The β-informed network is the parameter map

$$
F_\vartheta:
\mathbb R^5\times\mathcal S
\longrightarrow
\Sigma_{K-1}\times(\mathbb R^4)^K
\times(\mathbb R_{>0}^4)^K\times\Sigma_{C-1},
$$

where $K$ is the Gaussian-component count and $C:=|\mathcal R|$. Its output is

$$
F_\vartheta(z,s)=
\left(\pi,\{\mu_k,\sigma_k\}_{k=1}^{K},\rho\right),
$$

with mixture weights $\pi$, component parameters $(\mu_k,\sigma_k)$, and PID
probabilities $\rho$. Equivalently,

$$
q_\vartheta:
\mathbb R^5\times\mathcal S
\longrightarrow
\mathcal P(\mathbb R^4\times\mathcal R),
$$

where $\mathcal P(A)$ is the set of probability measures on $A$, and

$$
q_\vartheta(\delta,r\mid z,s)=
\left[
\sum_{k=1}^{K}\pi_k(z,s)
\prod_{j=1}^{4}
\mathcal N\!\left(\delta_j;\mu_{kj}(z,s),\sigma_{kj}^2(z,s)\right)
\right]\rho_r(z,s).
$$

Thus the output is a distribution, not one reconstructed particle.

## 2. Physics coordinates

Define the generated-particle space

$$
\mathcal X:=
\mathbb R_{>0}\times[0,\pi]\times
(\mathbb R/2\pi\mathbb Z)\times\mathcal S,
$$

$$
x=(p_{\rm gen},\theta_{\rm gen},\phi_{\rm gen},s_{\rm gen})\in\mathcal X.
$$

Momentum $p$ is in GeV; angles are in radians. For species $s$ with mass $m_s$,

$$
\beta_{\rm gen}(p,s):=
\frac{p}{\sqrt{p^2+m_s^2}}.
$$

| $s$ | PDG | $m_s$ [GeV] |
|---|---:|---:|
| $\pi^-$ | -211 | 0.13957039 |
| $\pi^+$ | 211 | 0.13957039 |
| $p$ | 2212 | 0.93827208816 |

The 5-feature map is

$$
\Phi_\beta:\mathcal X\longrightarrow\mathbb R^5\times\mathcal S,
$$

$$
\Phi_\beta(x)=
\left(
\left[
\log(1+p_{\rm gen}),
\theta_{\rm gen},
\sin\phi_{\rm gen},
\cos\phi_{\rm gen},
\beta_{\rm gen}
\right],
s_{\rm gen}
\right).
$$

Numerically, $p_{\rm gen}$ is expressed in GeV before <code>log1p</code>.
The $(\sin\phi,\cos\phi)$ pair represents
$\mathbb R/2\pi\mathbb Z$ without a cut at $\pm\pi$.

Only the $\mathbb R^5$ component is standardized:

$$
z_j:=\frac{\Phi_{\beta,j}-\mu_j^{\rm train}}{\sigma_j^{\rm train}},
\qquad j=1,\ldots,5.
$$

Species remains categorical. The input is therefore

$$
(z,s)\in\mathbb R^5\times\mathcal S,
$$

not $\mathbb R^4$, and not one $\mathbb R^6$ vector.

## 3. Response variables

For a matched reconstructed particle,

$$
\Delta p:=p_{\rm rec}-p_{\rm gen},
\qquad
\Delta\theta:=\theta_{\rm rec}-\theta_{\rm gen},
$$

$$
\Delta\phi:=
\mathrm{wrap}_{[-\pi,\pi)}
(\phi_{\rm rec}-\phi_{\rm gen}),
\qquad
\Delta\beta:=\beta_{\rm rec}-\beta_{\rm gen}.
$$

Define

$$
\Delta:=(\Delta p,\Delta\theta,\Delta\phi,\Delta\beta)\in\mathbb R^4.
$$

Training uses $\delta:=Z_\Delta(\Delta)\in\mathbb R^4$ and
$r:=s_{\rm rec}\in\mathcal R$. PID mismatches are labels, not rejected rows.

A draw $(\widehat\Delta,\widehat s_{\rm rec})$ is mapped back by

$$
\widehat p_{\rm rec}=p_{\rm gen}+\widehat{\Delta p},
\qquad
\widehat\theta_{\rm rec}=\theta_{\rm gen}+\widehat{\Delta\theta},
$$

$$
\widehat\phi_{\rm rec}=\mathrm{wrap}_{[-\pi,\pi)}
(\phi_{\rm gen}+\widehat{\Delta\phi}),
\qquad
\widehat\beta_{\rm rec}=\beta_{\rm gen}+\widehat{\Delta\beta}.
$$

## 4. Conditional scope

The fitted law is

$$
P(\Delta,s_{\rm rec}\mid x,T=1,C={\rm FD},F=1),
$$

where $T=1$ means valid trigger electron, $C={\rm FD}$ means matched FD
reconstruction, and $F=1$ means the stated fiducial/quality selection.

The selected set satisfies

$$
s_{\rm gen}\in\mathcal S,\quad
\theta_{\rm rec}<33^\circ,\quad -5.5<z_{\rm gen}<-0.5\ {\rm cm},
$$

$$
|\Delta p|\leq10\ {\rm GeV},\quad
0<\beta_{\rm rec}\leq1.2,
$$

plus reciprocal matching, finite residuals, nonzero reconstructed PID, and
<code>usable_for_hadron_response_training</code>.

The split unit is the event key $E$, represented in the data by
`(source_file_id, event_id)`. Event-disjointness means

$$
E_a\cap E_b=\varnothing
\qquad
(a\neq b;\ a,b\in\{{\rm train},{\rm val},{\rm test}\}).
$$

The intended larger factorization is

$$
P(Y\mid X)=
P(T\mid x_e)
\prod_i P(C_i\mid x_i,T)
P(\Delta_i,s_{{\rm rec},i}\mid x_i,T,C_i).
$$

This branch learns only the last factor for $C_i={\rm FD}$.

![Simulation chain and learned conditional-response shortcut](docs/figures/real_simulation_forward_model.png)

## 5. Objective

For $\lambda_{\rm PID}>0$,

$$
\mathcal L(\vartheta)=-\mathbb E_{\rm train}
\log q_\vartheta(\delta\mid z,s)
{}-\lambda_{\rm PID}\,
\mathbb E_{\rm train}
\log q_\vartheta(r\mid z,s).
$$

The recommended configuration is

$$
\dim z=5,\qquad
\dim\delta=4,\qquad
\lambda_{\rm PID}=1.
$$

Its checkpoint minimizes validation PID cross-entropy. Test data are used only
after selection.

## 6. Closure

For generated species $s$, momentum bin $b$, and reconstructed class $r$,

$$
P_{\rm CJ}(r\mid s,b)
:=
\frac{1}{N_{s,b}}
\sum_{i\in(s,b)}\mathbf 1(r_i=r),
$$

$$
P_{\rm FM}(r\mid s,b)
:=
\frac{1}{N_{s,b}}
\sum_{i\in(s,b)}\rho_r(z_i,s_i).
$$

This compares stochastic PID response, not top-1 accuracy. Define

$$
{\rm MAE}_s
:=
\frac{1}{|\mathcal B_s|}
\sum_{b\in\mathcal B_s}
\left|
P_{\rm FM}(s\mid s,b)-P_{\rm CJ}(s\mid s,b)
\right|,
$$

$$
{\rm TV}_{s,b}
:=
\frac12\sum_{r\in\mathcal R}
\left|
P_{\rm FM}(r\mid s,b)-P_{\rm CJ}(r\mid s,b)
\right|.
$$

Both vanish at exact PID closure.

## 7. Results

The matched ablation varies

$$
I_\beta:=\mathbf 1(\beta_{\rm gen}\ {\rm input}),\qquad
T_\beta:=\mathbf 1(\Delta\beta\ {\rm target}),\qquad
\lambda_{\rm PID}.
$$

| Model | $(I_\beta,T_\beta,\lambda_{\rm PID})$ | $(\dim z,\dim\delta)$ | Macro TV | Correct-ID MAE |
|---|---:|---:|---:|---:|
| A02 | $(0,0,0.2)$ | $(4,3)$ | $0.03440\pm0.00541$ | $0.01851\pm0.00405$ |
| B02 | $(1,0,0.2)$ | $(5,3)$ | $0.03413\pm0.00445$ | $0.01813\pm0.00279$ |
| C02 | $(0,1,0.2)$ | $(4,4)$ | $0.03509\pm0.00682$ | $0.01880\pm0.00307$ |
| D02 | $(1,1,0.2)$ | $(5,4)$ | $0.03593\pm0.00738$ | $0.01941\pm0.00630$ |
| A10 | $(0,0,1)$ | $(4,3)$ | $0.02489\pm0.00132$ | $0.01294\pm0.00114$ |
| B10 | $(1,0,1)$ | $(5,3)$ | $0.02492\pm0.00190$ | $0.01283\pm0.00224$ |
| D10 | $(1,1,1)$ | $(5,4)$ | **$0.02418\pm0.00160$** | **$0.01202\pm0.00102$** |

For a lower-is-better metric $M={\rm TV}$, define the paired improvement
$d=M_{\rm control}-M_{\rm treatment}$. The 4 sequential contrasts are

| Contrast | $\lambda_{\rm PID}$ | $\bar d$ | 95% CI |
|---|---:|---:|---:|
| $A\to B$: add $\beta_{\rm gen}$ | $0.2$ | $0.00027$ | $[-0.00179,0.00234]$ |
| $B\to D$: add $\Delta\beta$ | $0.2$ | $-0.00180$ | $[-0.00639,0.00278]$ |
| $A\to B$: add $\beta_{\rm gen}$ | $1$ | $-0.00004$ | $[-0.00073,0.00066]$ |
| $B\to D$: add $\Delta\beta$ | $1$ | $0.00074$ | $[-0.00120,0.00268]$ |

All 4 intervals contain $0$: neither β coordinate has a seed-stable PID effect
under this matched protocol. In contrast, increasing
$\lambda_{\rm PID}:0.2\to1$ gives

$$
\bar d_A=0.00952,\qquad
\bar d_B=0.00921,\qquad
\bar d_D=0.01175,
$$

with positive 95% intervals; favorable pairs are $10/10$, $9/10$, and
$10/10$. Thus the robust improvement is associated with PID-loss weighting,
not with adding $\beta_{\rm gen}$ or $\Delta\beta$.

![Correct-PID closure summary](runs/gpu_beta_gen_factorial/summary/pid_closure_mae_comparison_our_10seed.png)

![Correct-PID closure versus generated momentum](runs/gpu_beta_gen_factorial/summary/pid_closure_beta_comparison_our_10seed.png)

For the 3 models that learn $\Delta\beta$,

$$
W_1(C02)=0.00475,\qquad
W_1(D02)=0.00499,\qquad
W_1(D10)=0.00534.
$$

![Continuous β-response closure](runs/gpu_beta_gen_factorial/summary/beta_response_vs_gen_p_factorial.png)

Full tables:
[<code>runs/gpu_beta_gen_factorial/summary/</code>](runs/gpu_beta_gen_factorial/summary/).

## 8. Minimal use

Train D10:

```bash
python train.py \
  --config configs/gpu_beta_factorial_D_beta_input_target_pid1.yaml \
  --device cuda:0 \
  --run-dir runs/beta_informed_pid1
```

Sample from columns <code>gen_pid,gen_p,gen_theta,gen_phi</code>:

```bash
python sample.py \
  --checkpoint runs/beta_informed_pid1/model.pt \
  --input example_generated_hadrons.csv \
  --output artifacts/example_beta_informed_response.csv \
  --device cuda:0
```

The checkpoint determines feature/target order, scaling, masses, and PID
vocabulary.

## 9. Limitations

1. $T=1$ and $C={\rm FD}$ are preselected; no efficiency is learned.
2. $\mathcal S=\{\pi^-,\pi^+,p\}$; generated $e^-$ and $K^-$ are absent.
3. Particles are independent rows; event correlations are absent.
4. $\Delta$ and $s_{\rm rec}$ are conditionally factorized.
5. Each Gaussian component has diagonal covariance.
6. Simulation closure does not imply experimental-data closure.

## Appendix A. Network constants

These are implementation choices, not the mathematical definition:

| Quantity | Value |
|---|---:|
| $K$ | 8 |
| Hidden width | 256 |
| Hidden layers | 4 |
| Activation | SiLU |
| Normalization | LayerNorm |
| Dropout | 0.03 |
| Species-embedding dimension | 16 |
| Batch size | 8192 |
| Epoch budget | 30 |

For batch size $B$:

| Output | Shape |
|---|---|
| <code>mixture_logits</code> | $B\times K$ |
| <code>means</code> | $B\times K\times4$ |
| <code>log_scales</code> | $B\times K\times4$ |
| <code>pid_logits</code> | $B\times C$ |

Saved checkpoints:

- <code>model.pt</code>: configured primary validation metric;
- <code>model_min_validation_pid_cross_entropy.pt</code>: minimum validation PID loss;
- <code>model_min_validation_total_loss.pt</code>: minimum validation joint loss.

Each stores architecture, feature/target order, scalers, masses, vocabularies,
selection SQL, data fingerprint, seeds, and checkpoint provenance.

## Appendix B. Experiment provenance

Section 7 summarizes $10$ paired seeds,

$$
20260822,\ldots,20260831,
$$

with one teacher population and event split:

| Split | $\pi^-$ | $\pi^+$ | $p$ | Total |
|---|---:|---:|---:|---:|
| Train | 364,925 | 455,246 | 446,432 | 1,266,603 |
| Validation | 45,893 | 57,048 | 56,131 | 159,072 |
| Test | 45,817 | 56,774 | 55,891 | 158,482 |

All 70 runs realize 30 epochs and select checkpoints by validation PID
cross-entropy. In each paired 4/5-input comparison, the added β column starts
with zero first-layer weight; both models therefore represent the same initial
function. Before aggregation, <code>provenance.csv</code> verifies a common
dataset fingerprint, selection, event split, row counts, PID vocabulary,
momentum bins, optimizer budget, and checkpoint rule.

## Appendix C. Reproduction

```bash
git clone https://github.com/HHTseng/QuantumMC_Forward_Foundation_Model.git
cd QuantumMC_Forward_Foundation_Model
git switch experiment/beta-pid-weight-completion

conda create -n QuantumMC python=3.11 -y
conda activate QuantumMC
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Default data:

```text
../phase-space_parquet-Aug17-26/particle_responses/*.parquet
```

2-epoch CPU check:

```bash
python train.py \
  --config configs/gpu_beta_factorial_D_beta_input_target_pid1.yaml \
  --smoke \
  --device cpu \
  --run-dir runs/smoke_beta_informed
```

Full matched experiment:

```bash
python experiments/run_beta_gen_factorial.py \
  --device cuda:0 \
  --parquet-glob '/path/to/particle_responses/*.parquet'

python experiments/analyze_beta_gen_factorial.py
```

Per-run checkpoint:

```text
runs/gpu_beta_gen_factorial/seed_<seed>/<condition>/model.pt
```

| Path | Definition |
|---|---|
| <code>forwardfm_step1/data.py</code> | $\Phi_\beta$, $\Delta$, selection, split, scaling |
| <code>forwardfm_step1/model.py</code> | $F_\vartheta$ |
| <code>forwardfm_step1/training.py</code> | $\mathcal L$, checkpoints |
| <code>forwardfm_step1/evaluation.py</code> | closure |
| <code>train.py</code> | training/evaluation |
| <code>sample.py</code> | sampling |
| <code>experiments/</code> | paired runs, aggregate analysis |
