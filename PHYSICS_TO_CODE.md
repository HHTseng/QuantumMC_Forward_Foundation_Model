# Physics-to-code map

The full planned per-event surrogate begins from generated truth

\[
X=(x_e,x_1,\ldots,x_n),\qquad x_i=(p_i,\theta_i,\phi_i,s_i),
\]

and factorizes the detector/reconstruction response as

\[
P(Y\mid X)=P(T\mid x_e)\prod_i P(C_i\mid x_i,T)
P(\Delta_i,\widehat s_i\mid x_i,T,C_i).
\]

Here `T` is the event trigger-electron proxy, `C` is the reconstruction region
or failure outcome, `Delta` is reconstructed-minus-generated kinematics, and
`s_hat` is reconstructed PID.

## What step1 implements

The FD-cuts loader has already conditioned on `T=1` and `C=FD`. Step1 therefore
learns only

\[
P(\Delta p,\Delta\theta,\Delta\phi,\widehat s
\mid p,\theta,\phi,s,T=1,C=\mathrm{FD}).
\]

It cannot supply trigger or reconstruction efficiencies because those
denominators were removed by the FD selection.

| Physics operation | Equation | Code |
|---|---|---|
| Truth state | \(x=(p,\theta,\phi,s)\) | `data._feature_matrix`, species embedding in `model.ConditionalMDN` |
| Fiducial population | \(C=\mathrm{FD},\theta_{rec}<33^\circ,-5.5<z_{gen}<-0.5\) cm | `data.fiducial_sql` |
| Residual target | \(\Delta q=q_{rec}-q_{gen}\) | Parquet `delta_p`, `delta_theta`, `delta_phi` selected in `data._load_frame` |
| Periodic angle | \(\Delta\phi=\mathrm{wrap}(\phi_{rec}-\phi_{gen})\) | `sample.wrap_phi`; generated phi encoded by sine/cosine |
| Event-disjoint split | \(E=(source\_file\_id,event\_id)\) | `data.split_predicate`, `data.assert_event_disjoint` |
| Residual density | \(p(\Delta\mid x)=\sum_k\pi_k\prod_j\mathcal N(\Delta_j;\mu_{kj},\sigma_{kj}^2)\) | `model.ConditionalMDN` |
| Residual likelihood | \(-\mathbb E[\log p_\theta(\Delta\mid x)]\) | `model.mixture_nll` |
| PID response | \(P(\widehat s\mid x)=\mathrm{softmax}(a(x))\) | `model.pid_head` and cross entropy in `training.run_epoch` |
| Joint loss | \(L=L_R+\lambda_{PID}L_{PID}\) | `training.run_epoch` |
| Stochastic draw | \(k\sim Cat(\pi),\epsilon\sim N(0,I),\Delta=\mu_k+\sigma_k\epsilon\) | `model.sample_standardized_residuals` |
| Reconstructed sample | \(p_{rec}=p_{gen}+\Delta p\), etc. | `sample.py` |
| Bias/resolution closure | compare \(E[\Delta]\), \(Std[\Delta]\), quantiles | `evaluation.closure_rows` |
| Conditional closure | compare response in \((p,\theta,\phi,s)\) bins | `evaluation.kinematic_closure_rows` |
| Joint closure | compare residual correlation matrices | `evaluation.joint_and_physical_metrics` |

## Remaining factors

The next implementation must return to the all-event loader and add:

1. \(P(T=1\mid x_e)\), using every generated electron row;
2. \(P(C\mid x,T=1)\), using triggered-event hadrons including unreconstructed,
   FT, FD, and CD outcomes;
3. a sampler that executes `trigger -> outcome -> conditional response`;
4. corrected response targets only after the energy-loss and swum-back-phi
   algorithm, sign, coordinates, and version are frozen.

