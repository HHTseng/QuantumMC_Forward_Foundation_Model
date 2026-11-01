# Exclusive rho transfer protocol

1. **Data contract.** A native row is one generated particle. The event key is
   `(file_id,event)`. Reconstruction means mutual MC-to-REC-to-same-MC association.
   The electron trigger is the eventwise existence of a generated electron
   associated with the first negative-status REC row in FD. Reconstructed PID
   need not equal 11. Preserve all generated rows, including failures.
2. **Coordinates.** Convert degrees to radians and recompute REC−GEN residuals.
   Use Alex's pre-XGBoost backups for the raw response. The September export
   lacks raw backups; its native momentum must not be described as uncorrected.
   `detector=0` includes matched FT/other outcomes as well as failures.
3. **Inputs.** Select native whole events with `hash(file_id,event)%10=0`.
   Retain all four particles. The August control is the existing held-out
   event partition. Save exact input SHA-256 hashes. Copies on both platforms
   must match before execution. Neither native export contains generator weights.
4. **Response.** Freeze the 3 existing 5-input split/asinh checkpoints. Use their
   training scalers, feature vocabulary and selected FD/quality population.
   Compare raw and corrected rho on identical particles selected using raw
   momentum. No fitting or checkpoint selection uses rho. The August comparison
   also changes skim/trigger conventions; it is not a causal generator ablation.
5. **Efficiency.** Reuse the categorical machinery from the hadron-efficiency
   branch. Fit 3 species gates using September phase-space data only, with
   300,000 train, 50,000 validation and up to 100,000 test events/species.
   Fix event splits independently of network seeds. Fit scalers on training
   examples only. Do not cut the denominator on reconstruction, PID, beta,
   residuals or FD membership. Targets are failure/FD/FT/CD/other.
6. **Replication.** Use 3 network seeds per platform, 30 epochs maximum,
   phase-validation early stopping and identical inputs/hyperparameters.
   Record hardware, package versions, source hashes, selected event hashes,
   checkpoints and logs. Hardware/software differences preclude bitwise
   training equivalence. Frozen-response inference uses a shared CPU RNG.
7. **Metrics.** Conditional PID closure uses mean softmax probabilities,
   not argmax accuracy. Report correct-ID MAE and full-vocabulary total
   variation in fixed 1-GeV bins with at least 200 particles. Report both
   full-sample and common-bin comparisons. Momentum response uses central
   Gaussian fits, quantiles and explicit tails. Efficiency uses probability
   calibration, Brier score, and momentum/angular closure. Count-weighted bin
   MAE is not generator-weighted physics. Report coordinate-box support
   diagnostics; a bounding box does not prove learned-density coverage.
8. **Physics plots.** Show generated pion-pair mass and momentum conservation,
   then raw/corrected pion-pair and missing-mass distributions for the same
   fully matched FD events with truth species mass assignments. These are
   MC correction diagnostics, not an end-to-end FM event prediction.
9. **Scope.** Without HEPGen weights, all figures concern unweighted MC counts.
   Weighted physical yields/distributions need LUND weights and an explicit
   source-event mapping. Independent per-particle gates and selected response
   heads do not yet form a validated, correlated exclusive-event simulator.

The current work starts at `main` (`56c5e19`) and incorporates gate machinery
from `feature/trigger-conditioned-hadron-efficiency` (`4674686`). Existing
training/sampling paths remain available. New data, fitted gate weights and
event predictions remain outside Git publication; aggregate results and
reproduction commands are versioned.

## Appendix: coordinates, models and scores

Response seeds: `20260822,20260823,20260824`. Gate seeds:
`20261101,20261102,20261103`; fixed gate event-split seed: `20261101`.
Response uses 4 width-256 SiLU/LayerNorm layers, dropout 0.03, species embedding
16, and 8 components per mixture head. A 1D momentum head uses

$$
u_p=\mathrm{asinh}(\Delta p/a),\qquad a=0.020\,\mathrm{GeV},
\qquad \Delta p=a\sinh u_p.
$$

A separate joint 3D head describes angular/beta residuals. PID is categorical.
The 3 heads are conditionally independent; the angular/beta coordinates are
jointly modeled within their mixture. Gates use 4 width-256 layers, dropout
0.03, AdamW with learning rate 0.001 and weight decay 0.00001, batch size 8192,
gradient norm cap 5, and patience 8. No gate class weighting/resampling is used.

The saved response quality selection requires reconstructed polar angle below
33 degrees, generated vertex between -5.5 and -0.5 cm, reciprocal FD matching,
nonzero reconstructed PID, beta greater than the -99 sentinel, finite residuals,
and absolute raw momentum residual at most 10 GeV. Gate denominators do not
use these requirements. Physics plots start with 100,000 complete events per
source; their reconstructed comparison additionally requires the electron
trigger, all 4 mutually matched FD particles, finite positive raw/corrected
momenta and per-track absolute raw residual at most 10 GeV. There are 5,934
common rho events. Truth-species masses are used without a reconstructed-PID
agreement cut.

For generated species $s$, reconstructed class $r$, momentum bin $b$ and its
population $N_{sb}$, conditional PID closure is

$$
c_{sbr}=\frac{1}{N_{sb}}\sum_{i\in(s,b)}\mathbf1(r_i=r),\qquad
f_{sbr}=\frac{1}{N_{sb}}\sum_{i\in(s,b)}P_\vartheta(r\mid x_i,s),
$$

$$
\mathrm{MAE}_s=\frac{100}{|\mathcal B_s|}
\sum_{b\in\mathcal B_s}|f_{sbs}-c_{sbs}|,\qquad
\mathrm{TV}_{sb}=\frac12\sum_{r\in\mathcal R}|f_{sbr}-c_{sbr}|.
$$

Eligible bins contain at least 200 particles. Common-bin scores intersect
eligibility across sources; width comparisons separately intersect successful
Gaussian fits. Equal-bin MAE is in percentage points; reported TV averages
use bin populations as weights. Reconstructed species is compared with its
corresponding generated PDG code, retaining all migrations and `OTHER`.

For 20 equal probability bins $I_l$, observed efficiency $e_l$, mean predicted
probability $\hat e_l$, and bin population $n_l$, calibration error in pp is

$$
\mathrm{ECE}=\frac{100}{N}\sum_l n_l|e_l-\hat e_l|.
$$

Wilson intervals use 68% binomial coverage. Seed bands are sample standard
deviations on identical teacher events. These uncertainties exclude detector
systematics and omitted generator weights. Gaussian-core fitting follows the
existing 2-stage 1-MeV-histogram procedure in `evaluation_split_dp.py`.

The additional `identified_fd` sensitivity repeats one frozen response model
per platform with reconstructed trigger PID 11 required. It changes the
selected population and is not an event-matched intervention.

Identical detector settings do not imply an exactly universal single-particle
gate. Let $G=(g_h,G_{-h})$ be the generated event, with hadron coordinates
$g_h$ and remaining generated particles $G_{-h}$. For dataset $d$ and common
detector/reconstruction configuration $\kappa$, the unweighted MC law obeys

$$
P_d(C_h=c\mid g_h,T_e=1)=
\int K_\kappa(C_h=c\mid G,T_e=1)\,
P_d(dG_{-h}\mid g_h,T_e=1).
$$

The marginalized distribution changes between independent phase-space
throwing and exclusive production. Single-particle inputs omit this event
context. Coverage and omitted correlations are therefore distinct possible
sources of transfer error; the present experiment does not isolate them.
