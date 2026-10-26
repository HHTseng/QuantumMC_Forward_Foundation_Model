# Hadron reconstruction-efficiency closure

## Task and evidence

**Inferred next task:** add separate acceptance/reconstruction probabilities for
generated $\pi^-$, $\pi^+$, and protons, conditional on a valid electron-trigger
event. Compare predicted probabilities with full-MC reconstruction fractions,
as in the poster's electron probability-closure study. This is not a hadron
hardware-trigger study and is not correct-PID classification.

The local record
`Data_processing/Forward_FM_Data_Email_Summary.md` (2026-08-15, 11:43 AM)
explicitly summarizes Joo's distinction: the electron factor uses all generated
events; hadron factors use triggered events only. Counting untriggered hadrons
as reconstruction failures would include trigger loss twice. This is supporting
evidence from a saved correspondence summary, not the original text of the
latest remembered conversation. The 2026-08-17 revised skim retains the required
generated failures and detector-region labels.

Base: `feature/trigger-electron-efficiency-four-species`. Preserve its electron
checkpoint and conditional response model; add the missing hadron gate without
retraining either reference.

## Mapping and labels

Let $T\in\lbrace0,1\rbrace$ be the existing valid-trigger-electron label,
$s\in S=\lbrace-211,211,2212\rbrace$ the generated PDG species, and
$x_h=(p,\theta,\phi,v_z)$ the generated momentum, polar angle, azimuth, and
longitudinal vertex. Units are GeV, radians, and cm. Define

$$
\Phi(x_h)=(\log(1+p),\theta,\sin\phi,\cos\phi,v_z),
\qquad
z=Z_{\rm train}(\Phi(x_h)).
$$

$Z_{\rm train}$ is coordinate-wise standardization fitted on training rows
only. No reconstructed quantity enters $z$. Three independent species networks
implement

$$
q_\vartheta:\mathbb R^5\times S\longrightarrow\Delta^4,
\qquad
\Delta^4=\lbrace q\in[0,1]^5:\sum_{c=0}^{4}q_c=1\rbrace,
$$

$$
q_\vartheta(c\mid z,s)\simeq P(C_h=c\mid x_h,s,T=1),
\qquad
C_h\in\lbrace\mathrm{unreconstructed},\mathrm{FD},\mathrm{FT},
\mathrm{CD},\mathrm{other}\rbrace.
$$

`reconstructed` determines whether a truth-matched reconstructed object exists;
the retained detector-region information determines its category when present.
Unreconstructed rows remain in the denominator. Preserve and audit categories
with zero observed support; an unobserved category is not validated by training.

Two useful efficiencies follow without an additional classifier:

$$
\eta_{\rm FD}(x_h,s)=q_{\rm FD}(z,s),
\qquad
\eta_{\rm any}(x_h,s)=1-q_{\rm unreconstructed}(z,s).
$$

FD is the primary gate for the existing FD-conditioned resolution/PID model;
CD or FT reconstruction is an `any` success but not an FD success. A wrong
reconstructed PID remains a reconstruction success. Do not require correct PID,
reciprocal matching, physical reconstructed beta, small residuals, or response
fiducial cuts to define the efficiency denominator. Audit matching diagnostics
and malformed generated inputs separately rather than silently converting
association issues into PID or efficiency labels.

Without an independent geometrical-acceptance truth label, these are
**acceptance times reconstruction** probabilities, not intrinsic detector
tracking efficiencies. They apply only on the generated phase-space support.

## Training and closure

Use the event key `(source_file_id, event_id)` for the fixed event-disjoint
train/validation/test partition; never split different particles from one event
across partitions. Freeze split seed `20260822` independently of network seeds
`20260822`, `20260823`, and `20260824`. Each seed trains 3 species networks on
the same rows. Select checkpoints with validation cross-entropy, not test plots.
Use unweighted categorical cross-entropy to retain the physical class prior:

$$
\mathcal L_s=-\frac{1}{N_{s,\rm train}}
\sum_{i\in(s,\rm train,T=1)}\log q_{\vartheta,C_i}(z_i,s).
$$

For every generated-kinematic bin $B$ and category $c$, compare

$$
f_{\rm MC}(c;B,s)=\frac{\sum_{i\in(B,s,T=1)}\mathbf1[C_i=c]}{N_{B,s}},
\qquad
f_{\rm FM}(c;B,s)=\frac{\sum_{i\in(B,s,T=1)}q_{i,c}}{N_{B,s}}.
$$

Compare probabilities, not thresholded or sampled decisions. Produce separate
species closure in $p,\theta,\phi,v_z$, and $(p,\theta)$; show bin counts and
binomial intervals. Summaries use a fixed minimum of 100 rows per supported
kinematic cell and report excluded cells explicitly. Report mean signed error,
weighted and unweighted absolute closure error, categorical cross-entropy,
binary Brier scores for FD/any, and reliability diagrams with 20 fixed equal
probability bins. Empty bins stay empty, not zero-efficiency measurements.

Compare with the constant training-set category-frequency predictor on the
same test rows. This control matters: a constant predictor can be calibrated
overall while missing all kinematic structure. Report seed variation separately
from finite-test-sample uncertainty. Reuse the frozen electron reference to
place its all-event reliability curve beside the 3 conditional hadron curves;
the different conditioning must be explicit in titles and denominators.

## Factorization and scope

For the complete generated event $X$, the chain rule is exact:

$$
P(T,C_h,Y_h\mid X)=P(T\mid X)P(C_h\mid T,X)
P(Y_h\mid C_h,T,X),
$$

where $Y_h$ denotes reconstructed response/PID when a reconstruction exists
(the unreconstructed outcome carries an empty response). Replacing these
factors by single-particle models depending only on $x_e$ or $x_h,s$ is an
approximation: shared hits, occupancy, event start time, and generator-dependent
correlations need not factorize. Do not multiply marginal electron and hadron
efficiencies and call the result an exact event-level efficiency. Trigger
failures belong to the electron gate, not to each conditional hadron gate.

Closure here is against the same full-MC phase-space production, not a
data/MC detector-efficiency measurement. Transfer to physics-specific
generators needs new closure and support checks. Good reliability does not
guarantee kinematic closure; agreement of single-particle rates does not prove
joint reconstruction closure. No new conditional response, PID, beta-response,
or event-correlation model is claimed by this study.

## Primary physics references

1. [Ziegler et al., *The CLAS12 software framework and event reconstruction*,
   NIM A 959, 163472 (2020)](https://eprints.gla.ac.uk/208714/1/208714.pdf),
   sections 6.4 and 6.12. Charged tracks are reconstructed before response
   association and PID; figures 20-21 exhibit charge-, momentum-, and
   angle-dependent pion tracking losses. This motivates separate species and
   kinematic closure rather than a single global rate.
2. [Tyson et al., *Deep learning level-3 electron trigger for CLAS12*,
   CPC 290, 108783 (2023)](https://eprints.gla.ac.uk/298277/1/298277.pdf),
   section 2. The hardware electron trigger combines ECAL, DC, and HTCC
   responses. Its eligible-electron efficiency is not the broad generated
   acceptance/reconstruction probability used here.
3. [Anderlini et al., *The LHCb ultra-fast simulation option, Lamarr: design
   and validation* (2023)](https://arxiv.org/pdf/2309.13213), sections 3.1-3.2.
   The charged-particle pipeline separates acceptance, reconstruction-efficiency
   classification, resolution, and PID: a direct methodological precedent for
   adding this gate before an existing conditional response.
