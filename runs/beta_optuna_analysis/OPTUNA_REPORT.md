# Optuna transfer to the β-response branch

The 5-input, 4-target D10 physics definition and
$\lambda_{\rm PID}=1$ were fixed. Optuna varied only width, depth, dropout,
batch size, learning rate, weight decay, and learning-rate schedule.

Selection minimized validation-only

$$
T_{\rm val}:=
\frac{\sum_{s,b}N_{s,b}{\rm TV}_{s,b}}
{\sum_{s,b}N_{s,b}}
$$

over 16 fixed-seed trials. The test split was evaluated once after selection.

| Quantity | D10 default | Optuna | Change |
|---|---:|---:|---:|
| validation $T$ | 0.021382 | 0.009104 | -57.4% |
| test $T$ | 0.023674 | 0.008658 | -63.4% |
| test PID CE | 0.991976 | 0.977011 | -1.5% |
| test PID accuracy | 0.677787 | 0.679901 | +0.21 pp |
| test residual NLL | -4.952066 | -5.816826 | -0.864760 |
| test macro $W_1(\beta)$ | 0.005159 | 0.002497 | -51.6% |

The selected recipe uses width 768, 6 layers, dropout 0.14047, batch size
4096, learning rate 0.003360, weight decay $5.85\times10^{-5}$, and cosine
decay. The exact earlier Optuna recipe was the 2nd-best anchor
($T_{\rm val}=0.009776$); a local search around it found the reported winner.

This is a same-seed model-selection result, not a 10-seed causal ablation.
It improves the D10 endpoint but does not change the factorial conclusion about
$\beta_{\rm gen}$, $\Delta\beta$, or $\lambda_{\rm PID}$.
