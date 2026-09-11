# Optuna transfer to trigger-electron efficiency

The map $G_\psi(x_e)\simeq P(T=1\mid x_e)$, denominator, labels, and event
split were fixed. Optuna varied width, depth, dropout, batch size, learning
rate, weight decay, and schedule. Sixteen fixed-seed trials minimized validation
BCE; the test split was evaluated once after selection.

| Test quantity | Baseline | Optuna | Change |
|---|---:|---:|---:|
| BCE | 0.228773 | 0.222790 | -2.6% |
| Brier | 0.066687 | 0.064880 | -2.7% |
| ECE | 0.003053 | 0.001548 | -50.7% |
| ROC AUC | 0.946061 | 0.947502 | +0.00144 |
| $|\epsilon_{\rm FM}-\epsilon_{\rm MC}|$ | 0.001744 | 0.000831 | -52.4% |

Weighted binned-efficiency MAE falls by 60.0% in $p_e$, 57.7% in $\theta_e$,
77.6% in $\phi_e$, 38.6% in $v_{z,e}$, and 47.0% in $(p_e,\theta_e)$.

The selected recipe uses width 256, 6 layers, dropout 0.0583, batch size 8192,
learning rate 0.003009, weight decay $1.10\times10^{-4}$, and cosine decay.
The exact earlier Optuna recipe also improved validation BCE, but the local
search found this smaller model. All comparisons use seed 20260822; seed
robustness remains unmeasured.
