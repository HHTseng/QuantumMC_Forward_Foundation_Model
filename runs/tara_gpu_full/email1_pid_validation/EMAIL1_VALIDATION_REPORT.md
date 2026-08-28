# First-email PID validation reproduction

- Checkpoint SHA-256: `22dde8fe78c5bec337e5014be46e4c8037673015bc88d4fbe812c05bebcffe11`
- Dataset metadata SHA-256: `6a7245cb0ec4125610b9dcd8c1635d70a7773eeb2b29d146dd80d5f149eb43ab`
- Best epoch: 16
- Held-out test rows: 158,985
- Momentum bins: fixed 1-GeV intervals from 0 to 9 GeV
- Forward-FM statistic: mean softmax probability
- Model weights changed: no

## Momentum-integrated correct-ID response

| Generated species | COATJAVA | Forward FM | FM - COATJAVA |
|---|---:|---:|---:|
| pi- | 0.550184 | 0.546732 | -0.003452 |
| pi+ | 0.591098 | 0.406650 | -0.184448 |
| proton | 0.812642 | 0.577365 | -0.235277 |

## Largest fixed-bin distribution discrepancies

| Generated species | Momentum [GeV] | N | TV distance | Worst reconstructed PID | Max channel error |
|---|---:|---:|---:|---:|---:|
| pi+ | 0–1 | 3118 | 0.466248 | 211 | 0.463535 |
| pi+ | 1–2 | 7438 | 0.371406 | 2212 | 0.346877 |
| pi+ | 2–3 | 7854 | 0.309191 | 2212 | 0.286087 |
| proton | 0–1 | 3730 | 0.290214 | 2212 | 0.289488 |
| pi+ | 3–4 | 8147 | 0.287351 | 2212 | 0.249215 |

The low-momentum matrices use only physical generated pi+ and proton rows. The `other` column is the sum of all reconstructed PID classes other than pi+, proton, and K+.
