# QuantumMC Simulations

This folder contains the CLAS12 Forward Foundation Model design material,
physics guide, canonical Aug17-26 phase-space Parquet sample, data-audit
loaders, and the first implemented stochastic neural response component.

## Directory map

- `Data_processing/` — consolidated design/startup plan, correspondence summary,
  and DuckDB loaders defining the full and FD-selected training views.
- `Understanding Physics/` — detector, reconstruction, probability, validation,
  and foundation-model guide.
- `phase-space_parquet-Aug17-26/` — canonical 20-million-row, five-million-event
  Parquet dataset.
- `step1/` — implemented conditional FD residual/PID mixture-density network,
  pseudocode, physics-to-code annotations, tests, configurations, checkpoints,
  and closure reports.

## Statistical scope

The planned response is

\[
P(Y\mid X)=P(T\mid x_e)\prod_i P(C_i\mid x_i,T)
P(\Delta_i,\widehat s_i\mid x_i,T,C_i).
\]

`step1` currently implements the final factor for triggered, FD-reconstructed
hadrons. It is not yet a complete trigger/efficiency model. See
`step1/PHYSICS_TO_CODE.md` and `step1/PSEUDOCODE.md`.

## Local training

```bash
cd step1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --config configs/fd_response_seed.yaml
```

## Tara one-GPU training

The uploaded location is `/home/htseng/QuantumMC_Simulations`. The large run
uses the complete selected population, a wider eight-component MDN, and one
visible GPU:

```bash
ssh tara
cd /home/htseng/QuantumMC_Simulations/step1
conda activate QuantumMC
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/tara_gpu_full.yaml
```

Results are stored in `step1/runs/tara_gpu_full/`. The exact environment,
commands, GPU, timings, metrics, and limitations are recorded in
`step1/TARA_TRAINING_REPORT.md`.

The completed large run used 1,270,698 training rows on one H100, selected
epoch 16, reached test residual NLL -4.758140 and REC-PID accuracy 67.22%, and
saved its checkpoint plus aggregate/kinematic closure artifacts remotely and
locally.

## Important limitations

- The step-one targets are raw residuals; versioned energy-loss/swum-back-phi
  corrected targets were not present in the delivered schema.
- The FD selection removes failure denominators, so it must not be used for
  trigger or reconstruction-efficiency training.
- A model checkpoint is a simulation surrogate, not evidence of agreement with
  real CLAS12 data. Physics release requires event/analysis closure and agreed
  numerical gates.
