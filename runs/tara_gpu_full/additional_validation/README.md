# Additional PID validation versus generated momentum

- Checkpoint: `runs/tara_gpu_full/model.pt`
- Checkpoint SHA-256: `22dde8fe78c5bec337e5014be46e4c8037673015bc88d4fbe812c05bebcffe11`
- Checkpoint best epoch: 16
- Dataset metadata SHA-256 recorded in checkpoint: `6a7245cb0ec4125610b9dcd8c1635d70a7773eeb2b29d146dd80d5f149eb43ab`
- Evaluation split: event-disjoint held-out test split
- Test rows: 158,985
- Binning: 10 equal-population momentum bins, computed separately per generated species
- PID decision for accuracy: `argmax` of reconstructed-PID probabilities
- No checkpoint weights were changed during this evaluation.

## Reproduce after cloning

Install the repository dependencies and place the external Parquet sample next
to the cloned repository:

```text
QuantumMC_Simulations/
|-- QuantumMC_Forward_Foundation_Model/
`-- phase-space_parquet-Aug17-26/
    `-- particle_responses/*.parquet
```

The source Parquet data are not stored in Git. From the repository root, run:

```bash
python runs/tara_gpu_full/additional_validation/pid_vs_gen_p_validation.py
```

The portable defaults use `runs/tara_gpu_full/model.pt`,
`configs/gpu_full.yaml`, automatic CUDA/MPS/CPU selection, ten
species-specific equal-population momentum bins, and this directory for
outputs. Every option can be overridden with `--help`.

## Saved artifacts

- `pid_performance_vs_gen_p.png`
- `pid_response_vs_gen_p.png`
- `pid_performance_vs_gen_p.csv`
- `pid_response_vs_gen_p.csv`
- `pid_vs_gen_p_validation.py`

## Quick range check

- Lowest binned top-1 accuracy: 33.0175% for generated PID 211, 7.216–8.977 GeV (`n=5700`).
- Highest binned top-1 accuracy: 98.1610% for generated PID 2212, 0.435–1.209 GeV (`n=5601`).
