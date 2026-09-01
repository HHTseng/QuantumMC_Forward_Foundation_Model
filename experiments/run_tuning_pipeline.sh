#!/usr/bin/env bash
# End-to-end hyper-parameter tuning pipeline for the step-one FD response model.
#
# Stage 1  parallel Optuna search, one worker per GPU
# Stage 2  study analysis; writes configs/gpu_optuna_best.yaml
# Stage 3  baseline recipe and tuned recipe trained to completion, in parallel
# Stage 4  seed repeats of both recipes, so a gain can be separated from luck
# Stage 5  lambda_PID scan at the selected architecture (validation only)
# Stage 6  held-out comparison tables and figures
#
# The test split is touched only in stages 3, 4 and 6, by train.py's own
# evaluation. The search and the scan never see it.
#
# Usage: experiments/run_tuning_pipeline.sh [stage ...]      (default: all)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STORAGE="sqlite:///${ROOT}/runs/optuna_search/study.db"
STUDY="forwardfm-step1-capacity"
ANALYSIS="runs/optuna_analysis"
TRIALS_PER_WORKER="${TRIALS_PER_WORKER:-45}"
SEARCH_TIMEOUT="${SEARCH_TIMEOUT:-12600}"
REPEAT_SEEDS="${REPEAT_SEEDS:-20260823 20260824}"

STAGES=("$@")
stage_wanted() { [ "${#STAGES[@]}" -eq 0 ] || [[ " ${STAGES[*]} " == *" $1 "* ]]; }

if stage_wanted 1; then
  echo "== stage 1: parallel Optuna search =="
  mkdir -p runs/optuna_search
  for gpu in 0 1; do
    setsid nohup python experiments/tune_hyperparameters.py \
      --config configs/gpu_optuna_search.yaml \
      --storage "$STORAGE" --study-name "$STUDY" \
      --device "cuda:$gpu" --n-trials "$TRIALS_PER_WORKER" \
      --timeout "$SEARCH_TIMEOUT" --worker-tag "gpu$gpu" \
      > "runs/optuna_search/worker_gpu$gpu.log" 2>&1 < /dev/null &
    sleep 3
  done
  while pgrep -f tune_hyperparameters.py > /dev/null; do sleep 60; done
fi

if stage_wanted 2; then
  echo "== stage 2: study analysis =="
  python experiments/analyze_tuning.py \
    --storage "$STORAGE" --study-name "$STUDY" \
    --base-config configs/gpu_optuna_search.yaml \
    --output-dir "$ANALYSIS" \
    --best-config configs/gpu_optuna_best.yaml \
    --best-run-dir runs/optuna_best \
    > "$ANALYSIS/analyze_tuning.log" 2>&1 || { cat "$ANALYSIS/analyze_tuning.log"; exit 1; }
fi

if stage_wanted 3; then
  echo "== stage 3: baseline and tuned full runs =="
  mkdir -p runs/optuna_baseline_repro runs/optuna_best
  python train.py --config configs/gpu_full.yaml --run-dir runs/optuna_baseline_repro \
    --device cuda:0 > runs/optuna_baseline_repro/training.log 2>&1 &
  baseline_pid=$!
  python train.py --config configs/gpu_optuna_best.yaml --run-dir runs/optuna_best \
    --device cuda:1 > runs/optuna_best/training.log 2>&1 &
  tuned_pid=$!
  wait $baseline_pid $tuned_pid
fi

if stage_wanted 4; then
  echo "== stage 4: seed repeats =="
  for seed in $REPEAT_SEEDS; do
    mkdir -p "runs/seed_baseline_$seed" "runs/seed_tuned_$seed"
    python train.py --config configs/gpu_full.yaml --seed "$seed" \
      --run-dir "runs/seed_baseline_$seed" --device cuda:0 \
      > "runs/seed_baseline_$seed/training.log" 2>&1 &
    a=$!
    python train.py --config configs/gpu_optuna_best.yaml --seed "$seed" \
      --run-dir "runs/seed_tuned_$seed" --device cuda:1 \
      > "runs/seed_tuned_$seed/training.log" 2>&1 &
    b=$!
    wait $a $b
  done
fi

if stage_wanted 5; then
  echo "== stage 5: lambda_PID scan at the selected architecture =="
  python experiments/scan_pid_weight.py --config configs/gpu_optuna_best.yaml \
    --weights 0.05,0.1,0.2,0.5,1,2,5,10 --device cuda:0 \
    --output-dir "$ANALYSIS" --tag tuned_architecture \
    > "$ANALYSIS/pid_weight_scan.log" 2>&1
fi

if stage_wanted 6; then
  echo "== stage 6: held-out comparison =="
  python experiments/compare_final_models.py \
    --run "baseline=runs/optuna_baseline_repro" \
    --run "tuned=runs/optuna_best" \
    --output-dir "$ANALYSIS" > "$ANALYSIS/compare_final_models.log" 2>&1
  baseline_dirs="runs/optuna_baseline_repro"
  tuned_dirs="runs/optuna_best"
  for seed in $REPEAT_SEEDS; do
    baseline_dirs="$baseline_dirs,runs/seed_baseline_$seed"
    tuned_dirs="$tuned_dirs,runs/seed_tuned_$seed"
  done
  python experiments/summarize_seed_repeats.py \
    --group "baseline=$baseline_dirs" --group "tuned=$tuned_dirs" \
    --output-dir "$ANALYSIS" > "$ANALYSIS/seed_repeats.log" 2>&1
fi

echo "pipeline complete; artifacts in $ANALYSIS"
