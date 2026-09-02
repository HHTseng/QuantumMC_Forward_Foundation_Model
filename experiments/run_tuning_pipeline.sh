#!/usr/bin/env bash
# End-to-end hyper-parameter tuning pipeline for the step-one FD response model.
#
# Stage 1  parallel Optuna search, one worker per GPU
# Stage 2  study analysis; writes configs/gpu_optuna_best.yaml
# Stage 3  baseline recipe and tuned recipe trained to completion, in parallel
# Stage 4  seed repeats of both recipes, so a gain can be separated from luck
# Stage 5  lambda_PID scan at the selected architecture (validation only)
# Stage 6  held-out comparison tables and figures
# Stage 7  refit at the scan-optimal lambda_PID, with seed repeats, and redo the
#          comparison. The joint search samples lambda together with everything
#          else and can miss a productive setting that only pays off at the
#          selected architecture, which is exactly what happened here.
# Stage 8  the same lambda_PID with learning-rate warm-up, over WARMUP_SEEDS, to
#          test whether the early-training instability rather than the weight
#          itself is what makes stage 7's setting unusable.
# Stage 9  make the better solution the target rather than an accident, two ways:
#          (A) warm start from each seed's released checkpoint and fine-tune the
#              whole model at the large weight, so there is no fragile early
#              phase to survive; (B) train from scratch at the large weight but
#              give the shared trunk a smaller learning rate than the heads, so
#              the PID term can be strong without large trunk updates.
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
# Set from the stage 5 scan; see runs/optuna_analysis/pid_weight_tuned_architecture.csv
REFIT_PID_WEIGHT="${REFIT_PID_WEIGHT:-2.0}"
LR_WARMUP_EPOCHS="${LR_WARMUP_EPOCHS:-5}"
WARMUP_SEEDS="${WARMUP_SEEDS:-20260822 20260823 20260824 20260825 20260826 20260827}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
FINETUNE_LR_FACTOR="${FINETUNE_LR_FACTOR:-0.1}"
BACKBONE_LR_MULTIPLIER="${BACKBONE_LR_MULTIPLIER:-0.25}"

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

if stage_wanted 7; then
  echo "== stage 7: refit at lambda_PID=$REFIT_PID_WEIGHT =="
  python - "$REFIT_PID_WEIGHT" <<'PYEOF'
import sys
import yaml

with open("configs/gpu_optuna_best.yaml", "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
config["project"]["name"] = "clas12-forward-fm-step1-optuna-best-pidweight"
config["training"]["pid_loss_weight"] = float(sys.argv[1])
config["output"]["run_dir"] = "runs/optuna_best_pidweight"
with open("configs/gpu_optuna_best_pidweight.yaml", "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
print("wrote configs/gpu_optuna_best_pidweight.yaml")
PYEOF
  mkdir -p runs/optuna_best_pidweight
  python train.py --config configs/gpu_optuna_best_pidweight.yaml \
    --run-dir runs/optuna_best_pidweight --device cuda:0 \
    > runs/optuna_best_pidweight/training.log 2>&1 &
  first=$!
  seed_list=($REPEAT_SEEDS)
  mkdir -p "runs/seed_pidweight_${seed_list[0]}"
  python train.py --config configs/gpu_optuna_best_pidweight.yaml --seed "${seed_list[0]}" \
    --run-dir "runs/seed_pidweight_${seed_list[0]}" --device cuda:1 \
    > "runs/seed_pidweight_${seed_list[0]}/training.log" 2>&1 &
  second=$!
  wait $first $second
  for seed in "${seed_list[@]:1}"; do
    mkdir -p "runs/seed_pidweight_$seed"
    python train.py --config configs/gpu_optuna_best_pidweight.yaml --seed "$seed" \
      --run-dir "runs/seed_pidweight_$seed" --device cuda:0 \
      > "runs/seed_pidweight_$seed/training.log" 2>&1
  done

  python experiments/compare_final_models.py \
    --run "baseline=runs/optuna_baseline_repro" \
    --run "tuned=runs/optuna_best" \
    --run "tuned+lambda=runs/optuna_best_pidweight" \
    --output-dir "$ANALYSIS" > "$ANALYSIS/compare_final_models.log" 2>&1
  baseline_dirs="runs/optuna_baseline_repro"
  tuned_dirs="runs/optuna_best"
  pidweight_dirs="runs/optuna_best_pidweight"
  for seed in $REPEAT_SEEDS; do
    baseline_dirs="$baseline_dirs,runs/seed_baseline_$seed"
    tuned_dirs="$tuned_dirs,runs/seed_tuned_$seed"
    pidweight_dirs="$pidweight_dirs,runs/seed_pidweight_$seed"
  done
  python experiments/summarize_seed_repeats.py \
    --group "baseline=$baseline_dirs" --group "tuned=$tuned_dirs" \
    --group "tuned+lambda=$pidweight_dirs" \
    --output-dir "$ANALYSIS" > "$ANALYSIS/seed_repeats.log" 2>&1
fi

if stage_wanted 8; then
  echo "== stage 8: lambda_PID=$REFIT_PID_WEIGHT with $LR_WARMUP_EPOCHS-epoch warm-up =="
  python - "$REFIT_PID_WEIGHT" "$LR_WARMUP_EPOCHS" <<'PYEOF'
import sys
import yaml

with open("configs/gpu_optuna_best.yaml", "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
config["project"]["name"] = "clas12-forward-fm-step1-pidweight-warmup"
config["training"]["pid_loss_weight"] = float(sys.argv[1])
config["training"]["lr_warmup_epochs"] = float(sys.argv[2])
config["output"]["run_dir"] = "runs/optuna_best_pidweight_warmup"
with open("configs/gpu_optuna_best_pidweight_warmup.yaml", "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
print("wrote configs/gpu_optuna_best_pidweight_warmup.yaml")
PYEOF

  gpu=0
  pids=()
  for seed in $WARMUP_SEEDS; do
    dir="runs/seed_pidweight_warmup_$seed"
    mkdir -p "$dir"
    python train.py --config configs/gpu_optuna_best_pidweight_warmup.yaml \
      --seed "$seed" --run-dir "$dir" --device "cuda:$gpu" \
      > "$dir/training.log" 2>&1 &
    pids+=($!)
    if [ "$gpu" -eq 1 ]; then wait "${pids[@]}"; pids=(); gpu=0; else gpu=1; fi
  done
  if [ "${#pids[@]}" -gt 0 ]; then wait "${pids[@]}"; fi

  warmup_dirs=""
  plain_dirs=""
  for seed in $WARMUP_SEEDS; do
    warmup_dirs="$warmup_dirs,runs/seed_pidweight_warmup_$seed"
    if [ "$seed" = "20260822" ]; then
      plain_dirs="$plain_dirs,runs/optuna_best_pidweight"
    else
      plain_dirs="$plain_dirs,runs/seed_pidweight_$seed"
    fi
  done
  warmup_dirs="${warmup_dirs#,}"
  plain_dirs="${plain_dirs#,}"

  python experiments/plot_pid_weight_stability.py \
    --reference "lambda $REFIT_PID_WEIGHT, no warm-up=$plain_dirs" \
    --variant "lambda $REFIT_PID_WEIGHT, warm-up=$warmup_dirs" \
    --output-dir "$ANALYSIS" > "$ANALYSIS/pid_weight_warmup_stability.log" 2>&1
fi

if stage_wanted 9; then
  echo "== stage 9A: fine-tune the released checkpoints at lambda_PID=$REFIT_PID_WEIGHT =="
  python - "$REFIT_PID_WEIGHT" "$FINETUNE_EPOCHS" "$FINETUNE_LR_FACTOR" "$BACKBONE_LR_MULTIPLIER" <<'PYEOF'
import sys
import yaml

weight, epochs, lr_factor, backbone_multiplier = (
    float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
)
with open("configs/gpu_optuna_best.yaml", "r", encoding="utf-8") as handle:
    base = yaml.safe_load(handle)

finetune = yaml.safe_load(yaml.safe_dump(base))
finetune["project"]["name"] = "clas12-forward-fm-step1-pid-finetune"
finetune["training"]["pid_loss_weight"] = weight
finetune["training"]["epochs"] = epochs
finetune["training"]["learning_rate"] = base["training"]["learning_rate"] * lr_factor
finetune["training"]["early_stopping_patience"] = max(4, epochs // 3)
finetune["output"]["run_dir"] = "runs/pid_finetune"
with open("configs/gpu_pid_finetune.yaml", "w", encoding="utf-8") as handle:
    yaml.safe_dump(finetune, handle, sort_keys=False)

decoupled = yaml.safe_load(yaml.safe_dump(base))
decoupled["project"]["name"] = "clas12-forward-fm-step1-pid-decoupled-lr"
decoupled["training"]["pid_loss_weight"] = weight
decoupled["training"]["backbone_lr_multiplier"] = backbone_multiplier
decoupled["output"]["run_dir"] = "runs/pid_decoupled_lr"
with open("configs/gpu_pid_decoupled_lr.yaml", "w", encoding="utf-8") as handle:
    yaml.safe_dump(decoupled, handle, sort_keys=False)
print("wrote configs/gpu_pid_finetune.yaml and configs/gpu_pid_decoupled_lr.yaml")
PYEOF

  released_dir() {
    if [ "$1" = "20260822" ]; then echo "runs/optuna_best"; else echo "runs/seed_tuned_$1"; fi
  }

  # Fine-tuning needs a released checkpoint per seed, and it must come from the
  # same split: the seed drives the partition hash, so a checkpoint from another
  # seed would leak that seed's training rows into this one's test split.
  # train.py refuses the mismatch, and any missing checkpoint is trained here.
  gpu=0; pids=()
  for seed in $WARMUP_SEEDS; do
    source_dir="$(released_dir "$seed")"
    if [ ! -f "$source_dir/model.pt" ]; then
      echo "training the missing released checkpoint for seed $seed"
      mkdir -p "$source_dir"
      python train.py --config configs/gpu_optuna_best.yaml --seed "$seed" \
        --run-dir "$source_dir" --device "cuda:$gpu" \
        > "$source_dir/training.log" 2>&1 &
      pids+=($!)
      if [ "$gpu" -eq 1 ]; then wait "${pids[@]}"; pids=(); gpu=0; else gpu=1; fi
    fi
  done
  if [ "${#pids[@]}" -gt 0 ]; then wait "${pids[@]}"; pids=(); fi

  gpu=0; pids=()
  for seed in $WARMUP_SEEDS; do
    source_dir="$(released_dir "$seed")"
    dir="runs/seed_pid_finetune_$seed"
    mkdir -p "$dir"
    python train.py --config configs/gpu_pid_finetune.yaml --seed "$seed" \
      --init-from "$source_dir/model.pt" --run-dir "$dir" --device "cuda:$gpu" \
      > "$dir/training.log" 2>&1 &
    pids+=($!)
    if [ "$gpu" -eq 1 ]; then wait "${pids[@]}"; pids=(); gpu=0; else gpu=1; fi
  done
  if [ "${#pids[@]}" -gt 0 ]; then wait "${pids[@]}"; fi

  echo "== stage 9B: decoupled trunk learning rate at lambda_PID=$REFIT_PID_WEIGHT =="
  gpu=0; pids=()
  for seed in $WARMUP_SEEDS; do
    dir="runs/seed_pid_decoupled_$seed"
    mkdir -p "$dir"
    python train.py --config configs/gpu_pid_decoupled_lr.yaml --seed "$seed" \
      --run-dir "$dir" --device "cuda:$gpu" > "$dir/training.log" 2>&1 &
    pids+=($!)
    if [ "$gpu" -eq 1 ]; then wait "${pids[@]}"; pids=(); gpu=0; else gpu=1; fi
  done
  if [ "${#pids[@]}" -gt 0 ]; then wait "${pids[@]}"; fi
fi

echo "pipeline complete; artifacts in $ANALYSIS"
