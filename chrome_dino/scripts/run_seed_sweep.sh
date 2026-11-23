#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
METHODS_TEXT="${METHODS:-reinforce reinforce_baseline rloo rlb}"
SEEDS_TEXT="${SEEDS:-1 2 3 4 5}"
GPU_LIST_TEXT="${GPU_LIST:-0 1 2 3 4 5 6 7}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/policies}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"
RESULTS_ROOT="${RESULTS_ROOT:-$RUN_DIR/checkpoints}"

read -r -a METHODS_ARRAY <<< "$METHODS_TEXT"
read -r -a SEEDS_ARRAY <<< "$SEEDS_TEXT"
read -r -a GPUS_ARRAY <<< "$GPU_LIST_TEXT"

if (( ${#GPUS_ARRAY[@]} == 0 )); then
  echo "GPU_LIST must contain at least one GPU index" >&2
  exit 2
fi
if (( MAX_PARALLEL < 1 )); then
  echo "MAX_PARALLEL must be at least 1" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
job_index=0
failures=0
pids=()

for method in "${METHODS_ARRAY[@]}"; do
  for seed in "${SEEDS_ARRAY[@]}"; do
    gpu="${GPUS_ARRAY[$((job_index % ${#GPUS_ARRAY[@]}))]}"
    case "$method" in
      rloo_wm|rlb_cwm) launcher="$ROOT/scripts/train_policy_world_model.sh" ;;
      *) launcher="$ROOT/scripts/train_policy.sh" ;;
    esac
    echo "[$(date -Is)] launch method=$method seed=$seed gpu=$gpu"
    RUN_DIR="$RUN_DIR" LOG_DIR="$LOG_DIR" RESULTS_ROOT="$RESULTS_ROOT" \
      "$launcher" "$method" "$seed" "$gpu" &
    pids+=("$!")
    ((job_index += 1))
    if (( ${#pids[@]} >= MAX_PARALLEL )); then
      for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
          ((failures += 1))
        fi
      done
      pids=()
    fi
  done
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    ((failures += 1))
  fi
done
if (( failures > 0 )); then
  echo "[$(date -Is)] $failures run(s) failed" >&2
  exit 1
fi
echo "[$(date -Is)] completed ${#METHODS_ARRAY[@]} methods x ${#SEEDS_ARRAY[@]} seeds"
