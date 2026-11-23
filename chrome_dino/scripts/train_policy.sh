#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 {reinforce|reinforce_baseline|rloo|rlb} SEED [GPU]" >&2
  exit 2
fi

METHOD="$1"
SEED="$2"
GPU="${3:-0}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/policies}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"
RESULTS_ROOT="${RESULTS_ROOT:-$RUN_DIR/checkpoints}"

case "$METHOD" in
  reinforce|reinforce_baseline|rloo|rlb) ;;
  *) echo "unknown method: $METHOD" >&2; exit 2 ;;
esac

NAME="${METHOD}_s${SEED}"
mkdir -p "$LOG_DIR"
cd "$ROOT"

CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED="$SEED" \
NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES="$GPU" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
"$PYTHON" -u train_policy.py \
  --method "$METHOD" \
  --seed "$SEED" --results-dir "$NAME" --output-root "$RESULTS_ROOT" \
  --validation-log-dir "$RUN_DIR/validation" \
  --updates 2000 --num-groups 8 --group-size 4 \
  --root-ttc-bins 8-12 --root-speed-bins 21-26,27-31,32-37,38-43 \
  --action-repeat 4 --max-steps 80 \
  --collision-penalty -1 --num-workers "${WORKERS:-6}" --save-every 100 \
  --lr 1e-3 --value-lr 3e-2 --entropy-coef 0.04 --gamma 0.99 \
  --bird-spawn-prob 0.6 --bird-min-speed 20 --mid-bird-weight 2 \
  2>&1 | tee "$LOG_DIR/$NAME.log"
