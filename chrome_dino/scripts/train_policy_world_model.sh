#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 {rloo_wm|rlb_cwm} SEED [GPU]" >&2
  exit 2
fi

METHOD="$1"
SEED="$2"
GPU="${3:-0}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
WORLD_MODEL="${WORLD_MODEL:-$ROOT/checkpoints/world_model_step9000}"
REWARD_MODEL="${REWARD_MODEL:-$ROOT/checkpoints/reward_model.pt}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/policies}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"
RESULTS_ROOT="${RESULTS_ROOT:-$RUN_DIR/checkpoints}"

case "$METHOD" in
  rloo_wm|rlb_cwm) NAME="${METHOD}_s${SEED}" ;;
  *) echo "unknown method: $METHOD" >&2; exit 2 ;;
esac

[[ -d "$WORLD_MODEL" ]] || { echo "World model not found: $WORLD_MODEL" >&2; exit 2; }
[[ -f "$REWARD_MODEL" ]] || { echo "Reward model not found: $REWARD_MODEL" >&2; exit 2; }

mkdir -p "$LOG_DIR"
cd "$ROOT"
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED="$SEED" \
NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES="$GPU" \
DINO_VAE_PATH="$WORLD_MODEL/vae" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
"$PYTHON" -u train_policy_world_model.py \
  --method "$METHOD" --seed "$SEED" --results-dir "$NAME" \
  --output-root "$RESULTS_ROOT" \
  --model-folder "$WORLD_MODEL" --reward-model "$REWARD_MODEL" \
  --updates 2000 --num-groups 8 --group-size 4 --decisions 20 \
  --lr 1e-3 --gamma 0.99 --entropy-coef 0.04 --max-grad-norm 20 \
  --collision-penalty -1 --action-repeat 4 \
  --root-ttc-bins 8-12 --root-speed-bins 21-26,27-31,32-37,38-43 \
  --bird-spawn-prob 0.6 --bird-min-speed 20 --mid-bird-weight 2 \
  --num-inference-steps 10 --cfg-scale 1.0 --dtype bf16 \
  --save-every 100 --dump-every 200 \
  2>&1 | tee "$LOG_DIR/$NAME.log"
