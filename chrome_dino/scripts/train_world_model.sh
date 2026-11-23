#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${DATASET_DIR:-$ROOT/datasets/world_model}"
OUTPUT_DIR="${WORLD_MODEL_DIR:-$ROOT/runs/world_model_step9000}"
BASE_VAE="${BASE_VAE:-$ROOT/checkpoints/base_vae}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
PROCESSES="${PROCESSES:-8}"
SEED="${SEED:-0}"
RESUME="${RESUME:-0}"

EXTRA_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint latest)
elif [[ "$RESUME" != "0" ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 2
fi

if [[ "$RESUME" == "0" && -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite nonempty world-model directory: $OUTPUT_DIR" >&2
  exit 1
fi

cd "$ROOT"
[[ -d "$BASE_VAE" ]] || { echo "Pretrained VAE not found: $BASE_VAE" >&2; exit 2; }
export DINO_VAE_PATH="$BASE_VAE"
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="$GPU_LIST" accelerate launch \
  --module \
  --num_processes "$PROCESSES" --mixed_precision bf16 \
  diffusion.train_world_model \
  --unet_size small \
  --ddp_backend gloo \
  --dataset_name "$DATASET_DIR" \
  --seed "$SEED" \
  --collision_repeat 2 \
  --output_dir "$OUTPUT_DIR" \
  --train_batch_size 4 \
  --dataloader_num_workers 4 \
  --learning_rate 2e-4 \
  --lr_scheduler constant \
  --max_grad_norm 1.0 \
  --max_train_steps 9000 \
  --checkpointing_steps 1500 \
  --validation_steps 500 \
  --mixed_precision bf16 \
  --use_cfg \
  --use_ema \
  --ema_decay 0.9999 \
  --report_to tensorboard \
  "${EXTRA_ARGS[@]}"
