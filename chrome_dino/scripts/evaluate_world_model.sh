#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_FOLDER="${MODEL_FOLDER:-$ROOT/checkpoints/world_model_step9000}"
DATASET_PATH="${DATASET_PATH:-$ROOT/datasets/factual}"
TTC_CACHE="${TTC_CACHE:-$ROOT/datasets/factual_ttc.npy}"
BENCHMARK_REPO="${BENCHMARK_REPO:-$ROOT/../counterfactual-benchmark}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/world_model_evaluation}"
EVALUATOR_DIR="${EVALUATOR_DIR:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

for path in "$MODEL_FOLDER" "$DATASET_PATH" "$BENCHMARK_REPO"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 2
  fi
done
if [[ ! -f "$TTC_CACHE" ]]; then
  echo "TTC cache does not exist: $TTC_CACHE" >&2
  echo "Set TTC_CACHE to the cache paired with DATASET_PATH." >&2
  exit 2
fi

args=(
  "$ROOT/evaluate_world_model.py"
  --model-folder "$MODEL_FOLDER"
  --dataset-path "$DATASET_PATH"
  --ttc-cache "$TTC_CACHE"
  --benchmark-repo "$BENCHMARK_REPO"
  --out-dir "$OUT_DIR"
  --evaluator-train-per-class 3000
  --evaluator-valid-per-class 1000
  --predictor-epochs 5
  --vae-epochs 8
)
if [[ -n "$EVALUATOR_DIR" ]]; then
  args+=(--evaluator-dir "$EVALUATOR_DIR")
fi
if [[ "${REUSE_GENERATED:-0}" == "1" ]]; then
  args+=(--reuse-generated)
fi

if [[ -d "$MODEL_FOLDER/vae" ]]; then
  export DINO_VAE_PATH="${DINO_VAE_PATH:-$MODEL_FOLDER/vae}"
fi

exec "$PYTHON_BIN" "${args[@]}" "$@"
