#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA="${REWARD_DATA_DIR:-$ROOT/datasets/reward_model}"
CACHE="$ROOT/runs/reward_model_cache/reward_data.npz"
TRAINING="$ROOT/runs/reward_model_training"
OUTPUT="${REWARD_MODEL_OUTPUT:-$ROOT/runs/reward_model.pt}"

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing reward model: $OUTPUT" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT")"

cd "$ROOT"
"$PYTHON" reward_model.py prepare --episodes "$DATA/expert" "$DATA/random" \
  --cache "$CACHE" --workers "${WORKERS:-8}"
"$PYTHON" reward_model.py train --cache "$CACHE" --output "$TRAINING"
cp "$TRAINING/reward_model.pt" "$OUTPUT"
echo "Saved reward model to $OUTPUT"
