#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
OUT="${DATASET_DIR:-$ROOT/datasets/world_model}"

cd "$ROOT"
"$PYTHON" -m chrome_dino_game.generate_dataset \
  --out "$OUT" \
  --expert-episodes 1000 \
  --random-episodes 0 \
  --decisions 150 \
  --k 4 \
  --workers "${WORKERS:-8}" \
  --seed 0 \
  --filler-prob 0.7 \
  --safe-ttc 24 \
  --deviate-prob 0.3 \
  --deviate-ttc 16 \
  --expert-bird-spawn-prob 0.6 \
  --expert-bird-min-speed 20 \
  --expert-mid-bird-weight 2 \
  --bird-phase "$ROOT/configs/sprite_phase/world_model.txt"
