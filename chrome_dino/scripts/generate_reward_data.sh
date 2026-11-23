#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
OUT="${REWARD_DATA_DIR:-$ROOT/datasets/reward_model}"

cd "$ROOT"
"$PYTHON" -m chrome_dino_game.generate_reward_data \
  --policy oracle --episodes 100 --game-mode train --save-every 4 \
  --max-frames 1500 --k 4 --seed 0 \
  --out "$OUT/expert"
"$PYTHON" -m chrome_dino_game.generate_reward_data \
  --policy random --episodes 100 --game-mode train --save-every 4 \
  --max-frames 1500 --k 4 --seed 0 \
  --out "$OUT/random"
