#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET_PATH:-$ROOT/datasets/factual}"
OUTPUT="${TTC_CACHE:-$ROOT/datasets/factual_ttc.npy}"

cd "$ROOT"
exec "${PYTHON:-python}" -m utils.build_ttc_cache "$DATASET" "$OUTPUT" --workers "${WORKERS:-8}"
