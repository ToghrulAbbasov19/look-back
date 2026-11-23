#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 METHOD" >&2
  exit 2
fi

METHOD="$1"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
JOBS="${JOBS:-16}"
SEEDS_TEXT="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/policies}"
RESULTS_ROOT="${RESULTS_ROOT:-$RUN_DIR/checkpoints}"
read -r -a SEEDS_ARRAY <<< "$SEEDS_TEXT"

cd "$ROOT"
mkdir -p "$RUN_DIR"
# Set CHECKPOINT to skip validation and test that checkpoint directly.
if [[ -z "${CHECKPOINT:-}" ]]; then
  "$PYTHON" select_checkpoints.py \
    --method "$METHOD" --seeds "${SEEDS_ARRAY[@]}" \
    --start 100 --end 2000 --step 100 \
    --episodes 30 --episode-seed 2200000000 \
    --action-repeat 4 --jobs "$JOBS" --results-root "$RESULTS_ROOT" \
    --outdir "$RUN_DIR/validation"

  CHECKPOINT=$("$PYTHON" - "$METHOD" "$RUN_DIR" <<'PY'
import csv
import sys
from pathlib import Path

method = sys.argv[1]
path = Path(sys.argv[2]) / "validation" / f"validation_{method}_e2200000000.csv"
rows = list(csv.DictReader(path.open()))
print(max(rows, key=lambda row: float(row["frame0_mean"]))["checkpoint"])
PY
  )
fi

printf '%s %s\n' "$METHOD" "$CHECKPOINT" | tee -a "$RUN_DIR/selected_checkpoints.txt"
"$PYTHON" select_checkpoints.py \
  --method "$METHOD" --seeds "${SEEDS_ARRAY[@]}" \
  --start "$CHECKPOINT" --end "$CHECKPOINT" --step 100 \
  --episodes 100 --episode-seed 2300000000 \
  --action-repeat 4 --jobs "$JOBS" --results-root "$RESULTS_ROOT" \
  --outdir "$RUN_DIR/test"
