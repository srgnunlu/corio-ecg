#!/usr/bin/env bash
# Run train_omi_consistency.py over folds x consistency weights x seeds.
# One JSON per cell, named so summarise_consistency_sweep.py can pair them.
# A cell whose JSON already exists is skipped, so a killed sweep resumes.
#
# Usage:
#   scripts/run_consistency_sweep.sh <corpus_manifest> <results_dir> [extra train args...]
# Grid via environment (defaults reproduce the Phase B v1 sweep):
#   SWEEP_FOLDS="0 1"  SWEEP_WEIGHTS="0 1.0"  SWEEP_SEEDS="20260802 20260814 20260815"

set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <corpus_manifest> <results_dir> [extra train args...]" >&2
  exit 2
fi

CORPUS="$1"
RESULTS="$2"
shift 2

FOLDS=${SWEEP_FOLDS:-"0 1"}
WEIGHTS=${SWEEP_WEIGHTS:-"0 1.0"}
SEEDS=${SWEEP_SEEDS:-"20260802 20260814 20260815"}
PYTHON=${PYTHON:-.venv/bin/python}
SWEEP_NAME=$(basename "$RESULTS")

mkdir -p "$RESULTS" "models/omi/sweeps"

for fold in $FOLDS; do
  for weight in $WEIGHTS; do
    for seed in $SEEDS; do
      cell="f${fold}_w${weight}_s${seed}"
      out="$RESULTS/consistency_${cell}.json"
      if [[ -f "$out" ]]; then
        echo "[sweep] skip ${cell} (exists)"
        continue
      fi
      echo "[sweep] ${SWEEP_NAME} ${cell} $(date '+%H:%M:%S')"
      "$PYTHON" scripts/train_omi_consistency.py \
        --corpus "$CORPUS" \
        --validation-fold "$fold" \
        --consistency-weight "$weight" \
        --seed "$seed" \
        --output "$out" \
        --save-to "models/omi/sweeps/${SWEEP_NAME}_${cell}.pt" \
        "$@" || echo "[sweep] FAILED ${cell} (exit $?)"
    done
  done
done
echo "[sweep] ${SWEEP_NAME} complete $(date '+%H:%M:%S')"
