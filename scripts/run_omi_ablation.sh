#!/usr/bin/env bash
# Sequential ablation for the OMI head: MLP, +multi-task, +NSTEMI up-weighting.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export TQDM_DISABLE=1
export CORIO_CALIBRATION=0

echo "=== v2a: MLP head ==="
python scripts/train_omi_finetune.py --hidden-dim 256 --tag v2a

echo
echo "=== v2b: MLP head + multi-task ==="
python scripts/train_omi_finetune.py --hidden-dim 256 --multi-task --tag v2b

echo
echo "=== v2c: MLP head + multi-task + NSTEMI up-weighting ==="
python scripts/train_omi_finetune.py --hidden-dim 256 --multi-task --nstemi-weight 3.0 --tag v2c

echo
echo "ABLATION DONE"
