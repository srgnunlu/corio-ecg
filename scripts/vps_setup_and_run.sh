#!/usr/bin/env bash
# VPS one-shot setup & evaluation runner for Corio ECG Phase 2
# Usage: curl the repo, then run this script on a Vast.ai GPU instance
#
#   ssh into VPS, then:
#     git clone https://github.com/<your-repo>/corio-ecg.git
#     cd corio-ecg
#     export KAGGLE_USERNAME="your_kaggle_username"
#     export KAGGLE_KEY="your_kaggle_api_key"
#     bash scripts/vps_setup_and_run.sh [--max-samples N]
#
# Kaggle credentials: Go to kaggle.com → Settings → API → Create New Token
# Requirements: Ubuntu 22.04+, NVIDIA GPU with CUDA, git-lfs, Python 3.11+

set -euo pipefail

# --- Configuration ---
MAX_SAMPLES="${1:-}"  # optional: pass --max-samples N
PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "Corio ECG — VPS Phase 2 Setup & Evaluation"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Python:       $($PYTHON --version 2>&1)"

# Check Kaggle credentials
if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    echo "Kaggle:       $KAGGLE_USERNAME (credentials set)"
else
    echo "Kaggle:       NOT SET — will fall back to PhysioNet (slow)"
    echo "  Tip: export KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx"
fi
echo ""

# --- Step 1: System dependencies ---
echo "--- Step 1: System dependencies ---"
if ! command -v git-lfs &> /dev/null; then
    echo "[INSTALL] git-lfs..."
    apt-get update -qq && apt-get install -y -qq git-lfs
    git lfs install
else
    echo "[OK] git-lfs available"
fi

if ! command -v unzip &> /dev/null; then
    echo "[INSTALL] unzip..."
    apt-get update -qq && apt-get install -y -qq unzip
else
    echo "[OK] unzip available"
fi

# --- Step 2: Python virtual environment ---
echo ""
echo "--- Step 2: Python virtual environment ---"
if [ ! -d "$VENV_DIR" ]; then
    echo "[CREATE] Virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "[OK] venv activated: $(which python)"

# Upgrade pip
pip install --quiet --upgrade pip

# --- Step 3: Install project dependencies ---
echo ""
echo "--- Step 3: Install Python dependencies ---"

# Install PyTorch with CUDA (auto-detect CUDA version)
CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
if [ -n "$CUDA_VERSION" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "[INSTALL] PyTorch with CUDA 12.x..."
        pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "[INSTALL] PyTorch with CUDA 11.8..."
        pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
else
    echo "[WARN] No NVIDIA GPU detected — installing CPU PyTorch"
    pip install --quiet torch torchvision torchaudio
fi

# Install project in editable mode
echo "[INSTALL] Project dependencies..."
pip install --quiet -e ".[dev]"

# Phase 2 extra deps (Open-ECG-Digitizer needs these)
echo "[INSTALL] Phase 2 extras..."
pip install --quiet yacs torch-tps scikit-image kaggle

echo "[OK] All Python packages installed"

# --- Step 4: Clone external repos ---
echo ""
echo "--- Step 4: External dependencies ---"
python scripts/setup_phase2.py

# --- Step 5: Download ECGFounder model ---
echo ""
echo "--- Step 5: ECGFounder checkpoint ---"
python scripts/download_models.py

# --- Step 6: Download PTB-XL dataset ---
echo ""
echo "--- Step 6: PTB-XL dataset ---"
python scripts/download_ptbxl.py

# --- Step 7: Quick smoke test ---
echo ""
echo "--- Step 7: Smoke test (5 samples) ---"
python scripts/run_full_evaluation.py --max-samples 5
echo "[OK] Smoke test passed"

# --- Step 8: Full evaluation ---
echo ""
echo "============================================================"
echo "STARTING FULL EVALUATION"
echo "============================================================"

EVAL_CMD="python scripts/run_full_evaluation.py"
if [ -n "$MAX_SAMPLES" ]; then
    EVAL_CMD="$EVAL_CMD --max-samples $MAX_SAMPLES"
    echo "Running with --max-samples $MAX_SAMPLES"
else
    echo "Running on FULL PTB-XL test fold (~2000 records)"
fi

# Run with nohup so it survives SSH disconnection
echo "Logging to: results/vps_evaluation.log"
echo ""

mkdir -p results
$EVAL_CMD 2>&1 | tee results/vps_evaluation.log

echo ""
echo "============================================================"
echo "EVALUATION COMPLETE"
echo "============================================================"
echo "Results: results/metrics/roundtrip_comparison.json"
echo "Log:     results/vps_evaluation.log"
echo "Plots:   results/figures/"
echo ""
echo "To download results to your Mac:"
echo "  scp -r <vps>:$(pwd)/results/ ./results-vps/"
echo "============================================================"
