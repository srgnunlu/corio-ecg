#!/usr/bin/env bash
# Vast.ai VPS setup script — installs Corio ECG pipeline with CUDA support
# Usage: Upload this script to VPS, then run: bash setup_vps.sh
# After setup, run: bash run_app.sh

set -euo pipefail

echo "================================================"
echo "Corio ECG — VPS Setup (CUDA)"
echo "================================================"

# Step 1: System packages
echo ""
echo "[1/6] Installing system packages..."
apt-get update -qq && apt-get install -y -qq git git-lfs ffmpeg libsm6 libxext6 > /dev/null 2>&1
echo "  OK"

# Step 2: Clone project
echo ""
echo "[2/6] Cloning Corio ECG..."
WORK_DIR="/workspace/corio-ecg"
if [ -d "$WORK_DIR" ]; then
    echo "  Already exists, pulling latest..."
    cd "$WORK_DIR" && git pull
else
    git clone https://github.com/sergenunlu/Corio-ECG.git "$WORK_DIR"
    cd "$WORK_DIR"
    git checkout feature/test-ui-gradio
fi

# Step 3: Install Python dependencies
echo ""
echo "[3/6] Installing Python dependencies..."
pip install -q -e ".[digitiser,web]"
pip install -q yacs torch-tps
echo "  OK"

# Step 4: Clone external repos (Open-ECG-Digitizer)
echo ""
echo "[4/6] Setting up Open-ECG-Digitizer..."
python scripts/setup_phase2.py

# Step 5: Download ECGFounder model
echo ""
echo "[5/6] Downloading ECGFounder model..."
python scripts/download_models.py

# Step 6: Verify CUDA
echo ""
echo "[6/6] Verifying CUDA setup..."
python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  WARNING: No CUDA — digitiser will run on CPU (slow)')
"

echo ""
echo "================================================"
echo "Setup complete!"
echo ""
echo "To run the app:"
echo "  cd $WORK_DIR"
echo "  python -m src.web.app"
echo ""
echo "App will be available at http://localhost:7860"
echo "Use Vast.ai port forwarding or --share for remote access"
echo "================================================"
