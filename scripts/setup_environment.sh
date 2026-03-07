#!/bin/bash
# Corio ECG - Environment Setup Script
# Sets up Python via mise, creates venv, installs dependencies

set -e

echo "=== Corio ECG - Environment Setup ==="
echo ""

# Step 1: Check mise is installed
if ! command -v mise &> /dev/null; then
    echo "ERROR: mise is not installed."
    echo "Install it from: https://mise.jdx.dev/"
    exit 1
fi

echo "[1/6] Installing Python 3.11 via mise..."
mise install python@3.11
mise use python@3.11

# Step 2: Create virtual environment
echo "[2/6] Creating virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate

# Step 3: Upgrade pip
echo "[3/6] Upgrading pip..."
pip install --upgrade pip

# Step 4: Install core dependencies
echo "[4/6] Installing core dependencies..."
pip install -e ".[dev]"

# Step 5: Install ECG-Digitiser dependencies
echo "[5/6] Installing ECG-Digitiser dependencies..."
pip install -e ".[digitiser]"

# Step 6: Create folder structure
echo "[6/6] Creating folder structure..."
mkdir -p data/raw/ptb-xl
mkdir -p data/raw/mimic-iv-ecg
mkdir -p data/processed/signals
mkdir -p data/processed/images
mkdir -p data/splits
mkdir -p models/digitiser
mkdir -p models/ecgfounder/base
mkdir -p models/ecgfounder/finetuned
mkdir -p models/llm
mkdir -p src/pipeline
mkdir -p src/training
mkdir -p src/vtsvt
mkdir -p src/utils
mkdir -p src/web
mkdir -p notebooks
mkdir -p configs
mkdir -p scripts
mkdir -p results/metrics
mkdir -p results/figures
mkdir -p results/reports

# Add .gitkeep to empty directories
find data models results -type d -empty -exec touch {}/.gitkeep \;

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Activate environment:  source .venv/bin/activate"
echo "  2. Copy env file:         cp .env.example .env"
echo "  3. Download models:       ./scripts/download_models.sh"
echo "  4. Download PTB-XL:       ./scripts/download_ptbxl.sh"
