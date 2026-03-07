# Phase 1: Pipeline Setup Design

## Goal
Get ECG-Digitiser + ECGFounder working as a connected pipeline.
Input: paper ECG photo → Output: list of diagnoses with probabilities.

## Approach
Build ECGFounder (signal→diagnosis) first, then ECG-Digitiser (image→signal), then connect.

## Components

### 1. ECGFounder Diagnosis Module (`src/pipeline/diagnose.py`)
- Load Net1D model from HuggingFace checkpoint
- Input: 12-lead ECG signal as numpy array (12 x 5000, 500 Hz)
- Z-score normalize each lead
- Run inference → 150-class sigmoid probabilities
- Return diagnoses above threshold (default 0.5)
- Device: auto-detect (CUDA > MPS > CPU)

### 2. ECG-Digitiser Module (`src/pipeline/digitize.py`)
- Load nnU-Net 2D segmentation model
- Input: PNG/JPG paper ECG image
- Segment leads via nnU-Net
- Extract signals via Hough Transform + pixel-to-mV conversion
- Output: WFDB format (.dat + .hea), 12-lead, 500 Hz

### 3. Pipeline Orchestration (`src/pipeline/run.py`)
- CLI entry point: `python -m src.pipeline.run --image path.png --output dir/`
- Chain: digitize() → diagnose() → print results
- Support signal-only mode (skip digitization, use WFDB directly)

### 4. Utilities
- `src/utils/wfdb_helpers.py` — WFDB read/write operations
- `src/utils/ecg_labels.py` — 150 diagnosis labels from ECGFounder

### 5. Scripts
- `scripts/download_models.py` — Download ECGFounder from HuggingFace
- `scripts/download_ptbxl.py` — Download PTB-XL from PhysioNet

### 6. Baseline Evaluation
- Run ECGFounder on PTB-XL test set
- Metrics: per-class AUROC, macro AUROC, sensitivity/specificity for critical diagnoses
- Save results to `results/metrics/`

## Order of Implementation
1. Download scripts (model + data)
2. ECG labels utility
3. WFDB helpers
4. ECGFounder diagnose module
5. Baseline evaluation on PTB-XL
6. ECG-Digitiser digitize module
7. Pipeline orchestration (run.py)
