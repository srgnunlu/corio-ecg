# Phase 1: Pipeline Setup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Get ECGFounder (signal to diagnosis) working with PTB-XL data, then add ECG-Digitiser (image to signal), then connect into full pipeline.

**Architecture:** ECGFounder uses Net1D (RegNet-based 1D CNN, 76.3M params) for 150-class multi-label ECG diagnosis. ECG-Digitiser uses nnU-Net 2D segmentation + Hough Transform for paper ECG digitization. Both are chained: image to signal to diagnosis.

**Tech Stack:** Python 3.11, PyTorch >= 2.4.0, wfdb, numpy, scipy, huggingface_hub

---

## Key Technical Details (from ECGFounder source code)

### Net1D Model Parameters
```python
in_channels=12, base_filters=64, ratio=1,
filter_list=[64, 160, 160, 400, 400, 1024, 1024],
m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
kernel_size=16, stride=2, groups_width=16,
n_classes=150, use_bn=False, use_do=False
```

### Preprocessing
1. Read WFDB signal
2. Replace NaN with 0
3. Reorder leads to: I, II, III, aVR, aVL, aVF, V1-V6
4. Resample to 500 Hz if needed
5. Pad/truncate to 5000 samples (10 seconds)
6. Z-score normalize: `(signal - mean) / (std + 1e-8)`

### Inference
- Multi-label classification (sigmoid, NOT softmax)
- Output: 150 probabilities per ECG record

---

### Task 1: Download Scripts

**Files:**
- Create: `scripts/download_models.py`
- Create: `scripts/download_ptbxl.py`
- Modify: `pyproject.toml` (add huggingface_hub dependency)

Create scripts to download ECGFounder model from HuggingFace (~370 MB) and PTB-XL from PhysioNet (~3 GB). Use `huggingface_hub.hf_hub_download` for model and `curl` for PTB-XL zip. Add `.download_complete` marker files to avoid re-downloading. Add `huggingface_hub>=0.20.0` to pyproject.toml dependencies.

After creating, run both scripts to verify downloads work.

Commit: `feat: add download scripts for ECGFounder model and PTB-XL dataset`

---

### Task 2: ECG Labels Utility

**Files:**
- Create: `src/utils/ecg_labels.py`

Create a file containing the complete list of 150 ECGFounder diagnosis labels (from HEEDB/arXiv paper Table 6). Include constants: `ECG_FOUNDER_LABELS` (list of 150 strings), `LEAD_NAMES` (12-lead order), `NUM_CLASSES=150`, `DEFAULT_THRESHOLD=0.5`, and `CRITICAL_DIAGNOSIS_INDICES` (indices for AF, VT, STEMI, etc.).

The 150 labels in order (0-indexed):
0: ABNORMAL ECG, 1: NORMAL SINUS RHYTHM, 2: NORMAL ECG, 3: SINUS RHYTHM,
4: SINUS BRADYCARDIA, 5: ATRIAL FIBRILLATION, 6: SINUS TACHYCARDIA,
7: OTHERWISE NORMAL ECG, 8: LEFT AXIS DEVIATION,
9: PREMATURE VENTRICULAR COMPLEXES, 10: BORDERLINE ECG,
11: RIGHT BUNDLE BRANCH BLOCK, 12: SEPTAL INFARCT,
13: LEFT ATRIAL ENLARGEMENT, 14: NONSPECIFIC T WAVE ABNORMALITY,
15: LOW VOLTAGE QRS, 16: PREMATURE ATRIAL COMPLEXES,
17: ANTERIOR INFARCT, 18: INCOMPLETE RIGHT BUNDLE BRANCH BLOCK,
19: PREMATURE SUPRAVENTRICULAR COMPLEXES, 20: LEFT BUNDLE BRANCH BLOCK,
21-149: (see full list in design doc or arXiv 2410.04133 Table 6)

Commit: `feat: add ECGFounder 150 diagnosis labels`

---

### Task 3: Net1D Model Architecture

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/net1d.py`

Copy Net1D architecture from ECGFounder repo (MIT license). Classes needed:
- `MyConv1dPadSame` - Conv1d with SAME padding
- `MyMaxPool1dPadSame` - MaxPool1d with SAME padding
- `Swish` - activation function
- `BasicBlock` - bottleneck block with SE attention
- `BasicStage` - multiple BasicBlocks
- `Net1D` - main model class

Do NOT include MyDataset class from original (we have our own data loading).
Add type hints to all function signatures.

Commit: `feat: add Net1D model architecture from ECGFounder`

---

### Task 4: WFDB Helpers

**Files:**
- Create: `src/utils/wfdb_helpers.py`

Functions needed:
- `read_ecg_signal(record_path) -> (np.ndarray, dict)`: Read WFDB record, reorder leads, resample to 500Hz, pad/truncate to 5000 samples, z-score normalize. Returns shape (12, 5000).
- `_reorder_leads(signal, source_leads) -> np.ndarray`: Match ECGFounder lead order
- `_resample(signal, original_rate, target_rate) -> np.ndarray`: Linear interpolation
- `_pad_or_truncate(signal, target_length) -> np.ndarray`: Zero-pad or truncate
- `_z_score_normalize(signal) -> np.ndarray`: `(x - mean) / (std + 1e-8)`

Constants: `ECGFOUNDER_LEAD_ORDER`, `TARGET_SAMPLE_RATE=500`, `TARGET_LENGTH=5000`

Commit: `feat: add WFDB helper functions for ECG signal loading`

---

### Task 5: ECGFounder Diagnosis Module

**Files:**
- Create: `src/pipeline/diagnose.py`

Classes:
- `DiagnosisResult` (dataclass): label, index, probability
- `ECGDiagnoser`: loads Net1D model, runs inference

Key methods:
- `__init__(checkpoint_path, device, threshold)`: Load model with MODEL_CONFIG params
- `_load_model(path)`: Handle various checkpoint formats (state_dict, module prefix)
- `diagnose(signal, threshold) -> list[DiagnosisResult]`: Run inference with sigmoid
- `diagnose_all(signal) -> list[DiagnosisResult]`: Return all 150 predictions
- `get_device()`: Auto-detect CUDA > MPS > CPU

MODEL_CONFIG dict must exactly match the parameters from Task 3.

Commit: `feat: add ECGFounder diagnosis module`

---

### Task 6: Pipeline CLI Runner

**Files:**
- Create: `src/pipeline/run.py`

CLI interface: `python -m src.pipeline.run --signal <wfdb_path> --threshold 0.5 --output results.json`

Functions:
- `run_signal_diagnosis(record_path, model_path, threshold) -> list[DiagnosisResult]`
- `print_results(results)`: Pretty-print with probability bars
- `main()`: argparse CLI with --signal, --model, --threshold, --output flags

Commit: `feat: add pipeline CLI runner for signal diagnosis`

---

### Task 7: Baseline Evaluation on PTB-XL

**Files:**
- Create: `src/utils/metrics.py`
- Create: `src/training/evaluate.py`

metrics.py:
- `compute_auroc_per_class(y_true, y_prob) -> dict[int, float]`
- `compute_macro_auroc(auroc_per_class) -> float`

evaluate.py:
- `load_ptbxl_metadata(data_dir) -> pd.DataFrame`: Load CSV, parse scp_codes with `ast.literal_eval` (safe - only parses Python literals, no code execution)
- `evaluate_ptbxl(data_dir, model_path, max_samples) -> dict`: Run ECGFounder on PTB-XL test fold (strat_fold==10), save probabilities
- CLI: `python -m src.training.evaluate --max-samples 10`

Note: PTB-XL scp_codes column contains Python dict literals like `{'NORM': 100.0}`. The `ast.literal_eval` function is the standard safe parser for this format (unlike `eval()`, it cannot execute arbitrary code).

Commit: `feat: add baseline evaluation on PTB-XL`

---

### Task 8: Integration Test

No new files. Run these commands in sequence:

1. `pip install -e "."` - install project
2. `python scripts/download_models.py` - download ECGFounder
3. `python scripts/download_ptbxl.py` - download PTB-XL
4. `python -m src.pipeline.run --signal data/raw/ptb-xl/records500/00000/00001_hr --threshold 0.3` - single record test
5. `python -m src.training.evaluate --max-samples 10` - quick baseline

Fix any issues found during testing.

Commit: `feat: verified end-to-end pipeline on PTB-XL`

---

### Task 9: ECG-Digitiser Placeholder

**Files:**
- Create: `src/pipeline/digitize.py`

Create placeholder `ECGDigitiser` class with `digitize(image_path) -> np.ndarray` method that raises `NotImplementedError` with setup instructions. Full implementation will come later when we clone ECG-Digitiser repo and set up nnU-Net.

Commit: `feat: add ECG-Digitiser placeholder module`

---

### Task 10: Barrel Exports

**Files:**
- Modify: `src/pipeline/__init__.py`
- Modify: `src/utils/__init__.py`
- Modify: `src/models/__init__.py`

Update __init__.py files to re-export key classes:
- pipeline: ECGDiagnoser, DiagnosisResult, ECGDigitiser
- utils: ECG_FOUNDER_LABELS, LEAD_NAMES, NUM_CLASSES, read_ecg_signal
- models: Net1D

Verify: `python -c "from src.pipeline import ECGDiagnoser; print('OK')"`

Commit: `chore: update barrel exports for pipeline, utils, models`

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | Download scripts | `scripts/download_models.py`, `scripts/download_ptbxl.py` |
| 2 | ECG labels | `src/utils/ecg_labels.py` |
| 3 | Net1D model | `src/models/net1d.py` |
| 4 | WFDB helpers | `src/utils/wfdb_helpers.py` |
| 5 | Diagnosis module | `src/pipeline/diagnose.py` |
| 6 | Pipeline CLI | `src/pipeline/run.py` |
| 7 | Baseline evaluation | `src/training/evaluate.py`, `src/utils/metrics.py` |
| 8 | Integration test | Run end-to-end on PTB-XL |
| 9 | Digitiser placeholder | `src/pipeline/digitize.py` |
| 10 | Barrel exports | All `__init__.py` files |
