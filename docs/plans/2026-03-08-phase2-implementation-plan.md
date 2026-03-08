# Phase 2: Digitization Robustness — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate ECG-Digitiser and ECG-Image-Kit, then measure diagnostic accuracy loss through the paper→digital round-trip.

**Architecture:** PTB-XL clean signals serve as ground truth. ECG-Image-Kit renders them as synthetic paper ECG images. ECG-Digitiser converts images back to signals. ECGFounder diagnoses both clean and digitized signals. We compare AUROC to quantify the "digitization gap."

**Tech Stack:** Python 3.12, PyTorch (MPS), TensorFlow 2.14 (ECG-Digitiser), nnU-Net 2D, ECG-Image-Kit, wfdb, numpy, matplotlib

---

## Task 1: Setup External Dependencies

**Files:**
- Modify: `.gitignore` — add `external/` directory
- Create: `scripts/setup_phase2.py` — automated setup script

**Step 1: Update .gitignore**

Add to `.gitignore`:
```
# External repos (cloned, not committed)
external/
```

**Step 2: Clone ECG-Digitiser**

```bash
cd "/Users/sergenunlu/Developer/active/apps/Corio ECG"
mkdir -p external
git clone https://github.com/felixkrones/ECG-Digitiser.git external/ecg-digitiser
cd external/ecg-digitiser && git lfs install && git lfs pull
```

Expected: `external/ecg-digitiser/` with `models/M3/` containing nnU-Net weights.

**Step 3: Install ECG-Digitiser's custom nnU-Net fork**

```bash
cd "/Users/sergenunlu/Developer/active/apps/Corio ECG"
source .venv/bin/activate
cd external/ecg-digitiser/nnUNet && pip install -e . && cd ../../..
```

**Step 4: Install ECG-Digitiser's other dependencies**

```bash
pip install tensorflow==2.14.0 keras==2.14.0 scikit-image>=0.23.2 imgaug>=0.4.0 imageio>=2.34.1 imutils>=0.5.4
```

**Step 5: Clone ECG-Image-Kit**

```bash
git clone https://github.com/alphanumericslab/ecg-image-kit.git external/ecg-image-kit
cd external/ecg-image-kit && pip install -r requirements.txt && cd ../..
```

**Step 6: Create setup script for reproducibility**

Create `scripts/setup_phase2.py` that automates all of the above steps.

**Step 7: Verify installations**

```bash
python -c "import tensorflow; print(f'TF {tensorflow.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import nnunetv2; print('nnU-Net OK')"
```

**Step 8: Commit**

```bash
git add .gitignore scripts/setup_phase2.py
git commit -m "chore: add Phase 2 setup script and gitignore external/"
```

---

## Task 2: Explore ECG-Digitiser API

**Purpose:** Before writing our wrapper, we need to understand ECG-Digitiser's actual Python API by reading its source code. This is a research task — no code written yet.

**Step 1: Read ECG-Digitiser entry point**

Read these files in `external/ecg-digitiser/`:
- `src/run/digitize.py` — main entry point
- `src/utils/helper_code.py` — data I/O functions
- `README.md` — usage instructions

**Step 2: Document the API**

Note down:
- How to load the nnU-Net model programmatically
- How to run inference on a single image
- What preprocessing happens (Hough transform, rotation, etc.)
- What the output format looks like (WFDB fields, lead order, sample rate)
- How to convert output to numpy array (12, 5000)

**Step 3: Check compatibility**

Verify that ECG-Digitiser's output can feed directly into our `wfdb_helpers.py` preprocessing or if we need an adapter.

---

## Task 3: Implement ECGDigitiser Wrapper

**Files:**
- Modify: `src/pipeline/digitize.py` — replace placeholder with real implementation
- Create: `tests/test_digitize.py` — integration test

**Step 1: Write the integration test**

```python
# tests/test_digitize.py
# Integration test for ECG-Digitiser wrapper — requires model weights

import numpy as np
import pytest
from pathlib import Path

from src.pipeline.digitize import ECGDigitiser

DIGITISER_MODEL_DIR = Path("external/ecg-digitiser/models/M3")
# Use a sample synthetic ECG image for testing (created in Task 5)
SAMPLE_IMAGE = Path("tests/fixtures/sample_ecg.png")


@pytest.mark.skipif(
    not DIGITISER_MODEL_DIR.exists(),
    reason="ECG-Digitiser model not downloaded",
)
@pytest.mark.skipif(
    not SAMPLE_IMAGE.exists(),
    reason="Sample ECG image not available",
)
class TestECGDigitiser:
    def test_digitize_returns_correct_shape(self):
        digitiser = ECGDigitiser(model_dir=DIGITISER_MODEL_DIR)
        signal = digitiser.digitize(SAMPLE_IMAGE)
        assert signal.shape == (12, 5000)

    def test_digitize_returns_float_array(self):
        digitiser = ECGDigitiser(model_dir=DIGITISER_MODEL_DIR)
        signal = digitiser.digitize(SAMPLE_IMAGE)
        assert signal.dtype in (np.float32, np.float64)

    def test_digitize_signal_not_all_zeros(self):
        digitiser = ECGDigitiser(model_dir=DIGITISER_MODEL_DIR)
        signal = digitiser.digitize(SAMPLE_IMAGE)
        assert np.any(signal != 0), "Digitized signal should not be all zeros"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_digitize.py -v
```
Expected: FAIL (placeholder raises NotImplementedError)

**Step 3: Implement ECGDigitiser wrapper**

Update `src/pipeline/digitize.py`:
- Import ECG-Digitiser modules from `external/ecg-digitiser/`
- Add `external/ecg-digitiser` to `sys.path` if needed
- Implement `digitize()` method:
  1. Load image
  2. Run Hough transform preprocessing (rotation/alignment)
  3. Run nnU-Net segmentation
  4. Extract signal from segmentation mask
  5. Resample to 500 Hz, pad/truncate to 5000 samples
  6. Return numpy array (12, 5000)

**Important:** The exact implementation depends on Task 2 findings. The wrapper should call ECG-Digitiser's functions, NOT reimplement them.

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_digitize.py -v
```
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/pipeline/digitize.py tests/test_digitize.py
git commit -m "feat: integrate ECG-Digitiser for image-to-signal conversion"
```

---

## Task 4: Explore ECG-Image-Kit API

**Purpose:** Understand how ECG-Image-Kit generates synthetic paper ECG images from signals.

**Step 1: Read ECG-Image-Kit source**

Read these files in `external/ecg-image-kit/`:
- Main generator module
- Configuration/parameter options
- Example scripts

**Step 2: Document the API**

Note down:
- How to generate an image from a numpy array or WFDB file
- Available augmentation parameters (grid, noise, wrinkles, etc.)
- Output format and resolution options
- How to configure the 3x4 + rhythm strip layout

---

## Task 5: Implement Synthetic Image Generator

**Files:**
- Create: `src/utils/ecg_render.py` — ECG-Image-Kit wrapper
- Create: `tests/test_ecg_render.py` — unit tests

**Step 1: Write failing tests**

```python
# tests/test_ecg_render.py
# Tests for synthetic ECG image generation

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from src.utils.ecg_render import render_ecg_image, DifficultyLevel


class TestRenderECGImage:
    def setup_method(self):
        """Create a dummy 12-lead signal for testing."""
        # Simple sine waves simulating ECG leads
        t = np.linspace(0, 10, 5000)
        self.signal = np.stack([np.sin(2 * np.pi * t + i) for i in range(12)])

    def test_render_returns_path(self, tmp_path):
        output = tmp_path / "test_ecg.png"
        result = render_ecg_image(self.signal, output, DifficultyLevel.CLEAN)
        assert result == output
        assert output.exists()

    def test_render_creates_valid_image(self, tmp_path):
        output = tmp_path / "test_ecg.png"
        render_ecg_image(self.signal, output, DifficultyLevel.CLEAN)
        img = Image.open(output)
        assert img.size[0] > 0 and img.size[1] > 0

    def test_render_all_difficulty_levels(self, tmp_path):
        for level in DifficultyLevel:
            output = tmp_path / f"test_{level.value}.png"
            render_ecg_image(self.signal, output, level)
            assert output.exists()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_ecg_render.py -v
```

**Step 3: Implement ecg_render.py**

Create `src/utils/ecg_render.py`:
- `DifficultyLevel` enum: CLEAN, MODERATE, HARD
- `render_ecg_image(signal, output_path, difficulty)` function
- Wraps ECG-Image-Kit with preset configurations for each difficulty level
- Input: numpy array (12, 5000) at 500 Hz
- Output: PNG image file path

**Difficulty presets:**
- CLEAN: Standard grid, no distortions, 300 DPI
- MODERATE: Grid + light Gaussian noise + small rotation (1-3 degrees) + paper texture
- HARD: Grid + wrinkles + perspective distortion + handwritten text + heavy noise

**Step 4: Run tests**

```bash
pytest tests/test_ecg_render.py -v
```

**Step 5: Commit**

```bash
git add src/utils/ecg_render.py tests/test_ecg_render.py
git commit -m "feat: add synthetic ECG image generator with 3 difficulty levels"
```

---

## Task 6: Batch Image Generation Script

**Files:**
- Create: `scripts/generate_synthetic_images.py`

**Step 1: Create the script**

This script:
1. Loads PTB-XL test fold metadata (strat_fold == 10)
2. For each record, reads the WFDB signal
3. Generates 3 images (clean/moderate/hard)
4. Saves to `data/processed/images/{clean,moderate,hard}/{ecg_id}.png`
5. Shows progress bar (tqdm)
6. Accepts `--max-samples` for quick testing

```bash
python scripts/generate_synthetic_images.py --max-samples 5
```

**Step 2: Run with 5 samples to verify**

```bash
python scripts/generate_synthetic_images.py --max-samples 5
ls data/processed/images/clean/ | wc -l  # Should be 5
```

**Step 3: Commit**

```bash
git add scripts/generate_synthetic_images.py
git commit -m "feat: add batch synthetic ECG image generation script"
```

---

## Task 7: Batch Digitization Script

**Files:**
- Create: `scripts/digitize_synthetic_images.py`

**Step 1: Create the script**

This script:
1. Takes a difficulty level as argument
2. Reads all PNG images from `data/processed/images/{level}/`
3. Runs ECG-Digitiser on each image
4. Saves digitized signals as numpy arrays to `data/processed/signals/{level}/{ecg_id}.npy`
5. Also saves a WFDB format version for inspection
6. Shows progress bar

```bash
python scripts/digitize_synthetic_images.py --level clean --max-samples 5
```

**Step 2: Run with 5 clean samples to verify**

```bash
python scripts/digitize_synthetic_images.py --level clean --max-samples 5
ls data/processed/signals/clean/ | wc -l
```

**Step 3: Commit**

```bash
git add scripts/digitize_synthetic_images.py
git commit -m "feat: add batch digitization script for synthetic images"
```

---

## Task 8: Signal Quality Metrics

**Files:**
- Modify: `src/utils/metrics.py` — add signal quality metrics
- Create: `tests/test_metrics.py` — unit tests

**Step 1: Write failing tests**

```python
# tests/test_metrics.py
import numpy as np
from src.utils.metrics import compute_snr, compute_pearson_per_lead


class TestSignalQualityMetrics:
    def test_snr_identical_signals(self):
        """SNR should be very high for identical signals."""
        signal = np.random.randn(12, 5000)
        snr = compute_snr(signal, signal)
        assert snr > 50  # essentially infinite

    def test_snr_noisy_signal(self):
        """SNR should decrease with more noise."""
        clean = np.sin(np.linspace(0, 10, 5000))
        clean = np.stack([clean] * 12)
        noisy = clean + np.random.randn(12, 5000) * 0.1
        snr = compute_snr(clean, noisy)
        assert 10 < snr < 30

    def test_pearson_identical_signals(self):
        """Pearson correlation should be 1.0 for identical signals."""
        signal = np.random.randn(12, 5000)
        correlations = compute_pearson_per_lead(signal, signal)
        assert len(correlations) == 12
        for r in correlations:
            assert abs(r - 1.0) < 1e-6

    def test_pearson_uncorrelated(self):
        """Correlation should be near 0 for random signals."""
        np.random.seed(42)
        signal_a = np.random.randn(12, 5000)
        signal_b = np.random.randn(12, 5000)
        correlations = compute_pearson_per_lead(signal_a, signal_b)
        for r in correlations:
            assert abs(r) < 0.1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_metrics.py -v
```

**Step 3: Implement signal quality metrics**

Add to `src/utils/metrics.py`:
- `compute_snr(clean, digitized)` — Signal-to-Noise Ratio in dB
- `compute_pearson_per_lead(clean, digitized)` — Pearson correlation for each of 12 leads

**Step 4: Run tests**

```bash
pytest tests/test_metrics.py -v
```

**Step 5: Commit**

```bash
git add src/utils/metrics.py tests/test_metrics.py
git commit -m "feat: add signal quality metrics (SNR, Pearson correlation)"
```

---

## Task 9: Round-Trip Evaluation Script

**Files:**
- Create: `src/training/evaluate_roundtrip.py`

**Step 1: Create the evaluation script**

This script compares diagnosis performance across 4 scenarios:
1. **Baseline:** Clean PTB-XL signals → ECGFounder
2. **Clean roundtrip:** Clean images → digitized → ECGFounder
3. **Moderate roundtrip:** Moderate images → digitized → ECGFounder
4. **Hard roundtrip:** Hard images → digitized → ECGFounder

For each scenario, it computes:
- Macro AUROC (across all 150 classes)
- Per-class AUROC (for top diagnoses)
- Signal quality (SNR, Pearson) for digitized scenarios
- Confidence intervals via bootstrap

Input: Pre-generated signals from `data/processed/signals/`
Output: `results/metrics/roundtrip_comparison.json`

CLI:
```bash
python -m src.training.evaluate_roundtrip --max-samples 20
```

**Step 2: Run with 20 samples for quick test**

```bash
python -m src.training.evaluate_roundtrip --max-samples 20
cat results/metrics/roundtrip_comparison.json
```

**Step 3: Commit**

```bash
git add src/training/evaluate_roundtrip.py
git commit -m "feat: add round-trip evaluation comparing clean vs digitized diagnosis"
```

---

## Task 10: Visualization & Results

**Files:**
- Create: `src/utils/plot_results.py` — comparison charts
- Create: `scripts/run_full_evaluation.py` — orchestrates the full pipeline

**Step 1: Create plotting utilities**

`src/utils/plot_results.py`:
- Bar chart: Macro AUROC across 4 scenarios
- Box plot: Per-lead SNR distribution
- Scatter plot: Per-class AUROC (clean vs digitized)
- Save to `results/figures/`

**Step 2: Create full evaluation orchestrator**

`scripts/run_full_evaluation.py`:
1. Generate synthetic images (all difficulty levels)
2. Digitize all images
3. Run round-trip evaluation
4. Generate comparison plots
5. Print summary

```bash
python scripts/run_full_evaluation.py --max-samples 50
```

**Step 3: Commit**

```bash
git add src/utils/plot_results.py scripts/run_full_evaluation.py
git commit -m "feat: add visualization and full evaluation pipeline for Phase 2"
```

---

## Task 11: Full PTB-XL Test Run

**Step 1: Generate images for full test fold (~2000 records)**

```bash
python scripts/generate_synthetic_images.py
```

**Step 2: Digitize all images**

```bash
python scripts/digitize_synthetic_images.py --level clean
python scripts/digitize_synthetic_images.py --level moderate
python scripts/digitize_synthetic_images.py --level hard
```

**Step 3: Run full evaluation**

```bash
python -m src.training.evaluate_roundtrip
```

**Step 4: Generate figures**

```bash
python scripts/run_full_evaluation.py --skip-generation --skip-digitization
```

**Step 5: Review results and commit**

```bash
git add results/metrics/ results/figures/
git commit -m "feat: Phase 2 round-trip evaluation results on full PTB-XL test fold"
```

---

## Execution Order & Dependencies

```
Task 1 (Setup)
    |
    +---> Task 2 (Explore Digitiser API)
    |         |
    |         +---> Task 3 (Digitiser Wrapper)
    |
    +---> Task 4 (Explore Image-Kit API)
              |
              +---> Task 5 (Image Generator)
                        |
                        +---> Task 6 (Batch Image Script)
                                    |
Task 8 (Signal Metrics) ----------+---> Task 7 (Batch Digitize Script)
                                            |
                                            +---> Task 9 (Round-Trip Eval)
                                                        |
                                                        +---> Task 10 (Visualization)
                                                                    |
                                                                    +---> Task 11 (Full Run)
```

**Parallelizable:** Tasks 2+4 can run in parallel. Task 8 is independent and can run anytime.
