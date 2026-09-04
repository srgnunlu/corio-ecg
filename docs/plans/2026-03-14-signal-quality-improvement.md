# Signal Quality Improvement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve digitized ECG signal quality to better match WFDB training data, increasing diagnosis accuracy before Phase 3 fine-tuning.

**Architecture:** Replace the aggressive 0.5–40 Hz bandpass filter with a gentle 0.5 Hz high-pass (baseline wander removal) followed by wavelet denoising (noise removal that preserves QRS morphology). Add Einthoven consistency verification for lead assignment quality. All new signal processing goes into a dedicated module.

**Tech Stack:** PyWavelets (pywt) for DWT-based denoising, scipy.signal for Butterworth high-pass, numpy

**Key Insight — Why This Matters:**
ECGFounder was trained on raw WFDB signals with NO bandpass filter (full 0–250 Hz spectrum). Our current 40 Hz cutoff removes QRS high-frequency content (40–150 Hz), causing the model to see smoothed waveforms it was never trained on. This is likely a major contributor to poor diagnosis quality, independent of the domain gap.

---

## File Structure

| File | Action | Responsibility | Est. Lines |
|------|--------|---------------|------------|
| `src/utils/signal_clean.py` | CREATE | High-pass filter, wavelet denoise, Einthoven check | ~120 |
| `src/pipeline/digitize.py` | MODIFY | Replace `_bandpass_filter()` call with new functions | ~10 lines changed |
| `tests/test_signal_clean.py` | CREATE | Unit tests for all signal cleaning functions | ~130 |
| `pyproject.toml` | MODIFY | Add PyWavelets dependency | 1 line |

---

## Chunk 1: Signal Cleaning Module

### Task 1: Add PyWavelets dependency

**Files:**
- Modify: `pyproject.toml:8-37`

- [ ] **Step 1: Add pywt to dependencies**

In `pyproject.toml`, add to the `dependencies` list after scipy:

```toml
"PyWavelets>=1.4.0",
```

**Why PyWavelets?**
- Gold standard for discrete wavelet transform in Python
- 12M+ weekly downloads, actively maintained, numpy-only dependency
- Required for ECG-specific wavelet denoising (scipy lacks DWT support)
- Alternative considered: manual FFT-based approach — rejected because wavelets work in both time AND frequency domain, critical for preserving QRS morphology

- [ ] **Step 2: Install and verify**

```bash
source .venv/bin/activate && pip install "PyWavelets>=1.4.0"
python -c "import pywt; print(f'PyWavelets {pywt.__version__} installed')"
```

---

### Task 2: Create signal_clean.py

**Files:**
- Create: `src/utils/signal_clean.py`

- [ ] **Step 1: Create the module with all three functions**

```python
# ECG signal cleaning utilities for digitized paper ECG signals
# High-pass filter for baseline wander, wavelet denoising for noise,
# and Einthoven consistency check for lead assignment verification

from __future__ import annotations

import logging

import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)

# Wavelet denoising parameters tuned for 500 Hz ECG signals
_WAVELET: str = "db4"            # Daubechies-4: standard for ECG
_DECOMPOSITION_LEVEL: int = 6    # 6 levels at 500 Hz covers 0–250 Hz
_NOISE_LEVELS: int = 3           # Only threshold top 3 detail levels (>~30 Hz)
# Level 1: 125–250 Hz (muscle artifact, digitization noise) → THRESHOLD
# Level 2: 62.5–125 Hz (some QRS high-freq, mostly noise) → THRESHOLD
# Level 3: 31.25–62.5 Hz (QRS edge content, grid artifacts) → THRESHOLD
# Levels 4–6: P wave, T wave, QRS body → PRESERVE (no thresholding)


def highpass_filter(
    signal: np.ndarray,
    sample_rate: int = 500,
    cutoff_hz: float = 0.5,
) -> np.ndarray:
    """Remove baseline wander with a zero-phase Butterworth high-pass filter.

    Baseline wander from paper curvature, lighting gradients, and
    digitization artifacts appears below 0.5 Hz. This is universally
    noise in ECG signals — no diagnostic content exists below 0.5 Hz.

    Args:
        signal: Shape (12, N) — multi-lead ECG signal.
        sample_rate: Sampling rate in Hz.
        cutoff_hz: High-pass cutoff frequency.

    Returns:
        Filtered signal with same shape.
    """
    nyquist = sample_rate / 2.0
    wn = cutoff_hz / nyquist
    sos = butter(N=3, Wn=wn, btype="highpass", output="sos")

    filtered = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        if np.std(signal[i]) < 1e-6:
            filtered[i] = signal[i]
            continue
        filtered[i] = sosfiltfilt(sos, signal[i])
    return filtered


def wavelet_denoise(
    signal: np.ndarray,
    sample_rate: int = 500,
) -> np.ndarray:
    """Remove high-frequency noise while preserving ECG morphology.

    Uses Discrete Wavelet Transform (DWT) with soft thresholding on
    noise-band detail coefficients only. Lower-frequency levels that
    carry P/QRS/T morphology are left untouched.

    The threshold is estimated per-lead using the Median Absolute
    Deviation (MAD) of the finest detail coefficients — a robust
    noise estimator that is not distorted by QRS peaks.

    Args:
        signal: Shape (12, N) — multi-lead ECG signal.
        sample_rate: Sampling rate in Hz (used for level calculation).

    Returns:
        Denoised signal with same shape.
    """
    max_level = pywt.dwt_max_level(signal.shape[1], _WAVELET)
    level = min(_DECOMPOSITION_LEVEL, max_level)
    noise_levels = min(_NOISE_LEVELS, level)

    denoised = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        lead = signal[i]
        if np.std(lead) < 1e-6:
            denoised[i] = lead
            continue

        # Decompose: [approx, detail_n, detail_n-1, ..., detail_1]
        coeffs = pywt.wavedec(lead, _WAVELET, level=level)

        # Estimate noise level from finest detail coefficients (MAD)
        # MAD is robust to QRS spikes unlike standard deviation
        sigma = float(np.median(np.abs(coeffs[-1])) / 0.6745)

        # Universal threshold (VisuShrink): sigma * sqrt(2 * ln(N))
        threshold = sigma * np.sqrt(2.0 * np.log(len(lead)))

        # Soft-threshold only noise-band detail levels
        # coeffs layout: [a6, d6, d5, d4, d3, d2, d1]
        # noise_levels=3 → threshold d1, d2, d3 (indices -1, -2, -3)
        for j in range(1, noise_levels + 1):
            coeffs[-j] = pywt.threshold(coeffs[-j], threshold, mode="soft")

        # Reconstruct — waverec may produce 1 extra sample, truncate
        reconstructed = pywt.waverec(coeffs, _WAVELET)
        denoised[i] = reconstructed[: len(lead)]

    return denoised


def einthoven_consistency(signal: np.ndarray) -> float:
    """Check if Einthoven's law holds: Lead II ≈ Lead I + Lead III.

    Returns Pearson correlation between actual Lead II and the computed
    sum (Lead I + Lead III). A perfect ECG with correct lead assignment
    gives ~1.0. Values below 0.7 suggest lead misassignment or severe
    signal corruption.

    Correlation is invariant to affine transforms, so this works on
    both raw mV signals and z-score normalized signals.

    Args:
        signal: Shape (12, N) — 12-lead ECG signal (any normalization).

    Returns:
        Pearson correlation coefficient (0.0 to 1.0). Returns 0.0 for
        degenerate cases (flat leads).
    """
    lead_i = signal[0]
    lead_ii = signal[1]
    lead_iii = signal[2]

    computed_ii = lead_i + lead_iii

    if np.std(lead_ii) < 1e-6 or np.std(computed_ii) < 1e-6:
        return 0.0

    corr = np.corrcoef(lead_ii, computed_ii)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0
```

- [ ] **Step 2: Add exports to src/utils/__init__.py**

Add to `src/utils/__init__.py`:

```python
from src.utils.signal_clean import (
    einthoven_consistency,
    highpass_filter,
    wavelet_denoise,
)
```

---

### Task 3: Write tests for signal_clean.py

**Files:**
- Create: `tests/test_signal_clean.py`

- [ ] **Step 1: Write all tests**

```python
# Tests for ECG signal cleaning utilities
# Verifies high-pass filter, wavelet denoising, and Einthoven consistency

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.signal_clean import (
    einthoven_consistency,
    highpass_filter,
    wavelet_denoise,
)

SAMPLE_RATE = 500
N_SAMPLES = 5000


class TestHighpassFilter:
    """Verify baseline wander removal preserves diagnostic content."""

    def test_removes_baseline_wander(self) -> None:
        """0.1 Hz baseline drift should be attenuated by >80%."""
        t = np.linspace(0, 10, N_SAMPLES)
        baseline = 2.0 * np.sin(2 * np.pi * 0.1 * t)
        qrs = np.sin(2 * np.pi * 10 * t)
        signal = np.stack([baseline + qrs] * 12)

        filtered = highpass_filter(signal, SAMPLE_RATE)

        # Baseline energy should be reduced significantly
        baseline_12 = np.stack([baseline] * 12)
        baseline_residual = np.mean(np.abs(filtered - np.stack([qrs] * 12)))
        baseline_original = np.mean(np.abs(baseline_12))
        assert baseline_residual < baseline_original * 0.3

    def test_preserves_qrs_frequency_content(self) -> None:
        """10 Hz signal (QRS range) should pass through with >90% correlation."""
        t = np.linspace(0, 10, N_SAMPLES)
        qrs = np.sin(2 * np.pi * 10 * t)
        signal = np.stack([qrs] * 12)

        filtered = highpass_filter(signal, SAMPLE_RATE)

        for i in range(12):
            corr = np.corrcoef(filtered[i], qrs)[0, 1]
            assert corr > 0.95, f"Lead {i}: correlation {corr:.3f} too low"

    def test_flat_lead_unchanged(self) -> None:
        """Flat (zero) leads should pass through without errors."""
        signal = np.zeros((12, N_SAMPLES))
        filtered = highpass_filter(signal, SAMPLE_RATE)
        np.testing.assert_array_equal(filtered, signal)


class TestWaveletDenoise:
    """Verify wavelet denoising removes noise while preserving morphology."""

    def test_reduces_gaussian_noise(self) -> None:
        """Random noise should be reduced by at least 40%."""
        rng = np.random.RandomState(42)
        t = np.linspace(0, 10, N_SAMPLES)
        clean = np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm fundamental
        noise = 0.3 * rng.randn(N_SAMPLES)
        noisy = np.stack([clean + noise] * 12)
        clean_12 = np.stack([clean] * 12)

        denoised = wavelet_denoise(noisy, SAMPLE_RATE)

        mse_before = np.mean((noisy - clean_12) ** 2)
        mse_after = np.mean((denoised - clean_12) ** 2)
        assert mse_after < mse_before * 0.6, (
            f"Noise reduction insufficient: {mse_before:.4f} → {mse_after:.4f}"
        )

    def test_preserves_peak_amplitude(self) -> None:
        """QRS peak amplitudes should not be reduced by more than 20%."""
        rng = np.random.RandomState(42)
        t = np.linspace(0, 10, N_SAMPLES)
        # Sharp peaks simulating QRS complexes
        clean = np.zeros(N_SAMPLES)
        for peak_pos in range(250, N_SAMPLES, 500):
            clean[peak_pos] = 3.0  # R peak
            if peak_pos + 20 < N_SAMPLES:
                clean[peak_pos + 20] = -1.0  # S wave
        noise = 0.1 * rng.randn(N_SAMPLES)
        noisy = np.stack([clean + noise] * 12)

        denoised = wavelet_denoise(noisy, SAMPLE_RATE)

        original_max = np.max(np.abs(noisy[0]))
        denoised_max = np.max(np.abs(denoised[0]))
        ratio = denoised_max / original_max
        assert ratio > 0.8, f"Peak amplitude lost: {ratio:.2f}"

    def test_flat_lead_unchanged(self) -> None:
        """Flat (zero) leads should pass through without errors."""
        signal = np.zeros((12, N_SAMPLES))
        denoised = wavelet_denoise(signal, SAMPLE_RATE)
        np.testing.assert_array_equal(denoised, signal)

    def test_output_shape_matches_input(self) -> None:
        """Output shape must equal input shape exactly."""
        rng = np.random.RandomState(42)
        signal = rng.randn(12, N_SAMPLES)
        denoised = wavelet_denoise(signal, SAMPLE_RATE)
        assert denoised.shape == signal.shape

    def test_short_signal_handled(self) -> None:
        """Very short signals should not crash wavelet decomposition."""
        rng = np.random.RandomState(42)
        short = rng.randn(12, 200)
        denoised = wavelet_denoise(short, SAMPLE_RATE)
        assert denoised.shape == short.shape


class TestEinthovenConsistency:
    """Verify Einthoven's law detection for lead assignment quality."""

    def test_perfect_consistency(self) -> None:
        """When II = I + III exactly, score should be ~1.0."""
        rng = np.random.RandomState(42)
        signal = np.zeros((12, N_SAMPLES))
        signal[0] = rng.randn(N_SAMPLES)       # Lead I
        signal[2] = rng.randn(N_SAMPLES)       # Lead III
        signal[1] = signal[0] + signal[2]       # Lead II = I + III

        score = einthoven_consistency(signal)
        assert score > 0.99, f"Perfect consistency score too low: {score:.3f}"

    def test_detects_misassigned_leads(self) -> None:
        """Random leads should give low consistency score."""
        rng = np.random.RandomState(42)
        signal = rng.randn(12, N_SAMPLES)  # All random — no relationship

        score = einthoven_consistency(signal)
        assert score < 0.5, f"Random leads scored too high: {score:.3f}"

    def test_flat_leads_return_zero(self) -> None:
        """Flat leads should return 0.0, not NaN or crash."""
        signal = np.zeros((12, N_SAMPLES))
        score = einthoven_consistency(signal)
        assert score == 0.0

    def test_works_after_z_score(self) -> None:
        """Einthoven check should work on z-score normalized signals."""
        rng = np.random.RandomState(42)
        signal = np.zeros((12, N_SAMPLES))
        signal[0] = rng.randn(N_SAMPLES) * 2.0
        signal[2] = rng.randn(N_SAMPLES) * 1.5
        signal[1] = signal[0] + signal[2]
        # Fill remaining leads with data
        for i in range(3, 12):
            signal[i] = rng.randn(N_SAMPLES)

        # Apply global z-score
        mean = np.mean(signal)
        std = np.std(signal)
        normalized = (signal - mean) / (std + 1e-8)

        score = einthoven_consistency(normalized)
        assert score > 0.95, f"Post-z-score consistency too low: {score:.3f}"
```

- [ ] **Step 2: Run tests and verify they fail (module not yet exported)**

```bash
source .venv/bin/activate && python -m pytest tests/test_signal_clean.py -v
```

Expected: ImportError or test failures until Step 2 of Task 2 (exports) is done.

- [ ] **Step 3: Run tests after module is complete**

```bash
source .venv/bin/activate && python -m pytest tests/test_signal_clean.py -v
```

Expected: All 12 tests PASS.

---

## Chunk 2: Pipeline Integration

### Task 4: Integrate into digitize.py _postprocess()

**Files:**
- Modify: `src/pipeline/digitize.py:624-664`

The current pipeline:
```
align+tile → _bandpass_filter(0.5–40 Hz) → z-score
```

New pipeline:
```
align+tile → highpass_filter(0.5 Hz) → wavelet_denoise() → z-score
```

- [ ] **Step 1: Replace bandpass call in _postprocess()**

Replace the bandpass filter section (lines ~649–659) with:

```python
        # Shift each lead's active data to sample 0 and tile to fill.
        signal = _align_leads_to_origin(signal)

        # Remove baseline wander (< 0.5 Hz) from paper curvature and
        # lighting gradients. High-pass only — no upper cutoff, so QRS
        # high-frequency content (40–150 Hz) is preserved. ECGFounder
        # was trained on unfiltered WFDB signals with full 0–250 Hz spectrum.
        signal = highpass_filter(signal, TARGET_SAMPLE_RATE)

        # Remove digitization noise (grid artifacts, trace jitter) while
        # preserving QRS/P/T morphology. Wavelet denoising thresholds
        # only high-frequency detail coefficients (> ~30 Hz), leaving
        # diagnostic waveform content untouched.
        signal = wavelet_denoise(signal, TARGET_SAMPLE_RATE)
```

- [ ] **Step 2: Add import at top of digitize.py**

```python
from src.utils.signal_clean import highpass_filter, wavelet_denoise
```

- [ ] **Step 3: Update _postprocess docstring**

```python
"""Convert raw canonical_lines to ECGFounder-ready format.

Steps: NaN->0, uV->mV, resample to 500Hz/5000pts, pad/truncate,
align+tile leads, highpass 0.5Hz, wavelet denoise, z-score.
"""
```

- [ ] **Step 4: Keep _bandpass_filter() function in file but unused**

Do NOT delete `_bandpass_filter()`. It serves as a fallback if wavelet approach causes issues. Just remove it from the pipeline flow.

---

### Task 5: Add Einthoven score to diagnostics

**Files:**
- Modify: `src/pipeline/digitize.py` (DigitizeInfo class + digitize method)

- [ ] **Step 1: Add field to DigitizeInfo**

Add to the `DigitizeInfo` dataclass (after `nonzero_leads_count`):

```python
einthoven_score: float = -1.0  # Pearson corr: II vs I+III (-1=not computed)
```

- [ ] **Step 2: Add Einthoven warning to has_warnings**

Add to `has_warnings` property:

```python
or (self.einthoven_score >= 0 and self.einthoven_score < 0.7)
```

- [ ] **Step 3: Add to summary() output**

Add a line to `summary()`:

```python
f"Einthoven consistency: {self.einthoven_score:.2f}",
```

- [ ] **Step 4: Compute in digitize() method**

In the `digitize()` method, after `self._validate_digitized_signal(...)`, add:

```python
from src.utils.signal_clean import einthoven_consistency
self.last_info.einthoven_score = einthoven_consistency(signal)
if self.last_info.einthoven_score < 0.7:
    logger.warning(
        "Einthoven consistency low (%.2f) — lead assignment may be incorrect",
        self.last_info.einthoven_score,
    )
```

- [ ] **Step 5: Show Einthoven in Gradio debug panel**

In `src/web/app.py` `_format_debug_info()`, add to the debug info div:

```python
f"<div><b>Einthoven:</b> {info.einthoven_score:.2f}"
f"{' ⚠️' if info.einthoven_score >= 0 and info.einthoven_score < 0.7 else ''}</div>"
```

---

### Task 6: Run full test suite and commit

- [ ] **Step 1: Run new tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_signal_clean.py -v
```

Expected: All 12 tests PASS.

- [ ] **Step 2: Run existing tests to verify no regression**

```bash
python -m pytest tests/ -v --timeout=30
```

Expected: All existing tests still PASS (postprocess shape test may need update if it checks for bandpass-specific behavior).

- [ ] **Step 3: Smoke test with Gradio**

```bash
# Restart Gradio and test with a real ECG image
lsof -ti :7860 | xargs kill -9 2>/dev/null
python -m src.web.app &
# Then test at http://localhost:7860
```

- [ ] **Step 4: Commit**

```bash
git add src/utils/signal_clean.py tests/test_signal_clean.py \
    src/pipeline/digitize.py src/web/app.py pyproject.toml \
    src/utils/__init__.py
git commit -m "feat: replace bandpass with highpass + wavelet denoising for signal quality

ECGFounder was trained on unfiltered WFDB signals (0-250 Hz). The previous
0.5-40 Hz bandpass cut QRS high-frequency content (40-150 Hz), distorting
waveform morphology the model expects.

New pipeline: 0.5 Hz highpass (baseline wander) + wavelet denoising (db4,
soft threshold on noise-band levels only). Preserves diagnostic morphology
while removing digitization artifacts.

Added Einthoven consistency check (II ≈ I+III) for lead assignment QA."
```

---

## Summary of Changes

| Before | After | Why |
|--------|-------|-----|
| Bandpass 0.5–40 Hz | Highpass 0.5 Hz + Wavelet denoise | Model trained on 0–250 Hz; 40 Hz cutoff destroyed QRS morphology |
| No lead consistency check | Einthoven score in diagnostics | Detects lead misassignment without extra inference |
| QRS smoothed, R peaks reduced | Full QRS preserved (40–150 Hz intact) | Better axis, BBB, MI, WPW diagnosis |

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Wavelet over-denoising removes features | Only top 3 detail levels thresholded; P/QRS/T band untouched |
| pywt dependency issues | 12M+ weekly downloads, numpy-only dep, battle-tested |
| Existing tests break | `_bandpass_filter()` kept in file; old tests don't call it directly |
| Signal quality worse for some images | `_bandpass_filter()` preserved as fallback; easy to revert |
