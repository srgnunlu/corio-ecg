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
_DECOMPOSITION_LEVEL: int = 6    # 6 levels at 500 Hz covers 0-250 Hz
_NOISE_LEVELS: int = 2           # Only threshold top 2 detail levels (>~62 Hz)
# Level 1: 125-250 Hz (muscle artifact, digitization noise) -> THRESHOLD
# Level 2: 62.5-125 Hz (mostly noise, some QRS harmonics) -> THRESHOLD
# Level 3: 31.25-62.5 Hz (QRS slope content) -> PRESERVE
# Levels 4-6: P wave, T wave, QRS body -> PRESERVE (no thresholding)


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
        # noise_levels=3 -> threshold d1, d2, d3 (indices -1, -2, -3)
        for j in range(1, noise_levels + 1):
            coeffs[-j] = pywt.threshold(coeffs[-j], threshold, mode="soft")

        # Reconstruct — waverec may produce 1 extra sample, truncate
        reconstructed = pywt.waverec(coeffs, _WAVELET)
        denoised[i] = reconstructed[: len(lead)]

    return denoised


def einthoven_consistency(signal: np.ndarray) -> float:
    """Check if Einthoven's law holds: Lead II ~ Lead I + Lead III.

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
