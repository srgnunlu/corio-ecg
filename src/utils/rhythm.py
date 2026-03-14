# Rhythm-level helper functions for paper-ECG post-processing.
# Heart rate estimation uses autocorrelation (robust to T-wave confusion)
# instead of peak detection which overcounts on z-score normalized signals.

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

TARGET_SAMPLE_RATE: int = 500
# Preferred leads for rhythm estimation (Lead II best, then precordials)
_PREFERRED_LEADS: tuple[int, ...] = (1, 6, 7, 8, 9, 10, 11)

# Physiological heart rate bounds (in bpm)
_MIN_HR: float = 25.0
_MAX_HR: float = 220.0


def estimate_heart_rate_bpm(
    signal: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> float | None:
    """Estimate heart rate from a 12-lead ECG signal.

    Uses autocorrelation to find the dominant RR interval, which is more
    robust than peak detection on z-score normalized signals where T-waves
    become as prominent as QRS complexes.
    """
    best_result: tuple[float, int, float] | None = None

    for lead_idx in _candidate_leads(signal):
        lead = signal[lead_idx]
        if np.std(lead) < 0.1:
            continue

        bpm = _autocorrelation_hr(lead, sample_rate)
        if bpm is None:
            continue

        # Score: prefer Lead II (index 1) and leads with clear periodicity
        lead_priority = 2.0 if lead_idx == 1 else 1.0
        score = lead_priority
        if best_result is None or score > best_result[2]:
            best_result = (bpm, lead_idx, score)

    # Fallback to peak-based if autocorrelation fails
    if best_result is None:
        best_result = _peak_based_hr(signal, sample_rate)

    return None if best_result is None else best_result[0]


def _autocorrelation_hr(
    lead: np.ndarray,
    sample_rate: int,
) -> float | None:
    """Estimate heart rate using autocorrelation of squared derivative.

    The derivative emphasizes QRS complexes (steep slopes) and suppresses
    T-waves (gentle slopes). Squaring makes all values positive, and
    autocorrelation finds the dominant RR interval. This is inspired by
    the Pan-Tompkins QRS detection algorithm.
    """
    # Derivative + square: steep QRS slopes dominate, T-waves suppressed
    diff = np.diff(lead)
    squared = diff ** 2
    squared = squared - np.mean(squared)

    # Autocorrelation via FFT (efficient for long signals)
    n = len(squared)
    fft = np.fft.rfft(squared, n=2 * n)
    acf = np.fft.irfft(fft * np.conj(fft))[:n]
    acf = acf / (acf[0] + 1e-10)  # normalize to [0, 1]

    # Search for the first significant peak in physiological HR range
    min_lag = int(sample_rate * 60.0 / _MAX_HR)  # ~136 samples (220 bpm)
    max_lag = int(sample_rate * 60.0 / _MIN_HR)  # ~1200 samples (25 bpm)
    max_lag = min(max_lag, n - 1)

    if min_lag >= max_lag:
        return None

    acf_segment = acf[min_lag:max_lag]
    if len(acf_segment) < 10:
        return None

    # Find peaks in the autocorrelation
    peaks, properties = find_peaks(
        acf_segment,
        distance=int(sample_rate * 0.15),  # min 0.15s between candidates
        prominence=0.05,
    )

    if len(peaks) == 0:
        return None

    # Take the first (shortest lag = highest physiological HR) prominent peak
    best_peak = peaks[0]
    rr_lag = best_peak + min_lag
    rr_seconds = rr_lag / sample_rate
    bpm = 60.0 / rr_seconds

    if bpm < _MIN_HR or bpm > _MAX_HR:
        return None

    return bpm


def _peak_based_hr(
    signal: np.ndarray,
    sample_rate: int,
) -> tuple[float, int, float] | None:
    """Fallback peak-based HR estimation with stricter distance."""
    for lead_idx in _candidate_leads(signal):
        lead = signal[lead_idx]
        if np.std(lead) < 0.1:
            continue

        envelope = _qrs_envelope(lead, sample_rate)
        prominence = max(float(np.std(envelope)) * 1.0, 0.1)
        # Use 0.35s minimum distance (max 171 bpm) to avoid T-wave counting
        peaks, _ = find_peaks(
            envelope,
            distance=max(int(sample_rate * 0.35), 1),
            prominence=prominence,
        )
        if len(peaks) < 3:
            continue

        rr_seconds = np.diff(peaks) / sample_rate
        median_rr = float(np.median(rr_seconds))
        if median_rr <= 0:
            continue

        bpm = 60.0 / median_rr
        if bpm < _MIN_HR or bpm > _MAX_HR:
            continue

        return (bpm, lead_idx, len(peaks))

    return None


def _candidate_leads(signal: np.ndarray) -> list[int]:
    ordered: list[int] = list(_PREFERRED_LEADS)
    ordered.extend(idx for idx in range(signal.shape[0]) if idx not in _PREFERRED_LEADS)
    return ordered


def _qrs_envelope(lead: np.ndarray, sample_rate: int) -> np.ndarray:
    """Create a smoothed absolute envelope that emphasizes QRS complexes."""
    abs_lead = np.abs(lead)
    window = max(int(sample_rate * 0.03), 1)
    kernel = np.ones(window, dtype=np.float64) / window
    envelope = np.convolve(abs_lead, kernel, mode="same")
    return envelope.astype(np.float64, copy=False)
