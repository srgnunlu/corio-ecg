# Rhythm-level helper functions for paper-ECG post-processing.

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

TARGET_SAMPLE_RATE: int = 500
_PREFERRED_LEADS: tuple[int, ...] = (1, 6, 7, 8, 9, 10, 11)


def estimate_heart_rate_bpm(
    signal: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> float | None:
    """Estimate heart rate from a 12-lead ECG signal.

    Works on the z-score normalized signal used by ECGFounder. The estimate is
    intentionally simple and only used to suppress obviously contradictory
    rhythm labels such as bradycardia on a clear tachycardia strip.
    """
    best_result: tuple[float, int, float] | None = None

    for lead_idx in _candidate_leads(signal):
        lead = signal[lead_idx]
        if np.std(lead) < 0.1:
            continue

        envelope = _qrs_envelope(lead, sample_rate)
        prominence = max(float(np.std(envelope)) * 0.6, 0.05)
        peaks, _ = find_peaks(
            envelope,
            distance=max(int(sample_rate * 0.18), 1),
            prominence=prominence,
        )
        if len(peaks) < 3:
            continue

        rr_seconds = np.diff(peaks) / sample_rate
        if len(rr_seconds) == 0:
            continue

        median_rr = float(np.median(rr_seconds))
        if median_rr <= 0:
            continue

        bpm = 60.0 / median_rr
        if bpm < 20.0 or bpm > 260.0:
            continue

        rr_stability = float(np.std(rr_seconds))
        score = len(peaks) - rr_stability
        if best_result is None or score > best_result[2]:
            best_result = (bpm, lead_idx, score)

    return None if best_result is None else best_result[0]


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
