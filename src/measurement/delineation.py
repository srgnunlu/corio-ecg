# ECG wave delineation: locate QRS onset/offset, P-wave onset, and T-wave end
# per beat on a single lead. Pure scipy/numpy — no external delineation library.
#
# Timing-only: the digitized signal is z-score normalized (amplitude is unitless),
# so every threshold here is RELATIVE to local energy/deflection, never an absolute
# microvolt level. Boundaries are returned in sample indices; the caller converts
# to milliseconds. The T-wave end uses the classic tangent (Lepeschkin) method.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

_MAX_HR: float = 220.0
_MIN_HR: float = 25.0
_MAX_P_PEAK_RISE_S: float = 0.07
_P_ONSET_FALLBACK_FRACTION: float = 0.15


@dataclass
class BeatBoundaries:
    """Sample indices of fiducial points for one beat (None when undetected)."""

    r_peak: int
    qrs_onset: int | None
    qrs_offset: int | None
    p_onset: int | None
    t_end: int | None


def detect_rpeaks(lead: np.ndarray, sample_rate: int) -> np.ndarray:
    """Detect R-peak sample indices with a Pan-Tompkins style pipeline.

    A 5-18 Hz bandpass isolates QRS energy, the squared derivative emphasizes the
    steep R upstroke, and a moving-window integrator merges each complex into one
    lobe. Each detected lobe is refined back to the largest absolute deflection on
    the raw lead within +/-50 ms so the index lands on the true R peak.
    """
    if lead.size < sample_rate:  # need ~1 s of signal
        return np.empty(0, dtype=int)

    nyquist = sample_rate / 2.0
    sos = butter(N=3, Wn=[5.0 / nyquist, 18.0 / nyquist], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, lead)
    squared = np.diff(filtered) ** 2
    window = max(int(sample_rate * 0.12), 1)
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")

    threshold = float(np.mean(integrated) + 0.5 * np.std(integrated))
    refractory = int(sample_rate * 60.0 / _MAX_HR)
    lobes, _ = find_peaks(integrated, distance=max(refractory, 1), height=threshold)

    half = max(int(sample_rate * 0.05), 1)
    refined: list[int] = []
    baseline = float(np.median(lead))
    for lobe in lobes:
        lo, hi = max(0, lobe - half), min(lead.size, lobe + half + 1)
        local = lo + int(np.argmax(np.abs(lead[lo:hi] - baseline)))
        refined.append(local)
    return np.unique(np.asarray(refined, dtype=int))


def delineate_beat(
    lead: np.ndarray,
    r_peak: int,
    sample_rate: int,
    rr_samples: float,
) -> BeatBoundaries:
    """Locate QRS onset/offset, P onset and T end around a single R peak."""
    energy = _qrs_energy(lead, sample_rate)
    qrs_onset, qrs_offset = _qrs_bounds(energy, r_peak, sample_rate)
    baseline = _baseline_level(lead, qrs_onset, sample_rate)
    smooth = _lowpass(lead, sample_rate, cutoff_hz=20.0)

    p_onset = (
        _p_onset(smooth, qrs_onset, baseline, sample_rate, rr_samples)
        if qrs_onset is not None
        else None
    )
    t_end = (
        _t_end(smooth, qrs_offset, baseline, sample_rate, rr_samples)
        if qrs_offset is not None
        else None
    )
    return BeatBoundaries(r_peak, qrs_onset, qrs_offset, p_onset, t_end)


def _qrs_energy(lead: np.ndarray, sample_rate: int) -> np.ndarray:
    """Smoothed absolute first-derivative — high over the QRS, near zero elsewhere."""
    smooth = _lowpass(lead, sample_rate, cutoff_hz=40.0)
    deriv = np.abs(np.gradient(smooth))
    window = max(int(sample_rate * 0.01), 1)
    return np.convolve(deriv, np.ones(window) / window, mode="same")


def _qrs_bounds(
    energy: np.ndarray,
    r_peak: int,
    sample_rate: int,
) -> tuple[int | None, int | None]:
    """Find QRS onset/offset as the outermost energy-threshold crossings.

    The QRS energy is bi-lobed (steep up- and down-slopes) with a dip at the R
    apex itself, so walking outward *from* R stalls immediately. Instead take the
    span of samples that clear a fraction of the local peak energy: onset is the
    first such sample in the window, offset the last.
    """
    search = max(int(sample_rate * 0.08), 1)  # QRS half-width search ~80 ms
    lo, hi = max(0, r_peak - search), min(energy.size, r_peak + search + 1)
    if hi <= lo:
        return None, None
    window = energy[lo:hi]
    peak_energy = float(np.max(window))
    if peak_energy <= 0.0:
        return None, None

    above = np.flatnonzero(window > 0.15 * peak_energy)
    if above.size == 0:
        return None, None
    return lo + int(above[0]), lo + int(above[-1])


def _p_onset(
    smooth: np.ndarray,
    qrs_onset: int,
    baseline: float,
    sample_rate: int,
    rr_samples: float,
) -> int | None:
    """Find P-wave onset just before QRS onset.

    The P wave is the last deflection before QRS, so the search window is
    capped by the RR interval (to avoid running into the previous beat at fast
    rates) and the P peak is taken as the deflection *closest* to QRS that clears
    a fraction of the QRS-region swing — not merely the largest, which at high
    rates can be the tail of the preceding T wave (P-on-T). Returns None when no
    organized P exists (e.g. atrial fibrillation).
    """
    lookback = min(int(sample_rate * 0.25), int(0.45 * rr_samples))
    win_start = max(0, qrs_onset - lookback)
    win_end = max(0, qrs_onset - int(sample_rate * 0.04))
    if win_end - win_start < int(sample_rate * 0.02):
        return None

    segment = smooth[win_start:win_end] - baseline
    swing = float(np.ptp(smooth[win_start : qrs_onset + 1]))
    if swing <= 0:
        return None

    abs_seg = np.abs(segment)
    threshold = 0.10 * swing
    peaks, _ = find_peaks(abs_seg, height=threshold)
    if peaks.size == 0:
        return None

    p_peak_local = int(peaks[-1])  # closest to QRS onset
    onset_threshold = 0.20 * abs_seg[p_peak_local]
    onset = p_peak_local
    while onset > 0 and abs_seg[onset] > onset_threshold:
        onset -= 1
    max_peak_rise = max(int(sample_rate * _MAX_P_PEAK_RISE_S), 1)
    if onset == 0 or p_peak_local - onset > max_peak_rise:
        onset = _p_onset_from_peak_window(segment, p_peak_local, max_peak_rise)
    return win_start + onset


def _p_onset_from_peak_window(
    segment: np.ndarray,
    p_peak_local: int,
    max_peak_rise: int,
) -> int:
    """Fallback P onset when residual baseline energy prevents threshold crossing."""
    polarity = 1.0 if segment[p_peak_local] >= 0 else -1.0
    signed = segment * polarity
    peak_amp = float(signed[p_peak_local])
    if peak_amp <= 0.0:
        return max(0, p_peak_local - max_peak_rise)

    lo = max(0, p_peak_local - max_peak_rise)
    threshold = _P_ONSET_FALLBACK_FRACTION * peak_amp
    below = np.flatnonzero(signed[lo : p_peak_local + 1] <= threshold)
    if below.size == 0:
        return lo
    return lo + int(below[-1])


def _t_end(
    smooth: np.ndarray,
    qrs_offset: int,
    baseline: float,
    sample_rate: int,
    rr_samples: float,
) -> int | None:
    """Locate T-wave end via the tangent method.

    The search window runs from just after QRS offset to a rate-scaled limit. The
    T peak is the largest deflection from baseline; from the peak the algorithm
    finds the steepest descending slope and intersects that tangent with the
    baseline — the standard Lepeschkin construction.
    """
    win_start = min(smooth.size - 1, qrs_offset + int(sample_rate * 0.04))
    span = min(int(0.55 * rr_samples), int(sample_rate * 0.60))
    win_end = min(smooth.size, qrs_offset + span)
    if win_end - win_start < int(sample_rate * 0.04):
        return None

    segment = smooth[win_start:win_end] - baseline
    t_peak_local = int(np.argmax(np.abs(segment)))
    if t_peak_local >= segment.size - 2:
        return None

    # Steepest |slope| on the descending limb after the T peak.
    descending = np.gradient(segment[t_peak_local:])
    slope_local = t_peak_local + int(np.argmax(np.abs(descending)))
    slope = float(descending[slope_local - t_peak_local])
    if abs(slope) < 1e-6:
        return None

    # Tangent y = segment[slope_local] + slope * (x - slope_local); solve y = 0.
    x_zero = slope_local - segment[slope_local] / slope
    t_end_local = int(round(x_zero))
    t_end_local = max(slope_local, min(segment.size - 1, t_end_local))
    return win_start + t_end_local


def _baseline_level(
    lead: np.ndarray,
    qrs_onset: int | None,
    sample_rate: int,
) -> float:
    """Isoelectric estimate from the PR segment just before QRS onset."""
    if qrs_onset is None:
        return float(np.median(lead))
    seg_start = max(0, qrs_onset - int(sample_rate * 0.05))
    if qrs_onset - seg_start < 2:
        return float(np.median(lead))
    return float(np.median(lead[seg_start:qrs_onset]))


def _lowpass(lead: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Zero-phase low-pass to suppress residual noise before delineation."""
    nyquist = sample_rate / 2.0
    cutoff = min(cutoff_hz / nyquist, 0.99)
    sos = butter(N=4, Wn=cutoff, btype="lowpass", output="sos")
    return np.asarray(sosfiltfilt(sos, lead), dtype=np.float64)
