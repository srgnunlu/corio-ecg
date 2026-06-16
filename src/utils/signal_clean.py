# ECG signal cleaning utilities for digitized paper ECG signals
# High-pass filter for baseline wander, wavelet denoising for noise,
# and Einthoven consistency check for lead assignment verification

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pywt
from scipy.signal import butter, correlate, sosfiltfilt

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


# Limb-lead derivation identities (Einthoven + Goldberger). Each maps a
# measured limb lead to the combination of the three bipolar limb leads it
# must equal in a simultaneous recording. Lead order: I=0, II=1, III=2,
# aVR=3, aVL=4, aVF=5.
#   II  = I + III          (Einthoven's law)
#   aVR = -(I + II) / 2     (Goldberger)
#   aVL = (I - III) / 2     (Goldberger)
#   aVF = (II + III) / 2    (Goldberger)
# Maximum lag (seconds) searched when matching a measured lead to its derived
# counterpart. On a 3x4 paper ECG the limb leads come from different columns
# (different 2.5 s time windows), so after column tiling the derived and
# measured traces are phase-shifted by an arbitrary fraction of a cardiac
# cycle. A lag search of just over one slow RR interval recovers that phase so
# the metric measures morphology agreement, not column timing.
_GOLDBERGER_MAX_LAG_SECONDS: float = 1.3


@dataclass(frozen=True)
class GoldbergerConsistency:
    """Phase-tolerant agreement of measured limb leads with their derivations.

    All correlations are best-lag Pearson values in ``[-1, 1]`` (higher is more
    physiologically consistent). Residuals are ``1 - correlation`` clamped to
    ``[0, 1]`` so that larger means a worse, internally inconsistent
    reconstruction — the failure mode a stable-but-wrong digitization exhibits.
    """

    correlations: dict[str, float]
    residuals: dict[str, float]
    mean_residual: float
    worst_residual: float


def _best_lag_correlation(
    measured: np.ndarray,
    derived: np.ndarray,
    max_lag: int,
) -> float:
    """Return the maximum normalized cross-correlation within ``+/- max_lag``.

    Both inputs are mean-centered and scaled to unit standard deviation, so the
    full FFT cross-correlation divided by the sample count equals the Pearson
    correlation at each integer lag. The maximum over the lag window recovers
    the best phase alignment between a measured lead and its derived twin.

    Args:
        measured: 1-D measured lead signal.
        derived: 1-D derived lead signal (a limb-lead combination).
        max_lag: Lag half-window in samples (``0`` reduces to zero-lag Pearson).

    Returns:
        Best-lag Pearson correlation, or ``0.0`` for degenerate flat inputs.
    """
    centered_measured = measured - np.mean(measured)
    centered_derived = derived - np.mean(derived)
    std_measured = float(np.std(centered_measured))
    std_derived = float(np.std(centered_derived))
    if std_measured < 1e-9 or std_derived < 1e-9:
        return 0.0

    normalized_measured = centered_measured / std_measured
    normalized_derived = centered_derived / std_derived
    n_samples = len(normalized_measured)

    cross = correlate(normalized_measured, normalized_derived, mode="full", method="fft")
    cross = cross / n_samples
    lags = np.arange(-(n_samples - 1), n_samples)
    window = np.abs(lags) <= max_lag
    best = float(np.max(cross[window]))
    # FFT rounding can nudge a perfect match a hair past 1.0; clamp for safety.
    return float(np.clip(best, -1.0, 1.0))


def goldberger_consistency(
    signal: np.ndarray,
    sample_rate: int = 500,
    max_lag_seconds: float = _GOLDBERGER_MAX_LAG_SECONDS,
) -> GoldbergerConsistency:
    """Score the full limb-lead redundancy of a digitized 12-lead signal.

    Extends the single Einthoven check (``II = I + III``) to all four
    Einthoven/Goldberger derivation identities. A faithful reconstruction
    satisfies every identity (residuals near zero); a stable-but-wrong
    digitization — distorted morphology or a swapped lead — violates one or
    more, regardless of how reproducible it is.

    Pass ``max_lag_seconds=0.0`` for the strict zero-lag variant (only valid
    when all six limb leads share one time window).

    Args:
        signal: Shape ``(>=6, N)`` — at least the six limb leads, any scaling.
        sample_rate: Sampling rate in Hz, used to convert the lag window.
        max_lag_seconds: Phase-search half-window in seconds.

    Returns:
        A :class:`GoldbergerConsistency` with per-rule correlations/residuals
        and the mean and worst residual across the four rules.
    """
    if signal.shape[0] < 6:
        raise ValueError(f"goldberger_consistency needs >=6 leads, got {signal.shape[0]}")

    lead_i, lead_ii, lead_iii = signal[0], signal[1], signal[2]
    lead_avr, lead_avl, lead_avf = signal[3], signal[4], signal[5]
    rules: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "II": (lead_ii, lead_i + lead_iii),
        "aVR": (lead_avr, -(lead_i + lead_ii) / 2.0),
        "aVL": (lead_avl, (lead_i - lead_iii) / 2.0),
        "aVF": (lead_avf, (lead_ii + lead_iii) / 2.0),
    }

    max_lag = max(0, int(round(max_lag_seconds * sample_rate)))
    correlations = {
        name: _best_lag_correlation(measured, derived, max_lag)
        for name, (measured, derived) in rules.items()
    }
    residuals = {name: float(np.clip(1.0 - corr, 0.0, 1.0)) for name, corr in correlations.items()}
    residual_values = list(residuals.values())
    return GoldbergerConsistency(
        correlations=correlations,
        residuals=residuals,
        mean_residual=float(np.mean(residual_values)),
        worst_residual=float(np.max(residual_values)),
    )
