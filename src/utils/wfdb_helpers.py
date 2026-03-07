# WFDB format ECG signal reader and preprocessor for ECGFounder inference
# Handles reading, reordering leads, resampling, normalization, and padding

import numpy as np
import wfdb
from scipy.interpolate import interp1d

# ECGFounder expects leads in this exact order
ECGFOUNDER_LEAD_ORDER: list[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

TARGET_SAMPLE_RATE: int = 500
TARGET_LENGTH: int = 5000  # 10 seconds at 500 Hz


def read_ecg_signal(record_path: str) -> tuple[np.ndarray, dict]:
    """Read a WFDB record and return preprocessed signal ready for ECGFounder.

    Args:
        record_path: Path to WFDB record (without .dat/.hea extension).

    Returns:
        Tuple of (signal, metadata) where signal has shape (12, 5000)
        and metadata contains recording information.
    """
    record = wfdb.rdrecord(record_path)

    raw_signal = record.p_signal  # shape: (n_samples, n_leads)
    raw_signal = np.nan_to_num(raw_signal, nan=0.0)

    original_rate = record.fs
    source_leads = record.sig_name

    # Reorder leads to match ECGFounder expected order
    signal = _reorder_leads(raw_signal, source_leads)

    # Resample to 500 Hz if the original rate differs
    if original_rate != TARGET_SAMPLE_RATE:
        signal = _resample(signal, int(original_rate), TARGET_SAMPLE_RATE)

    # Transpose to (12, n_samples) for ECGFounder
    signal = signal.T

    # Pad or truncate to exactly 5000 samples
    signal = _pad_or_truncate(signal, TARGET_LENGTH)

    # Z-score normalize the entire signal
    signal = _z_score_normalize(signal)

    duration_seconds = record.sig_len / original_rate

    metadata = {
        "sample_rate": TARGET_SAMPLE_RATE,
        "original_sample_rate": int(original_rate),
        "lead_names": ECGFOUNDER_LEAD_ORDER,
        "duration_seconds": float(duration_seconds),
        "record_name": record.record_name,
    }

    return signal, metadata


def _reorder_leads(
    signal: np.ndarray, source_leads: list[str]
) -> np.ndarray:
    """Reorder signal columns to match ECGFOUNDER_LEAD_ORDER.

    Args:
        signal: ECG signal with shape (n_samples, n_leads).
        source_leads: Lead names in the order they appear in signal.

    Returns:
        Reordered signal with shape (n_samples, 12).
    """
    n_samples = signal.shape[0]
    reordered = np.zeros((n_samples, len(ECGFOUNDER_LEAD_ORDER)), dtype=signal.dtype)

    # Build a lookup from cleaned lead name to column index
    cleaned_leads = {name.strip(): idx for idx, name in enumerate(source_leads)}

    for target_idx, lead_name in enumerate(ECGFOUNDER_LEAD_ORDER):
        if lead_name in cleaned_leads:
            reordered[:, target_idx] = signal[:, cleaned_leads[lead_name]]
        # Missing leads remain zero-filled

    return reordered


def _resample(
    signal: np.ndarray, original_rate: int, target_rate: int
) -> np.ndarray:
    """Resample signal from original_rate to target_rate using linear interpolation.

    Args:
        signal: ECG signal with shape (n_samples, n_leads).
        original_rate: Original sampling frequency in Hz.
        target_rate: Target sampling frequency in Hz.

    Returns:
        Resampled signal with shape (new_n_samples, n_leads).
    """
    n_samples, n_leads = signal.shape
    duration = n_samples / original_rate

    original_times = np.linspace(0, duration, n_samples, endpoint=False)
    new_n_samples = int(duration * target_rate)
    new_times = np.linspace(0, duration, new_n_samples, endpoint=False)

    resampled = np.zeros((new_n_samples, n_leads), dtype=signal.dtype)
    for lead_idx in range(n_leads):
        interpolator = interp1d(original_times, signal[:, lead_idx], kind="linear")
        resampled[:, lead_idx] = interpolator(new_times)

    return resampled


def _pad_or_truncate(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros or truncate signal to target_length samples.

    Args:
        signal: ECG signal with shape (12, n_samples).
        target_length: Desired number of samples.

    Returns:
        Signal with shape (12, target_length).
    """
    n_leads, n_samples = signal.shape

    if n_samples >= target_length:
        return signal[:, :target_length]

    # Zero-pad on the right
    padded = np.zeros((n_leads, target_length), dtype=signal.dtype)
    padded[:, :n_samples] = signal
    return padded


def _z_score_normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalize the entire signal (global mean and std).

    Args:
        signal: ECG signal array of any shape.

    Returns:
        Normalized signal with zero mean and unit variance.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + 1e-8)
