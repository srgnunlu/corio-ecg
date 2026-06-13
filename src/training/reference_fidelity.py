"""Matched printed-segment fidelity metrics for ECG image digitization."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.pipeline.digitize import LEAD_NAMES, NUM_LEADS


@dataclass(frozen=True)
class ShiftedCorrelationResult:
    """Best normalized comparison between two equal-length signal segments."""

    correlation: float
    normalized_rmse: float
    shift_samples: int


def _shifted_overlaps(
    reference: np.ndarray,
    predicted: np.ndarray,
    shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return overlapping arrays for a signed predicted-signal shift."""
    if shift < 0:
        return reference[-shift:], predicted[:shift]
    if shift > 0:
        return reference[:-shift], predicted[shift:]
    return reference, predicted


def _zscore(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    std = float(np.std(signal))
    if std < 1e-12:
        return np.zeros_like(signal)
    return (signal - float(np.mean(signal))) / std


def shifted_correlation(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    max_shift: int,
) -> ShiftedCorrelationResult:
    """Find the best correlation after a bounded horizontal alignment."""
    reference = np.asarray(reference, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if reference.ndim != 1 or predicted.ndim != 1 or reference.shape != predicted.shape:
        raise ValueError("reference and predicted must be equal-length 1D arrays")
    if max_shift < 0 or max_shift >= len(reference):
        raise ValueError("max_shift must be non-negative and shorter than the signals")

    best: ShiftedCorrelationResult | None = None
    for shift in range(-max_shift, max_shift + 1):
        reference_overlap, predicted_overlap = _shifted_overlaps(
            reference,
            predicted,
            shift,
        )

        reference_norm = _zscore(reference_overlap)
        predicted_norm = _zscore(predicted_overlap)
        correlation = float(np.mean(reference_norm * predicted_norm))
        normalized_rmse = float(np.sqrt(np.mean((reference_norm - predicted_norm) ** 2)))
        candidate = ShiftedCorrelationResult(
            correlation=correlation,
            normalized_rmse=normalized_rmse,
            shift_samples=shift,
        )
        if best is None or candidate.correlation > best.correlation:
            best = candidate

    if best is None:  # pragma: no cover - guarded by max_shift validation
        raise RuntimeError("No valid alignment candidates")
    return best


def evaluate_absolute_printed_segments(
    reference: np.ndarray,
    calibrated_digitized: np.ndarray,
    *,
    max_shift_samples: int = 50,
) -> dict[str, object]:
    """Measure absolute mV fidelity after bounded correlation alignment."""
    reference = np.asarray(reference)
    calibrated_digitized = np.asarray(calibrated_digitized)
    if reference.ndim != 2 or reference.shape[1] != NUM_LEADS:
        raise ValueError("reference must have shape (printed_samples, 12)")
    if calibrated_digitized.ndim != 2 or calibrated_digitized.shape[0] != NUM_LEADS:
        raise ValueError("calibrated_digitized must have shape (12, samples)")
    if calibrated_digitized.shape[1] < reference.shape[0]:
        raise ValueError("calibrated digitized signal is shorter than the printed reference")

    per_lead: list[dict[str, object]] = []
    for lead_index, lead_name in enumerate(LEAD_NAMES):
        reference_lead = reference[:, lead_index].astype(np.float64)
        predicted_lead = calibrated_digitized[
            lead_index, : reference.shape[0]
        ].astype(np.float64)
        alignment = shifted_correlation(
            reference_lead,
            predicted_lead,
            max_shift=max_shift_samples,
        )
        reference_overlap, predicted_overlap = _shifted_overlaps(
            reference_lead,
            predicted_lead,
            alignment.shift_samples,
        )
        error = predicted_overlap - reference_overlap
        rmse_mv = float(np.sqrt(np.mean(error**2)))
        reference_power = float(np.mean(reference_overlap**2))
        error_power = float(np.mean(error**2))
        snr_db = float(
            10.0
            * np.log10(max(reference_power, 1e-12) / max(error_power, 1e-12))
        )
        reference_rms = float(np.sqrt(reference_power))
        predicted_rms = float(np.sqrt(np.mean(predicted_overlap**2)))
        gain_ratio = predicted_rms / reference_rms if reference_rms > 1e-12 else 0.0
        per_lead.append(
            {
                "lead": lead_name,
                "shift_samples": alignment.shift_samples,
                "rmse_mv": rmse_mv,
                "snr_db": snr_db,
                "gain_ratio": gain_ratio,
            }
        )

    rmse_values = [float(item["rmse_mv"]) for item in per_lead]
    snr_values = [float(item["snr_db"]) for item in per_lead]
    gain_values = [float(item["gain_ratio"]) for item in per_lead]
    return {
        "reference_samples_per_lead": int(reference.shape[0]),
        "max_shift_samples": max_shift_samples,
        "mean_rmse_mv": float(np.mean(rmse_values)),
        "median_rmse_mv": float(np.median(rmse_values)),
        "mean_snr_db": float(np.mean(snr_values)),
        "median_snr_db": float(np.median(snr_values)),
        "mean_gain_ratio": float(np.mean(gain_values)),
        "median_gain_ratio": float(np.median(gain_values)),
        "per_lead": per_lead,
    }


def evaluate_printed_segments(
    reference: np.ndarray,
    digitized: np.ndarray,
    *,
    max_shift_samples: int = 50,
) -> dict[str, object]:
    """Compare matched printed lead segments against digitized lead segments.

    PMcardio references use shape ``(printed_samples, 12)``. Corio's digitizer
    emits ``(12, 5000)`` with each printed segment expanded across the output.
    The first matching-length segment is therefore compared for each lead.
    Both overlapping signals are z-scored because the diagnosis pipeline output
    is normalized and cannot support absolute-amplitude SNR or RMSE.
    """
    reference = np.asarray(reference)
    digitized = np.asarray(digitized)
    if reference.ndim != 2 or reference.shape[1] != NUM_LEADS:
        raise ValueError("reference must have shape (printed_samples, 12)")
    if digitized.ndim != 2 or digitized.shape[0] != NUM_LEADS:
        raise ValueError("digitized must have shape (12, samples)")
    if digitized.shape[1] < reference.shape[0]:
        raise ValueError("digitized signal is shorter than the printed reference")

    per_lead: list[dict[str, object]] = []
    for lead_index, lead_name in enumerate(LEAD_NAMES):
        result = shifted_correlation(
            reference[:, lead_index],
            digitized[lead_index, : reference.shape[0]],
            max_shift=max_shift_samples,
        )
        per_lead.append({"lead": lead_name, **asdict(result)})

    correlations = [float(item["correlation"]) for item in per_lead]
    normalized_rmse = [float(item["normalized_rmse"]) for item in per_lead]
    return {
        "reference_samples_per_lead": int(reference.shape[0]),
        "max_shift_samples": max_shift_samples,
        "mean_correlation": float(np.mean(correlations)),
        "median_correlation": float(np.median(correlations)),
        "minimum_correlation": float(np.min(correlations)),
        "mean_normalized_rmse": float(np.mean(normalized_rmse)),
        "median_normalized_rmse": float(np.median(normalized_rmse)),
        "per_lead": per_lead,
    }
