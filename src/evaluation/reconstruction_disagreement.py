"""Measure disagreement between two reconstructions of the same ECG image."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from src.pipeline.digitize import LEAD_NAMES, NUM_LEADS
from src.training.reference_fidelity import shifted_correlation


def _validate_pair(first: np.ndarray, second: np.ndarray, label: str) -> None:
    if first.ndim != 2 or first.shape[0] != NUM_LEADS:
        raise ValueError(f"{label} first signal must have shape (12, samples)")
    if second.shape != first.shape:
        raise ValueError(f"{label} signals must have identical shapes")


def _shifted_overlaps(
    first: np.ndarray,
    second: np.ndarray,
    shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    if shift < 0:
        return first[-shift:], second[:shift]
    if shift > 0:
        return first[:-shift], second[shift:]
    return first, second


def evaluate_reconstruction_pair(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_calibrated: np.ndarray | None = None,
    second_calibrated: np.ndarray | None = None,
    max_shift_samples: int = 50,
) -> dict[str, Any]:
    """Compare two inference attempts without using a matched reference signal."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    _validate_pair(first, second, "model-input")

    calibrated_available = first_calibrated is not None and second_calibrated is not None
    if calibrated_available:
        first_calibrated = np.asarray(first_calibrated, dtype=np.float64)
        second_calibrated = np.asarray(second_calibrated, dtype=np.float64)
        _validate_pair(first_calibrated, second_calibrated, "calibrated")
        if first_calibrated.shape != first.shape:
            raise ValueError("calibrated and model-input signals must have identical shapes")
    elif first_calibrated is not None or second_calibrated is not None:
        raise ValueError("both calibrated signals are required when either is provided")

    per_lead: list[dict[str, Any]] = []
    for lead_index, lead_name in enumerate(LEAD_NAMES):
        alignment = shifted_correlation(
            first[lead_index],
            second[lead_index],
            max_shift=max_shift_samples,
        )
        lead_result: dict[str, Any] = {"lead": lead_name, **asdict(alignment)}
        if calibrated_available:
            assert first_calibrated is not None and second_calibrated is not None
            first_overlap, second_overlap = _shifted_overlaps(
                first_calibrated[lead_index],
                second_calibrated[lead_index],
                alignment.shift_samples,
            )
            difference = second_overlap - first_overlap
            first_rms = float(np.sqrt(np.mean(first_overlap**2)))
            second_rms = float(np.sqrt(np.mean(second_overlap**2)))
            lead_result["calibrated_rmse_mv"] = float(
                np.sqrt(np.mean(difference**2))
            )
            lead_result["calibrated_gain_ratio"] = (
                second_rms / first_rms if first_rms > 1e-12 else 0.0
            )
        per_lead.append(lead_result)

    correlations = [float(item["correlation"]) for item in per_lead]
    normalized_rmse = [float(item["normalized_rmse"]) for item in per_lead]
    result: dict[str, Any] = {
        "max_shift_samples": max_shift_samples,
        "median_correlation": float(np.median(correlations)),
        "minimum_correlation": float(np.min(correlations)),
        "median_normalized_rmse": float(np.median(normalized_rmse)),
        "unstable_leads_below_0_8": sum(value < 0.8 for value in correlations),
        "per_lead": per_lead,
    }
    if calibrated_available:
        rmse_values = [float(item["calibrated_rmse_mv"]) for item in per_lead]
        gain_errors = [
            abs(float(item["calibrated_gain_ratio"]) - 1.0) for item in per_lead
        ]
        result["calibrated"] = {
            "median_rmse_mv": float(np.median(rmse_values)),
            "median_gain_error": float(np.median(gain_errors)),
            "maximum_gain_error": float(np.max(gain_errors)),
        }
    return result


__all__ = ["evaluate_reconstruction_pair"]
