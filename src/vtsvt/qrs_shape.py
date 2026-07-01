# QRS shape primitives shared by VT/SVT morphology criteria.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QRSShape:
    """Auditable shape summary for one median-like QRS complex."""

    pattern: str | None
    r_s_ratio: float | None
    qrs_onset_to_s_nadir_ms: float | None
    initial_r_taller_than_terminal_r: bool | None
    s_downstroke_notched: bool


def summarize_qrs_shape(qrs: np.ndarray, threshold: float, sample_rate: int) -> QRSShape:
    """Summarize terminal QRS morphology from a baseline-centered QRS segment."""
    positive = np.flatnonzero(qrs > threshold)
    negative = np.flatnonzero(qrs < -threshold)

    return QRSShape(
        pattern=_qrs_pattern(qrs, positive, negative, sample_rate),
        r_s_ratio=_r_s_ratio(qrs, positive, negative),
        qrs_onset_to_s_nadir_ms=_qrs_onset_to_s_nadir(qrs, negative, threshold, sample_rate),
        initial_r_taller_than_terminal_r=_initial_r_taller_than_terminal_r(
            qrs, positive, negative, sample_rate
        ),
        s_downstroke_notched=_s_downstroke_notched(qrs, negative, sample_rate),
    )


def _qrs_pattern(
    qrs: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    sample_rate: int,
) -> str | None:
    if positive.size == 0 and negative.size == 0:
        return None
    if positive.size == 0:
        return "qs"
    if negative.size == 0:
        return "r"

    first_positive = int(positive[0])
    first_negative = int(negative[0])
    if first_negative < first_positive:
        later_positive = positive[positive > first_negative]
        if later_positive.size == 0:
            return "qs"
        later_negative = negative[negative > int(later_positive[0])]
        return "qrs" if later_negative.size else "qr"

    later_negative = negative[negative > first_positive]
    if later_negative.size == 0:
        return "r"
    s_nadir = int(np.argmin(qrs[first_positive:])) + first_positive
    min_gap = max(2, int(0.012 * sample_rate))
    terminal_positive = positive[positive > s_nadir + min_gap]
    return "rsr" if terminal_positive.size else "rs"


def _r_s_ratio(
    qrs: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
) -> float | None:
    if positive.size == 0 or negative.size == 0:
        return None
    r_amp = float(np.max(qrs[positive]))
    s_amp = abs(float(np.min(qrs[negative])))
    return None if s_amp <= 1e-9 else r_amp / s_amp


def _qrs_onset_to_s_nadir(
    qrs: np.ndarray,
    negative: np.ndarray,
    threshold: float,
    sample_rate: int,
) -> float | None:
    if negative.size == 0:
        return None
    s_nadir = int(np.argmin(qrs))
    if qrs[s_nadir] >= -threshold:
        return None
    return s_nadir / sample_rate * 1000.0


def _initial_r_taller_than_terminal_r(
    qrs: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    sample_rate: int,
) -> bool | None:
    if positive.size == 0 or negative.size == 0:
        return None
    first_positive = int(positive[0])
    s_nadir = int(np.argmin(qrs[first_positive:])) + first_positive
    min_gap = max(2, int(0.012 * sample_rate))
    terminal_positive = positive[positive > s_nadir + min_gap]
    if terminal_positive.size == 0:
        return None
    initial_peak = float(np.max(qrs[first_positive : s_nadir + 1]))
    terminal_peak = float(np.max(qrs[terminal_positive]))
    return initial_peak > terminal_peak


def _s_downstroke_notched(
    qrs: np.ndarray,
    negative: np.ndarray,
    sample_rate: int,
) -> bool:
    if negative.size == 0:
        return False
    s_nadir = int(np.argmin(qrs))
    span = max(int(0.08 * sample_rate), 3)
    start = max(0, s_nadir - span)
    downstroke = -_smooth(qrs[start : s_nadir + 1], width=3)
    derivative = np.diff(downstroke)
    if derivative.size < 3:
        return False
    scale = float(np.max(downstroke) - np.min(downstroke))
    if scale <= 1e-9:
        return False
    tolerance = 0.08 * scale
    return any(
        left < -tolerance and right > tolerance
        for left, right in zip(derivative[:-1], derivative[1:])
    )


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if values.size < width:
        return np.asarray(values, dtype=np.float64)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")
