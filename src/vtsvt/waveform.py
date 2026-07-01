# Beat-level QRS morphology helpers for VT/SVT criteria feature extraction.

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.vtsvt.models import LeadMorphology
from src.vtsvt.qrs_shape import summarize_qrs_shape


@dataclass(frozen=True)
class _BeatMorphology:
    qrs_duration_ms: float
    has_rs_complex: bool
    rs_interval_ms: float | None
    initial_deflection: str | None
    initial_deflection_width_ms: float | None
    dominant_polarity: str | None
    vi_vt_ratio: float | None
    initial_downstroke_notched: bool
    qrs_pattern: str | None
    r_s_ratio: float | None
    qrs_onset_to_s_nadir_ms: float | None
    initial_r_taller_than_terminal_r: bool | None
    s_downstroke_notched: bool


def lead_morphology_from_rpeaks(
    lead: np.ndarray,
    rpeaks: np.ndarray,
    sample_rate: int,
    lead_name: str,
) -> LeadMorphology:
    """Summarize median QRS morphology for one lead over anchored beats."""
    beats = [
        morphology
        for r_peak in rpeaks
        if (morphology := _beat_morphology(lead, int(r_peak), sample_rate)) is not None
    ]
    if not beats:
        return LeadMorphology(
            lead=lead_name,
            qrs_duration_ms=None,
            has_rs_complex=False,
            rs_interval_ms=None,
            initial_deflection=None,
            initial_deflection_width_ms=None,
            dominant_polarity=None,
            vi_vt_ratio=None,
            initial_downstroke_notched=False,
            analyzed_beats=0,
        )

    rs_values = [beat.rs_interval_ms for beat in beats if beat.rs_interval_ms is not None]
    return LeadMorphology(
        lead=lead_name,
        qrs_duration_ms=_median([beat.qrs_duration_ms for beat in beats]),
        has_rs_complex=sum(beat.has_rs_complex for beat in beats) >= max(1, len(beats) // 2),
        rs_interval_ms=_median(rs_values),
        initial_deflection=_mode([beat.initial_deflection for beat in beats]),
        initial_deflection_width_ms=_median(
            [
                beat.initial_deflection_width_ms
                for beat in beats
                if beat.initial_deflection_width_ms is not None
            ]
        ),
        dominant_polarity=_mode([beat.dominant_polarity for beat in beats]),
        vi_vt_ratio=_median(
            [beat.vi_vt_ratio for beat in beats if beat.vi_vt_ratio is not None]
        ),
        initial_downstroke_notched=sum(beat.initial_downstroke_notched for beat in beats)
        >= max(1, len(beats) // 2),
        analyzed_beats=len(beats),
        qrs_pattern=_mode([beat.qrs_pattern for beat in beats]),
        r_s_ratio=_median([beat.r_s_ratio for beat in beats if beat.r_s_ratio is not None]),
        qrs_onset_to_s_nadir_ms=_median(
            [
                beat.qrs_onset_to_s_nadir_ms
                for beat in beats
                if beat.qrs_onset_to_s_nadir_ms is not None
            ]
        ),
        initial_r_taller_than_terminal_r=_majority_bool(
            [
                beat.initial_r_taller_than_terminal_r
                for beat in beats
                if beat.initial_r_taller_than_terminal_r is not None
            ]
        ),
        s_downstroke_notched=sum(beat.s_downstroke_notched for beat in beats)
        >= max(1, len(beats) // 2),
    )


def _beat_morphology(lead: np.ndarray, r_peak: int, sample_rate: int) -> _BeatMorphology | None:
    start = max(0, r_peak - int(0.06 * sample_rate))
    stop = min(lead.size, r_peak + int(0.23 * sample_rate))
    if stop - start < int(0.08 * sample_rate):
        return None

    segment = _smooth(lead[start:stop], width=5) - _baseline(lead, r_peak, sample_rate)
    amplitude = float(np.max(np.abs(segment)))
    if amplitude <= 1e-6:
        return None

    active = np.flatnonzero(np.abs(segment) >= 0.12 * amplitude)
    if active.size < 2:
        return None
    qrs = segment[int(active[0]) : int(active[-1]) + 1]
    threshold = 0.18 * float(np.max(np.abs(qrs)))
    initial = _initial_deflection(qrs, threshold, sample_rate)
    has_rs, rs_ms = _rs_interval(qrs, threshold, sample_rate)
    shape = summarize_qrs_shape(qrs, threshold, sample_rate)

    return _BeatMorphology(
        qrs_duration_ms=(qrs.size - 1) / sample_rate * 1000.0,
        has_rs_complex=has_rs,
        rs_interval_ms=rs_ms,
        initial_deflection=initial[0],
        initial_deflection_width_ms=initial[1],
        dominant_polarity=_dominant_polarity(qrs),
        vi_vt_ratio=_vi_vt_ratio(qrs, sample_rate),
        initial_downstroke_notched=_initial_downstroke_notched(qrs, initial[0], sample_rate),
        qrs_pattern=shape.pattern,
        r_s_ratio=shape.r_s_ratio,
        qrs_onset_to_s_nadir_ms=shape.qrs_onset_to_s_nadir_ms,
        initial_r_taller_than_terminal_r=shape.initial_r_taller_than_terminal_r,
        s_downstroke_notched=shape.s_downstroke_notched,
    )


def _baseline(lead: np.ndarray, r_peak: int, sample_rate: int) -> float:
    start = max(0, r_peak - int(0.22 * sample_rate))
    stop = max(start + 1, r_peak - int(0.08 * sample_rate))
    return float(np.median(lead[start:stop]))


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if values.size < width:
        return np.asarray(values, dtype=np.float64)
    return np.convolve(values, np.ones(width, dtype=np.float64) / width, mode="same")


def _rs_interval(qrs: np.ndarray, threshold: float, sample_rate: int) -> tuple[bool, float | None]:
    positive = np.flatnonzero(qrs > threshold)
    negative = np.flatnonzero(qrs < -threshold)
    if positive.size == 0 or negative.size == 0:
        return False, None

    r_start = int(positive[0])
    later_negative = negative[negative > r_start]
    if later_negative.size == 0:
        return False, None
    s_nadir = r_start + int(np.argmin(qrs[r_start:]))
    if qrs[s_nadir] >= -threshold:
        return False, None
    return True, (s_nadir - r_start) / sample_rate * 1000.0


def _initial_deflection(
    qrs: np.ndarray,
    threshold: float,
    sample_rate: int,
) -> tuple[str | None, float | None]:
    active = np.flatnonzero(np.abs(qrs) > threshold)
    if active.size == 0:
        return None, None
    first = int(active[0])
    sign = 1.0 if qrs[first] > 0 else -1.0
    signed = qrs * sign
    floor = 0.08 * float(np.max(signed))
    for idx in range(first + 1, qrs.size):
        if signed[idx] <= floor:
            return ("r" if sign > 0 else "q"), idx / sample_rate * 1000.0
    return ("r" if sign > 0 else "q"), qrs.size / sample_rate * 1000.0


def _dominant_polarity(qrs: np.ndarray) -> str | None:
    positive = float(np.max(qrs))
    negative = float(np.min(qrs))
    if max(abs(positive), abs(negative)) <= 1e-9:
        return None
    return "positive" if abs(positive) >= abs(negative) else "negative"


def _vi_vt_ratio(qrs: np.ndarray, sample_rate: int) -> float | None:
    window = max(int(0.04 * sample_rate), 1)
    if qrs.size <= window * 2:
        return None
    vi = abs(float(qrs[window] - qrs[0]))
    vt = abs(float(qrs[-1] - qrs[-window - 1]))
    return None if vt <= 1e-9 else vi / vt


def _initial_downstroke_notched(
    qrs: np.ndarray,
    initial_deflection: str | None,
    sample_rate: int,
) -> bool:
    if initial_deflection != "q":
        return False
    span = min(qrs.size, max(int(0.08 * sample_rate), 3))
    downstroke = _smooth(-qrs[:span], width=3)
    derivative = np.diff(downstroke)
    if derivative.size < 3:
        return False
    scale = float(np.max(downstroke) - np.min(downstroke))
    tolerance = 0.08 * scale
    if scale <= 1e-9:
        return False
    return any(
        left < -tolerance and right > tolerance
        for left, right in zip(derivative[:-1], derivative[1:])
    )


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _mode(values: list[str | None]) -> str | None:
    counts = Counter(value for value in values if value is not None)
    return counts.most_common(1)[0][0] if counts else None


def _majority_bool(values: list[bool]) -> bool | None:
    if not values:
        return None
    return sum(values) >= max(1, len(values) // 2)
