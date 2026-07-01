# Feature extraction orchestration for wide-complex tachycardia criteria.

from __future__ import annotations

import numpy as np

from src.measurement.delineation import detect_rpeaks
from src.measurement.intervals import IntervalMeasurements, measure_intervals
from src.measurement.rhythm_analysis import RhythmAnalysis, analyze_rhythm
from src.vtsvt.models import WCTFeatures
from src.vtsvt.waveform import lead_morphology_from_rpeaks

TARGET_SAMPLE_RATE: int = 500
_LEAD_NAMES: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)
_ANCHOR_LEADS: tuple[int, ...] = (1, 6, 7, 8, 9, 10, 11)
_PRECORDIAL_LEADS: tuple[int, ...] = (6, 7, 8, 9, 10, 11)
_AVR_LEAD: int = 3


def extract_wct_features(
    signal: np.ndarray,
    rhythm_strip: np.ndarray | None = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    intervals: IntervalMeasurements | None = None,
    rhythm: RhythmAnalysis | None = None,
) -> WCTFeatures:
    """Extract morphology features used by Brugada and aVR criteria.

    Args:
        signal: (12, samples) digitized ECG.
        rhythm_strip: optional full-duration strip; preferred for beat anchors.
        sample_rate: sampling rate in Hz.
        intervals: optional precomputed interval measurements.
        rhythm: optional precomputed rhythm analysis.

    Returns:
        WCTFeatures with missing values set to None when morphology is not measurable.
    """
    source = _source(signal, rhythm_strip)
    measured_intervals = intervals or measure_intervals(signal, rhythm_strip, sample_rate)
    measured_rhythm = rhythm or analyze_rhythm(signal, rhythm_strip, sample_rate)
    anchor_idx, rpeaks = _anchor_rpeaks(source, sample_rate)

    precordial = tuple(
        lead_morphology_from_rpeaks(
            source[lead_idx], rpeaks, sample_rate, _LEAD_NAMES[lead_idx]
        )
        for lead_idx in _PRECORDIAL_LEADS
        if lead_idx < source.shape[0]
    )
    avr = (
        lead_morphology_from_rpeaks(
            source[_AVR_LEAD], rpeaks, sample_rate, _LEAD_NAMES[_AVR_LEAD]
        )
        if _AVR_LEAD < source.shape[0]
        else None
    )
    rs_values = [
        lead.rs_interval_ms for lead in precordial if lead.rs_interval_ms is not None
    ]

    return WCTFeatures(
        heart_rate_bpm=_resolve_hr(measured_intervals, measured_rhythm),
        qrs_ms=measured_intervals.qrs_ms,
        regular=measured_rhythm.regular if measured_rhythm.quality != "unmeasurable" else None,
        n_beats=int(rpeaks.size),
        anchor_lead=_LEAD_NAMES[anchor_idx] if anchor_idx is not None else None,
        precordial_leads=precordial,
        avr=avr,
        max_precordial_rs_interval_ms=max(rs_values) if rs_values else None,
    )


def _source(signal: np.ndarray, rhythm_strip: np.ndarray | None) -> np.ndarray:
    source = rhythm_strip if rhythm_strip is not None else signal
    if source.ndim != 2:
        raise ValueError("ECG signal must have shape (leads, samples)")
    return np.asarray(source, dtype=np.float64)


def _resolve_hr(
    intervals: IntervalMeasurements,
    rhythm: RhythmAnalysis,
) -> float | None:
    if rhythm.heart_rate_bpm is not None:
        return rhythm.heart_rate_bpm
    return intervals.heart_rate_bpm


def _anchor_rpeaks(source: np.ndarray, sample_rate: int) -> tuple[int | None, np.ndarray]:
    ordered = list(_ANCHOR_LEADS) + [
        idx for idx in range(source.shape[0]) if idx not in _ANCHOR_LEADS
    ]
    for lead_idx in ordered:
        if lead_idx >= source.shape[0] or float(np.std(source[lead_idx])) < 0.05:
            continue
        rpeaks = detect_rpeaks(source[lead_idx], sample_rate)
        if rpeaks.size >= 3:
            return lead_idx, rpeaks
    return None, np.empty(0, dtype=int)
