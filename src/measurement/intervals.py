# ECG interval measurement: PR, QRS duration, QT and rate-corrected QTc.
#
# Measurements run on Lead II of the full-duration rhythm strip when available
# (the genuine 10 s strip preserves real beat-to-beat timing, which QTc needs);
# otherwise they fall back to the tiled diagnosis signal, where intra-beat
# morphology is still valid but RR is unreliable. Per-beat fiducials come from
# `delineation`; the reported interval is the median across beats (robust to the
# occasional missed P or T). QTc uses Bazett and Fridericia, with Fridericia
# preferred at rate extremes or irregular rhythms (e.g. atrial fibrillation).

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.measurement.delineation import delineate_beat, detect_rpeaks

TARGET_SAMPLE_RATE: int = 500
# Lead II first (standard rhythm/measurement lead), then precordials, then limbs.
_PREFERRED_LEADS: tuple[int, ...] = (1, 6, 7, 8, 9, 10, 11)
_LEAD_NAMES: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)
_MIN_BEATS: int = 3
# Plausibility bounds (ms) — measurements outside are dropped as delineation errors.
_PR_BOUNDS: tuple[float, float] = (80.0, 400.0)
_QRS_BOUNDS: tuple[float, float] = (40.0, 220.0)
_QT_BOUNDS: tuple[float, float] = (250.0, 700.0)


@dataclass
class IntervalMeasurements:
    """Median interval measurements across the analyzed beats."""

    heart_rate_bpm: float | None
    rr_ms: float | None
    pr_ms: float | None
    qrs_ms: float | None
    qt_ms: float | None
    qtc_bazett_ms: float | None
    qtc_fridericia_ms: float | None
    qtc_preferred_ms: float | None
    qtc_formula: str  # "bazett" | "fridericia" | "n/a"
    measured_lead: str | None
    n_beats: int
    quality: str  # "good" | "low" | "unmeasurable"


def measure_intervals(
    signal: np.ndarray,
    rhythm_strip: np.ndarray | None = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> IntervalMeasurements:
    """Measure PR, QRS, QT and QTc from a 12-lead digitized ECG.

    Args:
        signal: (12, samples) tiled diagnosis signal (fallback source).
        rhythm_strip: optional (12, samples) full-duration strip; preferred
            because it preserves the real RR sequence required for QTc.
        sample_rate: sampling rate in Hz.

    Returns:
        IntervalMeasurements; numeric fields are None when delineation fails.
    """
    source = rhythm_strip if rhythm_strip is not None else signal
    lead_idx = _select_lead(source)
    if lead_idx is None:
        return _empty_result()

    lead = source[lead_idx]
    rpeaks = detect_rpeaks(lead, sample_rate)
    if rpeaks.size < _MIN_BEATS:
        return _empty_result(measured_lead=_LEAD_NAMES[lead_idx])

    rr_samples = float(np.median(np.diff(rpeaks)))
    pr_list: list[float] = []
    qrs_list: list[float] = []
    qt_list: list[float] = []

    for r_peak in rpeaks:
        beat = delineate_beat(lead, int(r_peak), sample_rate, rr_samples)
        _collect(beat.qrs_onset, beat.qrs_offset, qrs_list, sample_rate, _QRS_BOUNDS)
        _collect(beat.p_onset, beat.qrs_onset, pr_list, sample_rate, _PR_BOUNDS)
        _collect(beat.qrs_onset, beat.t_end, qt_list, sample_rate, _QT_BOUNDS)

    rr_ms = rr_samples / sample_rate * 1000.0
    heart_rate = 60000.0 / rr_ms if rr_ms > 0 else None
    pr_ms = _median_or_none(pr_list)
    qrs_ms = _median_or_none(qrs_list)
    qt_ms = _median_or_none(qt_list)

    bazett, fridericia, preferred, formula = _corrected_qt(qt_ms, rpeaks, sample_rate)
    quality = _quality(rpeaks.size, qt_list)

    return IntervalMeasurements(
        heart_rate_bpm=heart_rate,
        rr_ms=rr_ms,
        pr_ms=pr_ms,
        qrs_ms=qrs_ms,
        qt_ms=qt_ms,
        qtc_bazett_ms=bazett,
        qtc_fridericia_ms=fridericia,
        qtc_preferred_ms=preferred,
        qtc_formula=formula,
        measured_lead=_LEAD_NAMES[lead_idx],
        n_beats=int(rpeaks.size),
        quality=quality,
    )


def interpret_intervals(
    measurements: IntervalMeasurements,
    sex: str | None = None,
) -> dict[str, str]:
    """Map numeric intervals to short clinical flags (English keys/values).

    `sex` ("male"/"female") sets the QTc-prolongation cutoff (450 vs 460 ms);
    when unknown the more permissive female cutoff is used to avoid over-flagging.
    """
    flags: dict[str, str] = {}

    if measurements.pr_ms is not None:
        if measurements.pr_ms > 200.0:
            flags["pr"] = "prolonged (>200 ms) — consider first-degree AV block"
        elif measurements.pr_ms < 120.0:
            flags["pr"] = "short (<120 ms) — consider pre-excitation"
        else:
            flags["pr"] = "normal"

    if measurements.qrs_ms is not None:
        if measurements.qrs_ms >= 120.0:
            flags["qrs"] = "wide (>=120 ms) — consider bundle branch block"
        else:
            flags["qrs"] = "normal"

    qtc = measurements.qtc_preferred_ms
    if qtc is not None:
        cutoff = 450.0 if sex == "male" else 460.0
        if qtc > cutoff:
            flags["qtc"] = f"prolonged (>{cutoff:.0f} ms) — review QT-prolonging causes"
        elif qtc < 350.0:
            flags["qtc"] = "short (<350 ms)"
        else:
            flags["qtc"] = "normal"

    return flags


def _select_lead(source: np.ndarray) -> int | None:
    """Pick Lead II if it carries signal, else the cleanest preferred lead."""
    ordered = list(_PREFERRED_LEADS) + [
        i for i in range(source.shape[0]) if i not in _PREFERRED_LEADS
    ]
    for lead_idx in ordered:
        if lead_idx < source.shape[0] and float(np.std(source[lead_idx])) >= 0.1:
            return lead_idx
    return None


def _collect(
    start: int | None,
    end: int | None,
    target: list[float],
    sample_rate: int,
    bounds: tuple[float, float],
) -> None:
    """Append a duration (ms) to target when both fiducials exist and are plausible."""
    if start is None or end is None or end <= start:
        return
    duration_ms = (end - start) / sample_rate * 1000.0
    if bounds[0] <= duration_ms <= bounds[1]:
        target.append(duration_ms)


def _corrected_qt(
    qt_ms: float | None,
    rpeaks: np.ndarray,
    sample_rate: int,
) -> tuple[float | None, float | None, float | None, str]:
    """Compute Bazett and Fridericia QTc; prefer Fridericia at rate extremes/irregularity."""
    if qt_ms is None:
        return None, None, None, "n/a"

    rr_intervals = np.diff(rpeaks) / sample_rate
    rr_sec = float(np.median(rr_intervals))
    if rr_sec <= 0:
        return None, None, None, "n/a"

    qt_sec = qt_ms / 1000.0
    bazett = qt_sec / np.sqrt(rr_sec) * 1000.0
    fridericia = qt_sec / np.cbrt(rr_sec) * 1000.0

    heart_rate = 60.0 / rr_sec
    rr_cv = float(np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-9))
    irregular = rr_cv > 0.15
    rate_extreme = heart_rate < 60.0 or heart_rate > 100.0
    if irregular or rate_extreme:
        return bazett, fridericia, fridericia, "fridericia"
    return bazett, fridericia, bazett, "bazett"


def _quality(n_beats: int, qt_list: list[float]) -> str:
    """Confidence in the measurement from beat count and QT detection rate."""
    if not qt_list:
        return "unmeasurable"
    detection_rate = len(qt_list) / max(n_beats, 1)
    if n_beats >= 6 and detection_rate >= 0.5:
        return "good"
    return "low"


def _median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _empty_result(measured_lead: str | None = None) -> IntervalMeasurements:
    return IntervalMeasurements(
        heart_rate_bpm=None,
        rr_ms=None,
        pr_ms=None,
        qrs_ms=None,
        qt_ms=None,
        qtc_bazett_ms=None,
        qtc_fridericia_ms=None,
        qtc_preferred_ms=None,
        qtc_formula="n/a",
        measured_lead=measured_lead,
        n_beats=0,
        quality="unmeasurable",
    )
