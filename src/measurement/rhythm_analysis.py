# Rhythm-level classification for digitized paper ECGs (Phase C.3.1).
#
# Works off the full-duration rhythm strip (the genuine 10 s Lead II recovered in
# C.1) when available — that is the only source carrying a real beat-to-beat RR
# sequence, which every rhythm call (regularity, AF, ectopy) depends on. The tiled
# diagnosis signal repeats a ~2.5 s segment and so is rhythm-blind; it is only a
# last-resort fallback for rate category.
#
# Scope is deliberately narrow and rule-based (scipy/numpy only): sinus vs atrial
# fibrillation, rate category (brady/normal/tachy), and a premature-ventricular
# (PVC) ectopy flag. ECGFounder's 150-class output remains the primary diagnostic
# source; this module adds the interpretable rhythm layer a clinician expects
# alongside it.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.measurement.delineation import delineate_beat, detect_rpeaks

TARGET_SAMPLE_RATE: int = 500
# Lead II first (standard rhythm lead), then precordials, then limbs.
_PREFERRED_LEADS: tuple[int, ...] = (1, 6, 7, 8, 9, 10, 11)
_LEAD_NAMES: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)
_MIN_BEATS: int = 4

# Rate thresholds (bpm) — standard adult cutoffs.
_BRADY_BPM: float = 60.0
_TACHY_BPM: float = 100.0

# Regularity: RR coefficient of variation. Sinus rhythm is regular (low CV);
# atrial fibrillation is "irregularly irregular" (high CV). 0.12 separates the
# physiologic sinus-arrhythmia band from genuine irregularity in practice.
_REGULAR_CV: float = 0.12
# AF needs sustained irregularity; a couple of ectopics should not trip it.
_AF_CV: float = 0.16
# P-wave presence below this fraction of beats supports an absent-P rhythm (AF).
_P_ABSENT_FRACTION: float = 0.35
_P_PRESENT_FRACTION: float = 0.50

# PVC heuristics (relative — the signal is z-scored so absolute mV is meaningless).
_PVC_PREMATURITY: float = 0.80   # beat arrives early: RR_prev < 0.80 * median RR
_PVC_WIDE_FACTOR: float = 1.35   # QRS markedly wider than the dominant beat
_PVC_MIN_WIDE_MS: float = 110.0  # ...and at least borderline-wide in absolute terms


@dataclass
class RhythmAnalysis:
    """Rule-based rhythm classification from a single rhythm-strip lead."""

    classification: str          # human-readable, e.g. "Sinus bradycardia"
    rhythm_basis: str            # "sinus" | "atrial_fibrillation" | "undetermined"
    rate_category: str           # "bradycardia" | "normal" | "tachycardia" | "unknown"
    heart_rate_bpm: float | None
    rr_mean_ms: float | None
    rr_cv: float | None          # coefficient of variation of RR intervals
    rr_rmssd_ms: float | None    # beat-to-beat variability
    regular: bool
    p_wave_fraction: float       # fraction of beats with a detectable P wave
    p_waves_present: bool
    pvc_count: int
    ectopy_present: bool
    n_beats: int
    measured_lead: str | None
    quality: str                 # "good" | "low" | "unmeasurable"


def analyze_rhythm(
    signal: np.ndarray,
    rhythm_strip: np.ndarray | None = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> RhythmAnalysis:
    """Classify the cardiac rhythm from a 12-lead digitized ECG.

    Args:
        signal: (12, samples) tiled diagnosis signal (fallback source).
        rhythm_strip: optional (12, samples) full-duration strip; strongly
            preferred because it carries the real RR sequence.
        sample_rate: sampling rate in Hz.

    Returns:
        RhythmAnalysis; fields are conservative (undetermined / None) whenever
        too few beats are detected to support a confident call.
    """
    source = rhythm_strip if rhythm_strip is not None else signal
    lead_idx = _select_lead(source)
    if lead_idx is None:
        return _empty_result()

    lead = source[lead_idx]
    rpeaks = detect_rpeaks(lead, sample_rate)
    if rpeaks.size < _MIN_BEATS:
        return _empty_result(measured_lead=_LEAD_NAMES[lead_idx])

    rr_samples = np.diff(rpeaks).astype(np.float64)
    rr_median = float(np.median(rr_samples))
    rr_mean_ms = float(np.mean(rr_samples)) / sample_rate * 1000.0
    rr_cv = float(np.std(rr_samples) / (np.mean(rr_samples) + 1e-9))
    rr_rmssd_ms = float(np.sqrt(np.mean(np.diff(rr_samples) ** 2))) / sample_rate * 1000.0
    heart_rate = 60.0 * sample_rate / rr_median if rr_median > 0 else None

    p_fraction = _p_wave_fraction(lead, rpeaks, sample_rate, rr_median)
    pvc_count = _count_pvcs(lead, rpeaks, sample_rate, rr_median)

    regular = rr_cv < _REGULAR_CV
    p_waves_present = p_fraction >= _P_PRESENT_FRACTION
    rate_category = _rate_category(heart_rate)
    basis = _rhythm_basis(rr_cv, p_fraction, regular)
    classification = _classification(basis, rate_category, heart_rate)
    quality = _quality(rpeaks.size, p_fraction)

    return RhythmAnalysis(
        classification=classification,
        rhythm_basis=basis,
        rate_category=rate_category,
        heart_rate_bpm=heart_rate,
        rr_mean_ms=rr_mean_ms,
        rr_cv=rr_cv,
        rr_rmssd_ms=rr_rmssd_ms,
        regular=regular,
        p_wave_fraction=p_fraction,
        p_waves_present=p_waves_present,
        pvc_count=pvc_count,
        ectopy_present=pvc_count > 0,
        n_beats=int(rpeaks.size),
        measured_lead=_LEAD_NAMES[lead_idx],
        quality=quality,
    )


def _rhythm_basis(rr_cv: float, p_fraction: float, regular: bool) -> str:
    """Decide the underlying rhythm: sinus, atrial fibrillation, or undetermined.

    AF is the classic "irregularly irregular RR with no organized P wave"; both
    conditions must hold so that PVC-driven irregularity (P waves still present)
    is not mislabelled. A regular rhythm with reliable P waves is sinus. Anything
    in between is left undetermined rather than guessed.
    """
    if rr_cv >= _AF_CV and p_fraction < _P_ABSENT_FRACTION:
        return "atrial_fibrillation"
    if regular and p_fraction >= _P_PRESENT_FRACTION:
        return "sinus"
    return "undetermined"


def _classification(basis: str, rate_category: str, heart_rate: float | None) -> str:
    """Compose the human-readable rhythm label from basis + rate."""
    if basis == "atrial_fibrillation":
        if rate_category == "tachycardia":
            return "Atrial fibrillation with rapid ventricular response"
        if rate_category == "bradycardia":
            return "Atrial fibrillation with slow ventricular response"
        return "Atrial fibrillation"
    if basis == "sinus":
        if rate_category == "bradycardia":
            return "Sinus bradycardia"
        if rate_category == "tachycardia":
            return "Sinus tachycardia"
        return "Normal sinus rhythm"
    # Undetermined — still surface the rate so the line is informative.
    if rate_category == "tachycardia":
        return "Undetermined rhythm (tachycardic)"
    if rate_category == "bradycardia":
        return "Undetermined rhythm (bradycardic)"
    return "Undetermined rhythm"


def _rate_category(heart_rate: float | None) -> str:
    if heart_rate is None:
        return "unknown"
    if heart_rate < _BRADY_BPM:
        return "bradycardia"
    if heart_rate > _TACHY_BPM:
        return "tachycardia"
    return "normal"


def _p_wave_fraction(
    lead: np.ndarray,
    rpeaks: np.ndarray,
    sample_rate: int,
    rr_median: float,
) -> float:
    """Fraction of beats with a P wave detectable just before QRS onset."""
    detected = 0
    counted = 0
    for r_peak in rpeaks:
        beat = delineate_beat(lead, int(r_peak), sample_rate, rr_median)
        if beat.qrs_onset is None:
            continue
        counted += 1
        if beat.p_onset is not None:
            detected += 1
    if counted == 0:
        return 0.0
    return detected / counted


def _count_pvcs(
    lead: np.ndarray,
    rpeaks: np.ndarray,
    sample_rate: int,
    rr_median: float,
) -> int:
    """Count premature beats with an abnormally wide QRS (PVC signature).

    A PVC arrives early (short preceding RR) and is conducted abnormally, so its
    QRS is markedly wider than the dominant beat. Both must hold: prematurity
    alone is a PAC, width alone is a bundle-branch pattern. Widths are compared
    relative to the patient's own median QRS (z-scored amplitude makes absolute
    thresholds meaningless), with a soft absolute floor to reject noise.
    """
    widths: list[float | None] = []
    for r_peak in rpeaks:
        beat = delineate_beat(lead, int(r_peak), sample_rate, rr_median)
        if beat.qrs_onset is None or beat.qrs_offset is None:
            widths.append(None)
            continue
        widths.append((beat.qrs_offset - beat.qrs_onset) / sample_rate * 1000.0)

    valid = [w for w in widths if w is not None]
    if len(valid) < _MIN_BEATS:
        return 0
    median_width = float(np.median(valid))
    rr_samples = np.diff(rpeaks).astype(np.float64)

    pvc_count = 0
    for beat_idx in range(1, len(rpeaks)):
        width = widths[beat_idx]
        if width is None:
            continue
        prev_rr = rr_samples[beat_idx - 1]
        premature = prev_rr < _PVC_PREMATURITY * rr_median
        wide = width > _PVC_WIDE_FACTOR * median_width and width >= _PVC_MIN_WIDE_MS
        if premature and wide:
            pvc_count += 1
    return pvc_count


def _select_lead(source: np.ndarray) -> int | None:
    """Pick Lead II if it carries signal, else the cleanest preferred lead."""
    ordered = list(_PREFERRED_LEADS) + [
        i for i in range(source.shape[0]) if i not in _PREFERRED_LEADS
    ]
    for lead_idx in ordered:
        if lead_idx < source.shape[0] and float(np.std(source[lead_idx])) >= 0.1:
            return lead_idx
    return None


def _quality(n_beats: int, p_fraction: float) -> str:
    """Confidence from beat count (more beats = more reliable RR statistics)."""
    if n_beats < _MIN_BEATS:
        return "unmeasurable"
    if n_beats >= 8:
        return "good"
    return "low"


def _empty_result(measured_lead: str | None = None) -> RhythmAnalysis:
    return RhythmAnalysis(
        classification="Undetermined rhythm",
        rhythm_basis="undetermined",
        rate_category="unknown",
        heart_rate_bpm=None,
        rr_mean_ms=None,
        rr_cv=None,
        rr_rmssd_ms=None,
        regular=False,
        p_wave_fraction=0.0,
        p_waves_present=False,
        pvc_count=0,
        ectopy_present=False,
        n_beats=0,
        measured_lead=measured_lead,
        quality="unmeasurable",
    )
