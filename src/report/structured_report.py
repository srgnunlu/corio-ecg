# Structured ECG report (Phase C.3.2 + C.3.3).
#
# Fuses the deterministic measurements (HR, rhythm classification, PR/QRS/QT/QTc)
# with ECGFounder's top diagnoses into one report object, and decides the single
# most clinically useful headline: is this a NORMAL ECG, or abnormal — and why.
#
# Design rule (from the roadmap): the model is the diagnostic engine; this layer
# only organizes and summarizes. The NORMAL determination is intentionally strict
# and explainable — every criterion that must hold is checked and any failure is
# reported in plain language, because telling a clinician "normal" carries weight.

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.measurement.intervals import IntervalMeasurements, interpret_intervals
from src.measurement.rhythm_analysis import RhythmAnalysis
from src.pipeline.diagnose import DiagnosisResult
from src.vtsvt.models import VTSVTAssessment

# Labels that, on their own, do NOT make an ECG abnormal. Rate/rhythm-normal
# descriptors plus the model's own "normal/borderline" verdicts. Sinus brady/
# tachy are rate findings handled by the rate criterion, not pathology here.
_BENIGN_LABELS: frozenset[str] = frozenset(
    {
        "NORMAL SINUS RHYTHM",
        "NORMAL ECG",
        "SINUS RHYTHM",
        "OTHERWISE NORMAL ECG",
        "BORDERLINE ECG",
        "SINUS TACHYCARDIA",
        "SINUS BRADYCARDIA",
        "WITH SINUS ARRHYTHMIA",
        "EARLY REPOLARIZATION",
    }
)

_TOP_DIAGNOSES_COUNT: int = 5


@dataclass
class DiagnosisEntry:
    """One AI diagnosis line in the report."""

    label: str
    probability: float
    above_threshold: bool


@dataclass
class ECGReport:
    """Complete structured ECG interpretation."""

    heart_rate_bpm: float | None
    rhythm_classification: str
    rhythm_basis: str
    rate_category: str
    ectopy_present: bool
    pvc_count: int
    pr_ms: float | None
    qrs_ms: float | None
    qt_ms: float | None
    qtc_ms: float | None
    qtc_formula: str
    interval_flags: dict[str, str]
    top_diagnoses: list[DiagnosisEntry]
    is_normal: bool
    overall_assessment: str       # "Normal ECG" | "Abnormal ECG" | "Indeterminate ECG"
    abnormal_reasons: list[str] = field(default_factory=list)
    normal_criteria: dict[str, bool] = field(default_factory=dict)
    vtsvt_assessment: VTSVTAssessment | None = None


def build_report(
    diagnoses: list[DiagnosisResult],
    intervals: IntervalMeasurements | None,
    rhythm: RhythmAnalysis | None,
    *,
    threshold: float,
    estimated_hr_bpm: float | None = None,
    sex: str | None = None,
    vtsvt_assessment: VTSVTAssessment | None = None,
) -> ECGReport:
    """Assemble a structured report from the pipeline outputs.

    Args:
        diagnoses: all ECGFounder results (sorted or not); top-5 are surfaced.
        intervals: PR/QRS/QT/QTc measurements (may be None / unmeasurable).
        rhythm: rule-based rhythm analysis (may be None).
        threshold: probability cutoff that defines an "active" diagnosis.
        estimated_hr_bpm: HR shown elsewhere in the UI; used when the rhythm
            analysis itself produced no rate (keeps the report consistent).
        sex: "male"/"female" for the QTc cutoff (passed to interval interpretation).
        vtsvt_assessment: optional deterministic WCT/VT criteria audit.
    """
    flags = interpret_intervals(intervals, sex=sex) if intervals is not None else {}
    top = _top_diagnoses(diagnoses, threshold)
    heart_rate = _resolve_hr(rhythm, estimated_hr_bpm)

    criteria, reasons = _evaluate_normality(
        diagnoses, intervals, rhythm, flags, heart_rate, threshold, vtsvt_assessment
    )
    is_normal = all(criteria.values()) and bool(criteria)
    overall = _overall_assessment(is_normal, criteria, rhythm)

    return ECGReport(
        heart_rate_bpm=heart_rate,
        rhythm_classification=rhythm.classification if rhythm else "Undetermined rhythm",
        rhythm_basis=rhythm.rhythm_basis if rhythm else "undetermined",
        rate_category=rhythm.rate_category if rhythm else "unknown",
        ectopy_present=bool(rhythm and rhythm.ectopy_present),
        pvc_count=rhythm.pvc_count if rhythm else 0,
        pr_ms=intervals.pr_ms if intervals else None,
        qrs_ms=intervals.qrs_ms if intervals else None,
        qt_ms=intervals.qt_ms if intervals else None,
        qtc_ms=intervals.qtc_preferred_ms if intervals else None,
        qtc_formula=intervals.qtc_formula if intervals else "n/a",
        interval_flags=flags,
        top_diagnoses=top,
        is_normal=is_normal,
        overall_assessment=overall,
        abnormal_reasons=reasons,
        normal_criteria=criteria,
        vtsvt_assessment=vtsvt_assessment,
    )


def report_to_dict(report: ECGReport) -> dict:
    """Serialize the report to a JSON-ready dict (for future PDF/API output)."""
    payload = asdict(report)
    # Round floats for stable, human-readable JSON.
    for key in ("heart_rate_bpm", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms"):
        value = payload.get(key)
        if isinstance(value, float):
            payload[key] = round(value, 1)
    for entry in payload["top_diagnoses"]:
        entry["probability"] = round(entry["probability"], 4)
    if payload.get("vtsvt_assessment") is not None:
        payload["vtsvt_assessment"] = _tuples_to_lists(payload["vtsvt_assessment"])
    return payload


def _resolve_hr(rhythm: RhythmAnalysis | None, fallback: float | None) -> float | None:
    if rhythm is not None and rhythm.heart_rate_bpm is not None:
        return rhythm.heart_rate_bpm
    return fallback


def _top_diagnoses(
    diagnoses: list[DiagnosisResult],
    threshold: float,
) -> list[DiagnosisEntry]:
    ranked = sorted(diagnoses, key=lambda r: r.probability, reverse=True)
    return [
        DiagnosisEntry(
            label=r.label,
            probability=r.probability,
            above_threshold=r.probability >= threshold,
        )
        for r in ranked[:_TOP_DIAGNOSES_COUNT]
    ]


def _significant_pathologies(
    diagnoses: list[DiagnosisResult],
    threshold: float,
) -> list[str]:
    """Active diagnoses (above threshold) that are not benign descriptors."""
    return [
        r.label
        for r in diagnoses
        if r.probability >= threshold and r.label not in _BENIGN_LABELS
    ]


def _evaluate_normality(
    diagnoses: list[DiagnosisResult],
    intervals: IntervalMeasurements | None,
    rhythm: RhythmAnalysis | None,
    flags: dict[str, str],
    heart_rate: float | None,
    threshold: float,
    vtsvt_assessment: VTSVTAssessment | None,
) -> tuple[dict[str, bool], list[str]]:
    """Check the NORMAL-ECG criteria and collect plain-language failure reasons.

    A NORMAL ECG requires ALL of: sinus rhythm, rate 60-100, normal PR, normal
    QRS, normal QTc, no ectopy, and no significant ECGFounder pathology above
    threshold. An unmeasurable interval blocks a "normal" claim (we will not
    assert what we could not verify) and is reported as such.
    """
    criteria: dict[str, bool] = {}
    reasons: list[str] = []

    # Sinus rhythm.
    sinus = rhythm is not None and rhythm.rhythm_basis == "sinus"
    criteria["sinus_rhythm"] = sinus
    if not sinus:
        label = rhythm.classification if rhythm else "rhythm not analyzable"
        reasons.append(f"Non-sinus or undetermined rhythm ({label})")

    # Rate 60-100.
    normal_rate = heart_rate is not None and 60.0 <= heart_rate <= 100.0
    criteria["normal_rate"] = normal_rate
    if heart_rate is None:
        reasons.append("Heart rate could not be determined")
    elif not normal_rate:
        band = "bradycardia" if heart_rate < 60.0 else "tachycardia"
        reasons.append(f"Abnormal rate: {heart_rate:.0f} bpm ({band})")

    # No ectopy.
    no_ectopy = not (rhythm and rhythm.ectopy_present)
    criteria["no_ectopy"] = no_ectopy
    if not no_ectopy and rhythm is not None:
        reasons.append(f"Ventricular ectopy detected ({rhythm.pvc_count} PVC-like beats)")

    # Interval criteria — each requires a successful measurement.
    criteria["normal_pr"] = _interval_normal(
        intervals.pr_ms if intervals else None, flags.get("pr"), "PR", reasons
    )
    criteria["normal_qrs"] = _interval_normal(
        intervals.qrs_ms if intervals else None, flags.get("qrs"), "QRS", reasons
    )
    criteria["normal_qtc"] = _interval_normal(
        intervals.qtc_preferred_ms if intervals else None,
        flags.get("qtc"),
        "QTc",
        reasons,
    )

    # No significant model-flagged pathology.
    pathologies = _significant_pathologies(diagnoses, threshold)
    no_pathology = not pathologies
    criteria["no_significant_pathology"] = no_pathology
    if not no_pathology:
        shown = ", ".join(pathologies[:3])
        more = "…" if len(pathologies) > 3 else ""
        reasons.append(f"AI-flagged finding(s): {shown}{more}")

    no_vtsvt_vt_support = not (vtsvt_assessment and vtsvt_assessment.supports_vt)
    criteria["no_vtsvt_vt_support"] = no_vtsvt_vt_support
    if not no_vtsvt_vt_support and vtsvt_assessment is not None:
        reasons.append(_vtsvt_reason(vtsvt_assessment))

    return criteria, reasons


def _interval_normal(
    value: float | None,
    flag: str | None,
    name: str,
    reasons: list[str],
) -> bool:
    """A normal interval needs both a measurement and a 'normal' flag."""
    if value is None or flag is None:
        reasons.append(f"{name} interval not measurable")
        return False
    if flag != "normal":
        reasons.append(f"{name}: {flag}")
        return False
    return True


def _overall_assessment(
    is_normal: bool,
    criteria: dict[str, bool],
    rhythm: RhythmAnalysis | None,
) -> str:
    """Headline verdict.

    "Normal ECG" only when every criterion is satisfied. When the sole blockers
    are *unmeasurable* values (not a positive abnormality) and nothing positive
    flags abnormal, we say "Indeterminate" rather than "Abnormal" — we did not
    prove abnormality, we just could not confirm normality.
    """
    if is_normal:
        return "Normal ECG"

    positively_abnormal = (
        not criteria.get("sinus_rhythm", True)
        or not criteria.get("no_ectopy", True)
        or not criteria.get("no_significant_pathology", True)
        or not criteria.get("no_vtsvt_vt_support", True)
        or (rhythm is not None and rhythm.heart_rate_bpm is not None
            and not criteria.get("normal_rate", True))
    )
    return "Abnormal ECG" if positively_abnormal else "Indeterminate ECG"


def _vtsvt_reason(assessment: VTSVTAssessment) -> str:
    evidence = ", ".join(assessment.evidence)
    return f"VT criteria support wide-complex tachycardia as VT ({evidence})"


def _tuples_to_lists(value: object) -> object:
    if isinstance(value, tuple):
        return [_tuples_to_lists(item) for item in value]
    if isinstance(value, list):
        return [_tuples_to_lists(item) for item in value]
    if isinstance(value, dict):
        return {key: _tuples_to_lists(item) for key, item in value.items()}
    return value
