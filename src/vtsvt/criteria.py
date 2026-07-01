# Conservative VT/SVT assessment from deterministic wide-complex criteria.

from __future__ import annotations

import numpy as np

from src.measurement.intervals import IntervalMeasurements
from src.measurement.rhythm_analysis import RhythmAnalysis
from src.vtsvt.features import TARGET_SAMPLE_RATE, extract_wct_features
from src.vtsvt.models import (
    BrugadaCriteriaResult,
    VereckeiCriteriaResult,
    VTSVTAssessment,
    WCTFeatures,
)
from src.vtsvt.terminal_morphology import evaluate_terminal_v1_v6_morphology

_TACHYCARDIA_BPM: float = 100.0
_WIDE_QRS_MS: float = 120.0
_AUTOMATED_MORPHOLOGY_QRS_MS: float = 140.0
_BRUGADA_RS_MS: float = 100.0
_RS_QRS_TOLERANCE_MS: float = 20.0
_VERECKEI_INITIAL_MS: float = 40.0
_MIN_BEATS: int = 3


def assess_vtsvt(
    signal: np.ndarray,
    rhythm_strip: np.ndarray | None = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    intervals: IntervalMeasurements | None = None,
    rhythm: RhythmAnalysis | None = None,
    av_dissociation_present: bool | None = None,
    capture_or_fusion_beats_present: bool | None = None,
) -> VTSVTAssessment:
    """Assess wide-complex tachycardia morphology without making SVT claims.

    Args:
        signal: (12, samples) digitized ECG.
        rhythm_strip: optional full-duration rhythm strip.
        sample_rate: sampling rate in Hz.
        intervals: optional precomputed PR/QRS/QT measurements.
        rhythm: optional precomputed rhythm analysis.
        av_dissociation_present: externally adjudicated AV dissociation flag.
        capture_or_fusion_beats_present: externally adjudicated capture/fusion flag.

    Returns:
        VTSVTAssessment. Positive criteria support VT; absent criteria leave the
        WCT indeterminate rather than calling SVT by exclusion.
    """
    features = extract_wct_features(
        signal,
        rhythm_strip,
        sample_rate,
        intervals=intervals,
        rhythm=rhythm,
    )
    brugada = _evaluate_brugada(
        features,
        av_dissociation_present,
        capture_or_fusion_beats_present,
    )
    vereckei = _evaluate_vereckei(features)
    in_scope, out_of_scope_classification = _scope(features)

    evidence: tuple[str, ...] = ()
    if in_scope:
        evidence = brugada.positive_criteria + vereckei.positive_criteria
    supports_vt = bool(evidence)
    classification = _classification(in_scope, out_of_scope_classification, supports_vt)

    return VTSVTAssessment(
        classification=classification,
        in_scope=in_scope,
        supports_vt=supports_vt,
        supports_svt=False,
        evidence=evidence,
        limitations=_limitations(
            features,
            in_scope,
            av_dissociation_present,
            capture_or_fusion_beats_present,
        ),
        features=features,
        brugada=brugada,
        vereckei=vereckei,
    )


def _scope(features: WCTFeatures) -> tuple[bool, str]:
    if (
        features.heart_rate_bpm is None
        or features.qrs_ms is None
        or features.n_beats < _MIN_BEATS
    ):
        return False, "unmeasurable"
    if features.heart_rate_bpm <= _TACHYCARDIA_BPM or features.qrs_ms < _WIDE_QRS_MS:
        return False, "not_wide_complex_tachycardia"
    return True, ""


def _classification(
    in_scope: bool,
    out_of_scope_classification: str,
    supports_vt: bool,
) -> str:
    if not in_scope:
        return out_of_scope_classification
    if supports_vt:
        return "vt_supported"
    return "indeterminate_wide_complex_tachycardia"


def _evaluate_brugada(
    features: WCTFeatures,
    av_dissociation_present: bool | None,
    capture_or_fusion_beats_present: bool | None,
) -> BrugadaCriteriaResult:
    criteria: list[str] = []
    analyzable = [
        lead for lead in features.precordial_leads if lead.analyzed_beats > 0
    ]
    rs_absent = len(analyzable) == 6 and all(not lead.has_rs_complex for lead in analyzable)
    morphology_eligible = _automated_morphology_eligible(features)

    if morphology_eligible and rs_absent:
        criteria.append("brugada_absent_rs_all_precordial")
    if (
        morphology_eligible
        and features.max_precordial_rs_interval_ms is not None
        and features.max_precordial_rs_interval_ms > _BRUGADA_RS_MS
        and _rs_interval_consistent_with_qrs(features)
    ):
        criteria.append("brugada_rs_interval_gt_100ms")
    if av_dissociation_present:
        criteria.append("brugada_av_dissociation")
    if capture_or_fusion_beats_present:
        criteria.append("brugada_capture_or_fusion_beats")

    terminal_supports_vt, terminal_criteria = evaluate_terminal_v1_v6_morphology(features)
    if morphology_eligible and terminal_supports_vt:
        criteria.extend(terminal_criteria)

    return BrugadaCriteriaResult(
        supports_vt=bool(criteria),
        positive_criteria=tuple(criteria),
        rs_absent_all_precordial=rs_absent,
        max_rs_interval_ms=features.max_precordial_rs_interval_ms,
        av_dissociation_present=av_dissociation_present,
        capture_or_fusion_beats_present=capture_or_fusion_beats_present,
        terminal_morphology_suggests_vt=terminal_supports_vt,
        terminal_morphology_criteria=terminal_criteria,
    )


def _evaluate_vereckei(features: WCTFeatures) -> VereckeiCriteriaResult:
    avr = features.avr
    if avr is None or avr.analyzed_beats == 0:
        return VereckeiCriteriaResult(False, (), None, None, None, None, None)

    initial_r = avr.initial_deflection == "r"
    wide_initial = (
        avr.initial_deflection_width_ms is not None
        and avr.initial_deflection_width_ms > _VERECKEI_INITIAL_MS
    )
    notched = avr.initial_downstroke_notched
    ratio_leq_1 = avr.vi_vt_ratio is not None and avr.vi_vt_ratio <= 1.0

    criteria: list[str] = []
    if _automated_morphology_eligible(features):
        if initial_r:
            criteria.append("vereckei_initial_r_in_avr")
        if wide_initial:
            criteria.append("vereckei_initial_r_or_q_gt_40ms")
        if notched:
            criteria.append("vereckei_initial_downstroke_notching")
        if ratio_leq_1:
            criteria.append("vereckei_vi_vt_ratio_leq_1")

    return VereckeiCriteriaResult(
        supports_vt=bool(criteria),
        positive_criteria=tuple(criteria),
        initial_r_in_avr=initial_r,
        initial_r_or_q_width_gt_40ms=wide_initial,
        initial_downstroke_notched=notched,
        vi_vt_ratio_leq_1=ratio_leq_1,
        vi_vt_ratio=avr.vi_vt_ratio,
    )


def _limitations(
    features: WCTFeatures,
    in_scope: bool,
    av_dissociation_present: bool | None,
    capture_or_fusion_beats_present: bool | None,
) -> tuple[str, ...]:
    limitations: list[str] = []
    if not in_scope:
        limitations.append("criteria_apply_only_to_tachycardia_with_qrs_at_least_120ms")
    if features.regular is False:
        limitations.append("irregular_rhythm_algorithms_validated_for_regular_wct")
    if in_scope and not _automated_morphology_eligible(features):
        limitations.append("automated_morphology_support_requires_qrs_at_least_140ms")
    if _rs_interval_candidate(features) and not _rs_interval_consistent_with_qrs(features):
        limitations.append("rs_interval_exceeds_measured_qrs_consistency")
    if av_dissociation_present is None:
        limitations.append("av_dissociation_not_assessed")
    if capture_or_fusion_beats_present is None:
        limitations.append("capture_or_fusion_beats_not_assessed")
    if evaluate_terminal_v1_v6_morphology(features)[0] is None:
        limitations.append("terminal_v1_v6_morphology_not_measurable")
    return tuple(limitations)


def _automated_morphology_eligible(features: WCTFeatures) -> bool:
    return features.qrs_ms is not None and features.qrs_ms >= _AUTOMATED_MORPHOLOGY_QRS_MS


def _rs_interval_candidate(features: WCTFeatures) -> bool:
    return (
        features.max_precordial_rs_interval_ms is not None
        and features.max_precordial_rs_interval_ms > _BRUGADA_RS_MS
    )


def _rs_interval_consistent_with_qrs(features: WCTFeatures) -> bool:
    if features.max_precordial_rs_interval_ms is None or features.qrs_ms is None:
        return False
    return features.max_precordial_rs_interval_ms <= features.qrs_ms + _RS_QRS_TOLERANCE_MS
