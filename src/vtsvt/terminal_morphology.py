# Brugada terminal V1/V6 morphology criteria for wide-complex tachycardia.

from __future__ import annotations

from typing import TypeGuard

from src.vtsvt.models import LeadMorphology, WCTFeatures

_LBBB_INITIAL_R_MS: float = 30.0
_LBBB_S_NADIR_MS: float = 60.0


def evaluate_terminal_v1_v6_morphology(
    features: WCTFeatures,
) -> tuple[bool | None, tuple[str, ...]]:
    """Evaluate Brugada terminal morphology criteria from V1 and V6 features.

    Returns:
        A tuple of (supports_vt, positive_criteria). supports_vt is None when
        V1/V6 morphology is not measurable enough for this criterion.
    """
    v1 = _lead(features, "V1")
    v6 = _lead(features, "V6")
    if not _measurable(v1):
        return None, ()

    if v1.dominant_polarity == "positive":
        if not _measurable(v6):
            return None, ()
        v1_hits = _rbbb_v1_hits(v1)
        v6_hits = _rbbb_v6_hits(v6)
        if v1_hits and v6_hits:
            return True, ("brugada_terminal_v1_v6_rbbb_morphology",)
        return False, ()

    if v1.dominant_polarity == "negative":
        criteria: list[str] = []
        if _lbbb_v1_hits(v1):
            criteria.append("brugada_terminal_v1_lbbb_morphology")
        if _measurable(v6) and _lbbb_v6_hits(v6):
            criteria.append("brugada_terminal_v6_lbbb_q_wave")
        if criteria:
            return True, tuple(criteria)
        if not _measurable(v6):
            return None, ()
        return False, ()

    return None, ()


def _lead(features: WCTFeatures, lead_name: str) -> LeadMorphology | None:
    for lead in features.precordial_leads:
        if lead.lead == lead_name:
            return lead
    return None


def _measurable(lead: LeadMorphology | None) -> TypeGuard[LeadMorphology]:
    return lead is not None and lead.analyzed_beats > 0 and lead.qrs_pattern is not None


def _rbbb_v1_hits(lead: LeadMorphology) -> tuple[str, ...]:
    hits: list[str] = []
    if lead.qrs_pattern == "r":
        hits.append("v1_monophasic_r")
    if lead.qrs_pattern == "qr":
        hits.append("v1_qr")
    if lead.qrs_pattern == "rs" and _ratio_gte(lead.r_s_ratio, 1.0):
        hits.append("v1_rs_positive_dominant")
    if lead.initial_r_taller_than_terminal_r:
        hits.append("v1_taller_left_rabbit_ear")
    return tuple(hits)


def _rbbb_v6_hits(lead: LeadMorphology) -> tuple[str, ...]:
    hits: list[str] = []
    if lead.qrs_pattern == "r":
        hits.append("v6_monophasic_r")
    if lead.qrs_pattern == "qr":
        hits.append("v6_qr")
    if lead.qrs_pattern == "qs":
        hits.append("v6_qs")
    if lead.qrs_pattern == "rs" and _ratio_lt(lead.r_s_ratio, 1.0):
        hits.append("v6_r_s_ratio_lt_1")
    return tuple(hits)


def _lbbb_v1_hits(lead: LeadMorphology) -> tuple[str, ...]:
    hits: list[str] = []
    if (
        lead.initial_deflection == "r"
        and lead.initial_deflection_width_ms is not None
        and lead.initial_deflection_width_ms > _LBBB_INITIAL_R_MS
    ):
        hits.append("v1_initial_r_gt_30ms")
    if (
        lead.qrs_onset_to_s_nadir_ms is not None
        and lead.qrs_onset_to_s_nadir_ms > _LBBB_S_NADIR_MS
    ):
        hits.append("v1_qrs_onset_to_s_nadir_gt_60ms")
    if lead.s_downstroke_notched:
        hits.append("v1_s_downstroke_notched")
    return tuple(hits)


def _lbbb_v6_hits(lead: LeadMorphology) -> tuple[str, ...]:
    if lead.qrs_pattern in {"qr", "qs", "qrs"}:
        return ("v6_q_wave",)
    return ()


def _ratio_lt(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold


def _ratio_gte(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold
