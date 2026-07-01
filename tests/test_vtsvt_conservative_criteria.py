# Tests for conservative VT/SVT criteria guardrails.

from __future__ import annotations

from src.vtsvt.criteria import assess_vtsvt
from src.vtsvt.models import LeadMorphology, WCTFeatures
from src.vtsvt.terminal_morphology import evaluate_terminal_v1_v6_morphology
from tests.test_vtsvt_criteria import _intervals, _rhythm, _signal_with_qrs


def test_borderline_qrs_suppresses_automated_morphology_support() -> None:
    signal = _signal_with_qrs(
        precordial_points=[(0.0, 0.0), (20.0, 1.2), (150.0, -1.0), (175.0, 0.0)]
    )

    result = assess_vtsvt(
        signal,
        intervals=_intervals(qrs_ms=132.0),
        rhythm=_rhythm(bpm=128.0),
    )

    assert result.in_scope
    assert result.classification == "indeterminate_wide_complex_tachycardia"
    assert "brugada_rs_interval_gt_100ms" not in result.evidence
    assert "automated_morphology_support_requires_qrs_at_least_140ms" in result.limitations


def test_inconsistent_rs_interval_is_not_counted_as_vt_support() -> None:
    signal = _signal_with_qrs(
        precordial_points=[(0.0, 0.0), (20.0, 1.2), (220.0, -1.0), (245.0, 0.0)]
    )

    result = assess_vtsvt(
        signal,
        intervals=_intervals(qrs_ms=150.0),
        rhythm=_rhythm(bpm=128.0),
    )

    assert result.features.max_precordial_rs_interval_ms is not None
    assert result.features.max_precordial_rs_interval_ms > 170.0
    assert "brugada_rs_interval_gt_100ms" not in result.evidence
    assert "rs_interval_exceeds_measured_qrs_consistency" in result.limitations


def test_terminal_rbbb_v6_rsr_ratio_is_not_a_conservative_vt_hit() -> None:
    features = WCTFeatures(
        heart_rate_bpm=132.0,
        qrs_ms=150.0,
        regular=True,
        n_beats=8,
        anchor_lead="II",
        precordial_leads=(
            _lead_morphology("V1", qrs_pattern="r", dominant_polarity="positive"),
            _lead_morphology("V6", qrs_pattern="rsr", r_s_ratio=0.7),
        ),
        avr=None,
        max_precordial_rs_interval_ms=82.0,
    )

    supports_vt, criteria = evaluate_terminal_v1_v6_morphology(features)

    assert supports_vt is False
    assert criteria == ()


def _lead_morphology(
    lead: str,
    *,
    qrs_pattern: str,
    dominant_polarity: str = "positive",
    r_s_ratio: float | None = None,
) -> LeadMorphology:
    return LeadMorphology(
        lead=lead,
        qrs_duration_ms=150.0,
        has_rs_complex=True,
        rs_interval_ms=82.0,
        initial_deflection="r",
        initial_deflection_width_ms=20.0,
        dominant_polarity=dominant_polarity,
        vi_vt_ratio=1.4,
        initial_downstroke_notched=False,
        analyzed_beats=8,
        qrs_pattern=qrs_pattern,
        r_s_ratio=r_s_ratio,
        qrs_onset_to_s_nadir_ms=52.0,
        initial_r_taller_than_terminal_r=None,
        s_downstroke_notched=False,
    )
