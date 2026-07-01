# Tests for presenting VT/SVT criteria audit in web and PDF outputs.

from __future__ import annotations

from src.report.pdf_report import _styles
from src.report.structured_report import ECGReport
from src.report.vtsvt_pdf import vtsvt_pdf_section
from src.vtsvt.models import (
    BrugadaCriteriaResult,
    VereckeiCriteriaResult,
    VTSVTAssessment,
    WCTFeatures,
)
from src.web.vtsvt_card import format_vtsvt_card


def _vt_supported_assessment() -> VTSVTAssessment:
    features = WCTFeatures(
        heart_rate_bpm=132.0,
        qrs_ms=164.0,
        regular=True,
        n_beats=16,
        anchor_lead="II",
        precordial_leads=(),
        avr=None,
        max_precordial_rs_interval_ms=126.0,
    )
    brugada = BrugadaCriteriaResult(
        supports_vt=True,
        positive_criteria=("brugada_rs_interval_gt_100ms",),
        rs_absent_all_precordial=False,
        max_rs_interval_ms=126.0,
        av_dissociation_present=None,
        capture_or_fusion_beats_present=None,
    )
    vereckei = VereckeiCriteriaResult(
        supports_vt=False,
        positive_criteria=(),
        initial_r_in_avr=False,
        initial_r_or_q_width_gt_40ms=False,
        initial_downstroke_notched=False,
        vi_vt_ratio_leq_1=False,
        vi_vt_ratio=1.3,
    )
    return VTSVTAssessment(
        classification="vt_supported",
        in_scope=True,
        supports_vt=True,
        supports_svt=False,
        evidence=("brugada_rs_interval_gt_100ms",),
        limitations=("av_dissociation_not_assessed",),
        features=features,
        brugada=brugada,
        vereckei=vereckei,
    )


def _report_with_vtsvt() -> ECGReport:
    return ECGReport(
        heart_rate_bpm=132.0,
        rhythm_classification="Undetermined rhythm (tachycardic)",
        rhythm_basis="undetermined",
        rate_category="tachycardia",
        ectopy_present=False,
        pvc_count=0,
        pr_ms=None,
        qrs_ms=164.0,
        qt_ms=None,
        qtc_ms=None,
        qtc_formula="n/a",
        interval_flags={},
        top_diagnoses=[],
        is_normal=False,
        overall_assessment="Abnormal ECG",
        abnormal_reasons=["VT criteria support wide-complex tachycardia as VT"],
        normal_criteria={"no_vtsvt_vt_support": False},
        vtsvt_assessment=_vt_supported_assessment(),
    )


def test_web_card_renders_positive_vt_audit() -> None:
    html = format_vtsvt_card(_vt_supported_assessment())

    assert "Wide-complex tachycardia criteria" in html
    assert "VT supported" in html
    assert "brugada rs interval gt 100ms" in html
    assert "132 bpm" in html
    assert "164 ms" in html
    assert "not a standalone diagnosis" in html


def test_web_card_omits_out_of_scope_audit() -> None:
    assessment = _vt_supported_assessment()
    assessment = VTSVTAssessment(
        classification="not_wide_complex_tachycardia",
        in_scope=False,
        supports_vt=False,
        supports_svt=False,
        evidence=(),
        limitations=assessment.limitations,
        features=assessment.features,
        brugada=assessment.brugada,
        vereckei=assessment.vereckei,
    )

    assert format_vtsvt_card(assessment) == ""
    assert format_vtsvt_card(None) == ""


def test_pdf_section_renders_when_vtsvt_audit_exists() -> None:
    section = vtsvt_pdf_section(_report_with_vtsvt(), _styles())

    assert section


def test_pdf_section_omits_when_no_vtsvt_audit() -> None:
    report = _report_with_vtsvt()
    report.vtsvt_assessment = None

    assert vtsvt_pdf_section(report, _styles()) == []
