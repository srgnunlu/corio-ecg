# Tests for structured report assembly + NORMAL-ECG determination.

from src.measurement.intervals import IntervalMeasurements
from src.measurement.rhythm_analysis import RhythmAnalysis
from src.pipeline.diagnose import DiagnosisResult
from src.report.structured_report import build_report, report_to_dict
from src.vtsvt.models import (
    BrugadaCriteriaResult,
    VereckeiCriteriaResult,
    VTSVTAssessment,
    WCTFeatures,
)


def _normal_intervals() -> IntervalMeasurements:
    return IntervalMeasurements(
        heart_rate_bpm=72.0, rr_ms=833.0, pr_ms=160.0, qrs_ms=90.0,
        qt_ms=380.0, qtc_bazett_ms=416.0, qtc_fridericia_ms=405.0,
        qtc_preferred_ms=416.0, qtc_formula="bazett", measured_lead="II",
        n_beats=10, quality="good",
    )


def _sinus_rhythm(bpm: float = 72.0, *, ectopy: int = 0) -> RhythmAnalysis:
    category = "normal" if 60 <= bpm <= 100 else ("bradycardia" if bpm < 60 else "tachycardia")
    return RhythmAnalysis(
        classification="Normal sinus rhythm", rhythm_basis="sinus",
        rate_category=category, heart_rate_bpm=bpm, rr_mean_ms=833.0,
        rr_cv=0.04, rr_rmssd_ms=20.0, regular=True, p_wave_fraction=0.9,
        p_waves_present=True, pvc_count=ectopy, ectopy_present=ectopy > 0,
        n_beats=10, measured_lead="II", quality="good",
    )


def _af_rhythm() -> RhythmAnalysis:
    return RhythmAnalysis(
        classification="Atrial fibrillation", rhythm_basis="atrial_fibrillation",
        rate_category="normal", heart_rate_bpm=88.0, rr_mean_ms=680.0,
        rr_cv=0.25, rr_rmssd_ms=120.0, regular=False, p_wave_fraction=0.1,
        p_waves_present=False, pvc_count=0, ectopy_present=False,
        n_beats=12, measured_lead="II", quality="good",
    )


def _vt_supported_assessment() -> VTSVTAssessment:
    features = WCTFeatures(
        heart_rate_bpm=130.0,
        qrs_ms=160.0,
        regular=True,
        n_beats=16,
        anchor_lead="II",
        precordial_leads=(),
        avr=None,
        max_precordial_rs_interval_ms=128.0,
    )
    brugada = BrugadaCriteriaResult(
        supports_vt=True,
        positive_criteria=("brugada_rs_interval_gt_100ms",),
        rs_absent_all_precordial=False,
        max_rs_interval_ms=128.0,
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
        vi_vt_ratio=1.4,
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


class TestNormalDetection:
    def test_all_normal_yields_normal_ecg(self) -> None:
        diagnoses = [DiagnosisResult("NORMAL SINUS RHYTHM", 1, 0.92)]
        report = build_report(
            diagnoses, _normal_intervals(), _sinus_rhythm(), threshold=0.5
        )
        assert report.is_normal
        assert report.overall_assessment == "Normal ECG"
        assert not report.abnormal_reasons

    def test_pathology_blocks_normal(self) -> None:
        diagnoses = [
            DiagnosisResult("NORMAL SINUS RHYTHM", 1, 0.6),
            DiagnosisResult("ANTERIOR INFARCT", 17, 0.81),
        ]
        report = build_report(
            diagnoses, _normal_intervals(), _sinus_rhythm(), threshold=0.5
        )
        assert not report.is_normal
        assert report.overall_assessment == "Abnormal ECG"
        assert any("ANTERIOR INFARCT" in r for r in report.abnormal_reasons)

    def test_af_blocks_normal(self) -> None:
        diagnoses = [DiagnosisResult("ATRIAL FIBRILLATION", 5, 0.7)]
        report = build_report(diagnoses, _normal_intervals(), _af_rhythm(), threshold=0.5)
        assert not report.is_normal
        assert report.overall_assessment == "Abnormal ECG"
        assert any("rhythm" in r.lower() for r in report.abnormal_reasons)

    def test_prolonged_qtc_blocks_normal(self) -> None:
        intervals = _normal_intervals()
        intervals.qtc_preferred_ms = 500.0
        diagnoses = [DiagnosisResult("NORMAL SINUS RHYTHM", 1, 0.9)]
        report = build_report(diagnoses, intervals, _sinus_rhythm(), threshold=0.5)
        assert not report.is_normal
        assert any("QTc" in r for r in report.abnormal_reasons)

    def test_ectopy_blocks_normal(self) -> None:
        diagnoses = [DiagnosisResult("NORMAL SINUS RHYTHM", 1, 0.9)]
        report = build_report(
            diagnoses, _normal_intervals(), _sinus_rhythm(ectopy=2), threshold=0.5
        )
        assert not report.is_normal
        assert any("ectopy" in r.lower() for r in report.abnormal_reasons)

    def test_unmeasurable_intervals_give_indeterminate(self) -> None:
        # Sinus, normal rate, no pathology, but intervals could not be measured:
        # we cannot claim normal, but nothing positively abnormal -> indeterminate.
        diagnoses = [DiagnosisResult("NORMAL SINUS RHYTHM", 1, 0.6)]
        report = build_report(diagnoses, None, _sinus_rhythm(), threshold=0.5)
        assert not report.is_normal
        assert report.overall_assessment == "Indeterminate ECG"

    def test_vtsvt_vt_support_blocks_normal(self) -> None:
        diagnoses = [DiagnosisResult("NORMAL SINUS RHYTHM", 1, 0.9)]
        report = build_report(
            diagnoses,
            _normal_intervals(),
            _sinus_rhythm(),
            threshold=0.5,
            vtsvt_assessment=_vt_supported_assessment(),
        )
        assert not report.is_normal
        assert report.overall_assessment == "Abnormal ECG"
        assert report.vtsvt_assessment is not None
        assert not report.normal_criteria["no_vtsvt_vt_support"]
        assert any("VT criteria" in reason for reason in report.abnormal_reasons)


class TestReportStructure:
    def test_top_diagnoses_capped_at_five(self) -> None:
        diagnoses = [DiagnosisResult(f"DX{i}", i, 0.9 - i * 0.1) for i in range(8)]
        report = build_report(diagnoses, _normal_intervals(), _sinus_rhythm(), threshold=0.5)
        assert len(report.top_diagnoses) == 5
        # Sorted descending by probability.
        probs = [d.probability for d in report.top_diagnoses]
        assert probs == sorted(probs, reverse=True)

    def test_to_dict_is_json_ready(self) -> None:
        import json

        diagnoses = [DiagnosisResult("NORMAL SINUS RHYTHM", 1, 0.923456)]
        report = build_report(diagnoses, _normal_intervals(), _sinus_rhythm(), threshold=0.5)
        payload = report_to_dict(report)
        # Round-trips through JSON without error.
        text = json.dumps(payload)
        restored = json.loads(text)
        assert restored["overall_assessment"] == "Normal ECG"
        assert restored["heart_rate_bpm"] == 72.0
        assert restored["top_diagnoses"][0]["label"] == "NORMAL SINUS RHYTHM"

    def test_hr_falls_back_to_estimated(self) -> None:
        rhythm = _sinus_rhythm()
        rhythm.heart_rate_bpm = None
        diagnoses = [DiagnosisResult("NORMAL SINUS RHYTHM", 1, 0.9)]
        report = build_report(
            diagnoses, _normal_intervals(), rhythm, threshold=0.5, estimated_hr_bpm=68.0
        )
        assert report.heart_rate_bpm == 68.0

    def test_to_dict_includes_vtsvt_audit(self) -> None:
        report = build_report(
            [DiagnosisResult("NORMAL SINUS RHYTHM", 1, 0.9)],
            _normal_intervals(),
            _sinus_rhythm(),
            threshold=0.5,
            vtsvt_assessment=_vt_supported_assessment(),
        )
        payload = report_to_dict(report)
        assert payload["vtsvt_assessment"]["classification"] == "vt_supported"
        assert payload["vtsvt_assessment"]["evidence"] == ["brugada_rs_interval_gt_100ms"]
