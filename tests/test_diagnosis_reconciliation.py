# Tests for the diagnosis reconciliation layer: contradictory multi-label findings
# must be resolved, but genuinely co-occurring findings must be preserved.

from __future__ import annotations

from src.measurement.rhythm_analysis import RhythmAnalysis
from src.pipeline.diagnose import DiagnosisResult
from src.report.diagnosis_reconciliation import reconcile_diagnoses

_LABEL_INDEX = {
    "NORMAL SINUS RHYTHM": 1,
    "SINUS RHYTHM": 3,
    "ATRIAL FIBRILLATION": 5,
    "RIGHT BUNDLE BRANCH BLOCK": 11,
    "LEFT BUNDLE BRANCH BLOCK": 20,
    "VENTRICULAR TACHYCARDIA": 98,
    "PREMATURE VENTRICULAR COMPLEXES": 9,
    "WITH RAPID VENTRICULAR RESPONSE": 28,
    "NORMAL ECG": 2,
    "ABNORMAL ECG": 0,
}


def _dx(label: str, probability: float) -> DiagnosisResult:
    return DiagnosisResult(label=label, index=_LABEL_INDEX[label], probability=probability)


def _rhythm(basis: str, quality: str = "good", *, p_fraction: float = 0.9) -> RhythmAnalysis:
    """Minimal RhythmAnalysis carrying only the fields reconciliation reads."""
    return RhythmAnalysis(
        classification="(test)",
        rhythm_basis=basis,
        rate_category="normal",
        heart_rate_bpm=75.0,
        rr_mean_ms=800.0,
        rr_cv=0.05,
        rr_rmssd_ms=20.0,
        regular=True,
        p_wave_fraction=p_fraction,
        p_waves_present=p_fraction >= 0.5,
        pvc_count=0,
        ectopy_present=False,
        n_beats=12,
        measured_lead="II",
        quality=quality,
    )


def _labels(results: list[DiagnosisResult]) -> set[str]:
    return {r.label for r in results}


def test_sinus_arbiter_suppresses_atrial_fibrillation() -> None:
    # The reported bug: deterministic sinus + ECGFounder AF above threshold.
    dx = [_dx("NORMAL SINUS RHYTHM", 0.95), _dx("ATRIAL FIBRILLATION", 0.75)]
    result = reconcile_diagnoses(dx, _rhythm("sinus"), threshold=0.7)

    assert "ATRIAL FIBRILLATION" not in _labels(result.kept)
    assert "NORMAL SINUS RHYTHM" in _labels(result.kept)
    assert len(result.suppressed) == 1
    assert result.suppressed[0].label == "ATRIAL FIBRILLATION"
    assert result.suppressed[0].superseded_by == "rhythm analysis"
    assert "sinus" in result.suppressed[0].reason


def test_sinus_arbiter_suppresses_ventricular_tachycardia() -> None:
    dx = [_dx("NORMAL SINUS RHYTHM", 0.9), _dx("VENTRICULAR TACHYCARDIA", 0.8)]
    result = reconcile_diagnoses(dx, _rhythm("sinus"), threshold=0.7)

    assert "VENTRICULAR TACHYCARDIA" not in _labels(result.kept)


def test_af_arbiter_suppresses_sinus_family() -> None:
    dx = [_dx("ATRIAL FIBRILLATION", 0.9), _dx("NORMAL SINUS RHYTHM", 0.8)]
    result = reconcile_diagnoses(dx, _rhythm("atrial_fibrillation", p_fraction=0.1), threshold=0.7)

    assert "NORMAL SINUS RHYTHM" not in _labels(result.kept)
    assert "ATRIAL FIBRILLATION" in _labels(result.kept)


def test_undetermined_rhythm_keeps_critical_label() -> None:
    # Safety: when the rhythm is undetermined the headline claims no rhythm, so a
    # critical label like VT must NOT be suppressed on weak grounds.
    dx = [_dx("NORMAL SINUS RHYTHM", 0.6), _dx("VENTRICULAR TACHYCARDIA", 0.8)]
    result = reconcile_diagnoses(dx, _rhythm("undetermined", quality="low"), threshold=0.7)

    assert "VENTRICULAR TACHYCARDIA" in _labels(result.kept)
    assert result.suppressed == []


def test_low_quality_sinus_does_not_arbitrate() -> None:
    dx = [_dx("NORMAL SINUS RHYTHM", 0.9), _dx("ATRIAL FIBRILLATION", 0.8)]
    result = reconcile_diagnoses(dx, _rhythm("sinus", quality="low"), threshold=0.7)

    assert "ATRIAL FIBRILLATION" in _labels(result.kept)
    assert result.suppressed == []


def test_rbbb_lbbb_keeps_higher_probability() -> None:
    dx = [_dx("RIGHT BUNDLE BRANCH BLOCK", 0.8), _dx("LEFT BUNDLE BRANCH BLOCK", 0.6)]
    result = reconcile_diagnoses(dx, None, threshold=0.5)

    assert "RIGHT BUNDLE BRANCH BLOCK" in _labels(result.kept)
    assert "LEFT BUNDLE BRANCH BLOCK" not in _labels(result.kept)
    assert result.suppressed[0].superseded_by == "RIGHT BUNDLE BRANCH BLOCK"


def test_sinus_and_rbbb_both_kept() -> None:
    # Different groups: sinus rhythm with a bundle branch block is a real combination
    # and must NOT be suppressed.
    dx = [_dx("NORMAL SINUS RHYTHM", 0.95), _dx("RIGHT BUNDLE BRANCH BLOCK", 0.8)]
    result = reconcile_diagnoses(dx, _rhythm("sinus"), threshold=0.7)

    assert _labels(result.kept) == {"NORMAL SINUS RHYTHM", "RIGHT BUNDLE BRANCH BLOCK"}
    assert result.suppressed == []


def test_below_threshold_member_not_suppressed() -> None:
    # A sub-threshold AF never surfaces as a finding, so it is left in the raw list.
    dx = [_dx("NORMAL SINUS RHYTHM", 0.95), _dx("ATRIAL FIBRILLATION", 0.5)]
    result = reconcile_diagnoses(dx, _rhythm("sinus"), threshold=0.7)

    assert "ATRIAL FIBRILLATION" in _labels(result.kept)
    assert result.suppressed == []


def test_ectopy_coexists_with_sinus() -> None:
    # PVCs are ectopy, not a competing primary rhythm — they stay with sinus.
    dx = [_dx("NORMAL SINUS RHYTHM", 0.95), _dx("PREMATURE VENTRICULAR COMPLEXES", 0.8)]
    result = reconcile_diagnoses(dx, _rhythm("sinus"), threshold=0.7)

    assert "PREMATURE VENTRICULAR COMPLEXES" in _labels(result.kept)


def test_af_modifier_survives() -> None:
    # "WITH RAPID VENTRICULAR RESPONSE" qualifies AF; it must not be suppressed.
    dx = [
        _dx("ATRIAL FIBRILLATION", 0.9),
        _dx("WITH RAPID VENTRICULAR RESPONSE", 0.8),
        _dx("NORMAL SINUS RHYTHM", 0.75),
    ]
    result = reconcile_diagnoses(dx, _rhythm("atrial_fibrillation", p_fraction=0.1), threshold=0.7)

    assert "WITH RAPID VENTRICULAR RESPONSE" in _labels(result.kept)
    assert "ATRIAL FIBRILLATION" in _labels(result.kept)
    assert "NORMAL SINUS RHYTHM" not in _labels(result.kept)


def test_global_read_normal_vs_abnormal() -> None:
    dx = [_dx("ABNORMAL ECG", 0.99), _dx("NORMAL ECG", 0.4)]
    result = reconcile_diagnoses(dx, None, threshold=0.3)

    assert "ABNORMAL ECG" in _labels(result.kept)
    assert "NORMAL ECG" not in _labels(result.kept)


def test_no_rhythm_only_static_groups() -> None:
    dx = [_dx("NORMAL SINUS RHYTHM", 0.95), _dx("ATRIAL FIBRILLATION", 0.8)]
    result = reconcile_diagnoses(dx, None, threshold=0.7)

    # No rhythm referee -> rhythm group untouched, both survive.
    assert _labels(result.kept) == {"NORMAL SINUS RHYTHM", "ATRIAL FIBRILLATION"}
