"""Tests for the interpretable digitization quality-gate benchmark."""

from pathlib import Path

import pytest

from scripts.evaluate_quality_gate import build_report
from src.training.quality_gate import (
    QualityGateOutcome,
    classify_fidelity_target,
    classify_quality,
    evaluate_quality_gate,
    load_quality_gate_config,
)


def _record(
    *,
    correlation: float = 0.9,
    rmse_mv: float = 0.1,
    snr_db: float = 6.0,
    layout_cost: float = 0.2,
    detected_leads: int = 10,
    active_leads: int = 12,
    einthoven_score: float = 0.95,
    pixel_per_mm: float = 9.0,
    status: str = "success",
) -> dict[str, object]:
    return {
        "category": "photos_test",
        "status": status,
        "diagnostics": {
            "layout_cost": layout_cost,
            "detected_leads_count": detected_leads,
            "nonzero_leads_count": active_leads,
            "einthoven_score": einthoven_score,
            "avg_pixel_per_mm": pixel_per_mm,
        },
        "fidelity": {
            "median_correlation": correlation,
            "absolute": {
                "median_rmse_mv": rmse_mv,
                "median_snr_db": snr_db,
            },
        },
    }


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"correlation": 0.59}, QualityGateOutcome.REJECT),
        ({"snr_db": -0.1}, QualityGateOutcome.REJECT),
        ({"rmse_mv": 0.21}, QualityGateOutcome.REJECT),
        ({"correlation": 0.79}, QualityGateOutcome.WARN),
        ({"snr_db": 2.9}, QualityGateOutcome.WARN),
        ({"rmse_mv": 0.16}, QualityGateOutcome.WARN),
        ({}, QualityGateOutcome.ACCEPT),
    ],
)
def test_classify_fidelity_target_uses_matched_reference_thresholds(
    changes: dict[str, float],
    expected: QualityGateOutcome,
) -> None:
    assert classify_fidelity_target(_record(**changes)) is expected


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"status": "failed"}, QualityGateOutcome.REJECT),
        ({"active_leads": 9}, QualityGateOutcome.REJECT),
        ({"detected_leads": 3}, QualityGateOutcome.REJECT),
        (
            {"einthoven_score": 0.59, "layout_cost": 0.61},
            QualityGateOutcome.REJECT,
        ),
        ({"active_leads": 11}, QualityGateOutcome.WARN),
        ({"einthoven_score": 0.29}, QualityGateOutcome.WARN),
        ({"layout_cost": 1.01}, QualityGateOutcome.WARN),
        ({"pixel_per_mm": 8.79}, QualityGateOutcome.WARN),
        ({"detected_leads": 7}, QualityGateOutcome.WARN),
        ({"einthoven_score": 0.89}, QualityGateOutcome.WARN),
        ({"layout_cost": 0.61}, QualityGateOutcome.WARN),
        ({}, QualityGateOutcome.ACCEPT),
    ],
)
def test_classify_quality_returns_interpretable_gate_outcome(
    changes: dict[str, float | int | str],
    expected: QualityGateOutcome,
) -> None:
    decision = classify_quality(_record(**changes))

    assert decision.outcome is expected
    assert decision.reasons if expected is not QualityGateOutcome.ACCEPT else True


def test_evaluate_quality_gate_reports_false_accepts_and_false_rejects() -> None:
    records = [
        _record(),
        _record(correlation=0.5, einthoven_score=0.95),
        _record(correlation=0.9, active_leads=9),
    ]

    report = evaluate_quality_gate(records)

    assert report["total"] == 3
    assert report["false_accepts"] == 1
    assert report["false_rejects"] == 1
    assert report["missed_rejects"] == 1
    assert report["false_accept_rate"] == pytest.approx(1.0)
    assert report["false_reject_rate"] == pytest.approx(0.5)
    assert report["reject_recall"] == pytest.approx(0.0)
    assert report["categories"]["photos_test"]["total"] == 3


def test_default_quality_gate_config_preserves_current_thresholds() -> None:
    config = load_quality_gate_config()

    assert config.version == "quality-gate-v1"
    assert config.fidelity.reject_correlation_below == pytest.approx(0.60)
    assert config.fidelity.warn_correlation_below == pytest.approx(0.80)
    assert config.inference.reject_active_leads_below == 10
    assert config.inference.warn_einthoven_below == pytest.approx(0.90)


def test_load_quality_gate_config_rejects_invalid_threshold_order(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid_quality_gate.yaml"
    config_path.write_text(
        """
version: invalid
fidelity:
  reject_correlation_below: 0.90
  warn_correlation_below: 0.80
  reject_rmse_mv_above: 0.20
  warn_rmse_mv_above: 0.15
  reject_snr_db_below: 0.0
  warn_snr_db_below: 3.0
inference:
  reject_active_leads_below: 10
  reject_detected_leads_below: 4
  severe_einthoven_below: 0.60
  warn_einthoven_below: 0.90
  layout_cost_above: 0.60
  severe_detected_leads_below: 6
  warn_detected_leads_below: 8
  pixel_per_mm_below: 8.8
  expected_active_leads: 12
  severe_flags_to_reject: 2
""".strip()
    )

    with pytest.raises(ValueError, match="correlation"):
        load_quality_gate_config(config_path)


def test_load_quality_gate_config_rejects_missing_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "incomplete_quality_gate.yaml"
    config_path.write_text("version: incomplete")

    with pytest.raises(ValueError, match="invalid quality-gate config structure"):
        load_quality_gate_config(config_path)


def test_build_report_contains_resolved_quality_gate_config() -> None:
    config = load_quality_gate_config()

    report = build_report({"records": [_record()]}, config=config)

    assert report["gate_version"] == config.version
    assert report["quality_gate_config"]["fidelity"]["reject_correlation_below"] == 0.60
    assert report["quality_gate_config"]["inference"]["reject_active_leads_below"] == 10
