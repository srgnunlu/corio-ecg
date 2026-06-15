"""Tests for controlled digitization experiment regression evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.digitization_regression import (
    compare_digitization_reports,
    load_digitization_regression_config,
)
from src.evaluation.digitization_regression_config import DigitizationRegressionConfig


def _record(
    *,
    category: str = "photos_scans",
    image_id: int = 1,
    correlation: float = 0.90,
    rmse_mv: float = 0.10,
    snr_db: float = 6.0,
    gain_ratio: float = 0.90,
    elapsed_seconds: float = 10.0,
    status: str = "success",
    active_leads: int = 12,
) -> dict[str, object]:
    fidelity = None
    if status == "success":
        fidelity = {
            "median_correlation": correlation,
            "absolute": {
                "median_rmse_mv": rmse_mv,
                "median_snr_db": snr_db,
                "median_gain_ratio": gain_ratio,
            },
        }
    return {
        "category": category,
        "image_id": image_id,
        "ecg_id": f"ecg-{image_id}",
        "source_sha256": f"sha-{image_id}",
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "diagnostics": {
            "layout_cost": 0.2,
            "detected_leads_count": 10,
            "nonzero_leads_count": active_leads,
            "einthoven_score": 0.95,
            "avg_pixel_per_mm": 9.0,
            "raw_lines_count": 4,
        },
        "fidelity": fidelity,
    }


def _report(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "selection_manifest": {"manifest_id": "matched-v1"},
        "records": records,
    }


def _focused_config() -> DigitizationRegressionConfig:
    config = load_digitization_regression_config()
    return DigitizationRegressionConfig(
        version=config.version,
        supported_categories=("photos_scans",),
        target_categories=("photos_bents",),
        tolerances=config.tolerances,
    )


def test_compare_reports_passes_supported_improvement_and_reports_target_category() -> None:
    baseline = _report(
        [
            _record(),
            _record(category="photos_bents", image_id=2, correlation=0.40),
        ]
    )
    candidate = _report(
        [
            _record(correlation=0.92, rmse_mv=0.09, snr_db=7.0, gain_ratio=0.95),
            _record(category="photos_bents", image_id=2, correlation=0.55),
        ]
    )

    result = compare_digitization_reports(baseline, candidate, config=_focused_config())

    assert result["promotion"]["passed"] is True
    assert result["promotion"]["failed_checks"] == []
    assert result["categories"]["photos_scans"]["deltas"]["median_correlation"] == (
        pytest.approx(0.02)
    )
    assert result["categories"]["photos_bents"]["role"] == "target"
    assert result["records"][0]["quality_gate"]["baseline"] == "accept"


def test_compare_reports_blocks_supported_category_regression() -> None:
    baseline = _report([_record()])
    candidate = _report([_record(correlation=0.80)])

    result = compare_digitization_reports(baseline, candidate, config=_focused_config())

    assert result["promotion"]["passed"] is False
    assert result["promotion"]["failed_checks"][0]["category"] == "photos_scans"
    assert result["promotion"]["failed_checks"][0]["metric"] == "median_correlation"


def test_compare_reports_allows_change_exactly_at_tolerance_boundary() -> None:
    baseline = _report([_record()])
    candidate = _report([_record(correlation=0.88)])

    result = compare_digitization_reports(baseline, candidate, config=_focused_config())

    assert result["promotion"]["passed"] is True


def test_compare_reports_counts_failures_and_gate_outcomes() -> None:
    baseline = _report([_record(), _record(image_id=2)])
    candidate = _report([_record(), _record(image_id=2, status="failed")])

    result = compare_digitization_reports(baseline, candidate, config=_focused_config())
    scan_result = result["categories"]["photos_scans"]

    assert scan_result["candidate"]["failure_rate"] == pytest.approx(0.5)
    assert scan_result["candidate"]["quality_gate"]["reject_rate"] == pytest.approx(0.5)
    assert {check["metric"] for check in result["promotion"]["failed_checks"]} >= {
        "failure_rate",
        "quality_gate_reject_rate",
    }


def test_compare_reports_rejects_record_set_mismatch() -> None:
    baseline = _report([_record()])
    candidate = _report([_record(image_id=2)])

    with pytest.raises(ValueError, match="record sets do not match"):
        compare_digitization_reports(baseline, candidate)


def test_compare_reports_rejects_source_image_mismatch() -> None:
    baseline = _report([_record()])
    candidate_record = _record()
    candidate_record["source_sha256"] = "different"

    with pytest.raises(ValueError, match="source SHA-256"):
        compare_digitization_reports(baseline, _report([candidate_record]))


def test_load_digitization_regression_config_rejects_invalid_tolerance(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
version: invalid
supported_categories: [photos_scans]
target_categories: [photos_bents]
tolerances:
  median_correlation_drop: -0.1
  median_rmse_mv_increase: 0.01
  median_snr_db_drop: 1.0
  median_gain_error_increase: 0.05
  failure_rate_increase: 0.0
  median_runtime_increase_fraction: 0.25
  quality_gate_reject_rate_increase: 0.10
""".strip()
    )

    with pytest.raises(ValueError, match="non-negative"):
        load_digitization_regression_config(config_path)
