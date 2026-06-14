"""Tests for quality-gate benchmark artifact safety and provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.evaluate_quality_gate import (
    EvaluationPurpose,
    EvaluationStage,
    build_report,
    calculate_file_sha256,
    validate_source_hash,
    write_report,
)


def _source_report() -> dict[str, object]:
    return {
        "selection_manifest": {"name": "test-manifest"},
        "records": [
            {
                "category": "photos_test",
                "image_id": "image-1",
                "ecg_id": "ecg-1",
                "status": "success",
                "diagnostics": {
                    "layout_cost": 0.2,
                    "detected_leads_count": 10,
                    "nonzero_leads_count": 12,
                    "einthoven_score": 0.95,
                    "avg_pixel_per_mm": 9.0,
                    "raw_lines_count": 4,
                },
                "fidelity": {
                    "median_correlation": 0.9,
                    "absolute": {
                        "median_rmse_mv": 0.1,
                        "median_snr_db": 6.0,
                    },
                },
            }
        ],
    }


def test_calculate_file_sha256_returns_content_digest(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_bytes(b"quality-gate-source")

    digest = calculate_file_sha256(source_path)

    assert digest == hashlib.sha256(b"quality-gate-source").hexdigest()


def test_validate_source_hash_rejects_mismatch(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text("{}")

    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        validate_source_hash(source_path, expected_sha256="incorrect")


def test_build_report_records_evaluation_stage_and_source_hash() -> None:
    report = build_report(
        _source_report(),
        evaluation_stage=EvaluationStage.HOLDOUT,
        source_sha256="source-digest",
    )

    assert report["evaluation_stage"] == "holdout"
    assert report["source_sha256"] == "source-digest"
    assert report["quality_feature_contract"]["version"] == "quality-feature-contract-v1"
    assert "reason_codes" in report["records"][0]


def test_build_report_records_split_evaluation_context() -> None:
    report = build_report(
        _source_report(),
        evaluation_stage=EvaluationStage.HOLDOUT,
        evaluation_purpose=EvaluationPurpose.LOCKED_EVALUATION,
        selected_split="test",
        split_manifest_id="manifest-id",
        split_evidence_status="pre-registered",
    )

    assert report["evaluation_purpose"] == "locked-evaluation"
    assert report["selected_split"] == "test"
    assert report["split_manifest_id"] == "manifest-id"
    assert report["split_evidence_status"] == "pre-registered"
    assert report["sample_size_warning"] is not None


def test_write_report_refuses_to_overwrite_existing_artifact(tmp_path: Path) -> None:
    report = build_report(_source_report())
    write_report(report, tmp_path)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_report(report, tmp_path)


def test_write_report_allows_explicit_overwrite(tmp_path: Path) -> None:
    report = build_report(_source_report())
    write_report(report, tmp_path)

    json_path, csv_path = write_report(report, tmp_path, allow_overwrite=True)

    assert json_path.exists()
    assert csv_path.exists()


def test_write_report_uses_repository_safe_csv_line_endings(tmp_path: Path) -> None:
    report = build_report(_source_report())

    _, csv_path = write_report(report, tmp_path)

    assert b"\r\n" not in csv_path.read_bytes()
