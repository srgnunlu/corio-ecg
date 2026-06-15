"""Tests for digitization experiment report artifact handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_digitization_experiment import build_report, write_report


def _comparison() -> dict[str, object]:
    return {
        "promotion": {"passed": True, "failed_checks": []},
        "overall": {"deltas": {"median_correlation": 0.01}},
        "categories": {},
        "records": [
            {
                "category": "photos_scans",
                "image_id": 1,
                "ecg_id": "ecg-1",
                "status_changed": False,
                "baseline_status": "success",
                "candidate_status": "success",
                "median_correlation_delta": 0.01,
            }
        ],
    }


def test_build_report_records_provenance_and_experiment_name() -> None:
    report = build_report(
        _comparison(),
        experiment_name="perspective-v1",
        baseline_sha256="baseline-sha",
        candidate_sha256="candidate-sha",
    )

    assert report["experiment_name"] == "perspective-v1"
    assert report["baseline_sha256"] == "baseline-sha"
    assert report["candidate_sha256"] == "candidate-sha"


def test_write_report_refuses_overwrite_and_uses_safe_line_endings(
    tmp_path: Path,
) -> None:
    report = build_report(
        _comparison(),
        experiment_name="perspective-v1",
        baseline_sha256="baseline-sha",
        candidate_sha256="candidate-sha",
    )
    json_path, csv_path = write_report(report, tmp_path)

    assert json.loads(json_path.read_text())["promotion"]["passed"] is True
    assert b"\r\n" not in csv_path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_report(report, tmp_path)
