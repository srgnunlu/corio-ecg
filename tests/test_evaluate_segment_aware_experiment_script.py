"""Tests for segment-aware experiment artifact handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_segment_aware_experiment import build_report, write_report


def _comparison() -> dict[str, object]:
    return {
        "research_candidate": {"passed": True, "failed_checks": []},
        "production_status": "experimental-only",
        "categories": {
            "photos_bents": {
                "role": "target",
                "source_counts": {"total": 10, "successful": 10, "failed": 0},
                "baseline": {"mean_cosine_similarity": 0.70},
                "candidate": {"mean_cosine_similarity": 0.72},
                "deltas": {"mean_cosine_similarity": 0.02},
            }
        },
    }


def test_build_report_records_source_provenance() -> None:
    report = build_report(_comparison(), source_sha256="source-sha")

    assert report["source_sha256"] == "source-sha"
    assert report["production_status"] == "experimental-only"


def test_write_report_refuses_overwrite_and_uses_safe_line_endings(
    tmp_path: Path,
) -> None:
    report = build_report(_comparison(), source_sha256="source-sha")
    json_path, csv_path = write_report(report, tmp_path)

    assert json.loads(json_path.read_text())["research_candidate"]["passed"] is True
    assert b"\r\n" not in csv_path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_report(report, tmp_path)
