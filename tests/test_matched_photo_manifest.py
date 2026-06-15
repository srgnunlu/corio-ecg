"""Tests for Level B matched real-photo manifest validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_matched_photo_dataset import build_validation_report, write_validation_report
from src.evaluation.matched_photo_manifest import (
    load_matched_photo_manifest_config,
    validate_matched_photo_manifest,
)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _manifest(case_count: int = 20) -> dict[str, object]:
    records = []
    for index in range(case_count):
        case_id = f"case{index + 1:03d}"
        records.append(
            {
                "case_id": case_id,
                "photo_id": f"{case_id}__front",
                "variant": "front",
                "layout": "3x4+1R",
                "device_model": "phone-a" if index % 2 == 0 else "phone-b",
                "printer_model": "printer-a",
                "photo_path": f"photos/{case_id}__front.jpg",
                "photo_sha256": _sha(f"photo-{index}".encode()),
                "reference_type": "digital_waveform",
                "reference_path": f"references/{case_id}.npy",
                "reference_sha256": _sha(f"reference-{index}".encode()),
                "split": "tune" if index < 5 else "test",
                "phi_reviewed": True,
            }
        )
    return {
        "manifest_version": "matched-photo-manifest-v1",
        "evidence_status": "pre-registered-external",
        "records": records,
    }


def test_validate_matched_photo_manifest_accepts_complete_external_dataset() -> None:
    summary = validate_matched_photo_manifest(
        _manifest(),
        load_matched_photo_manifest_config(),
    )

    assert summary["ecg_records"] == 20
    assert summary["photos"] == 20
    assert summary["split_case_counts"] == {"test": 15, "tune": 5}
    assert summary["capture_sources"] == 2


def test_validate_matched_photo_manifest_rejects_insufficient_cases() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        validate_matched_photo_manifest(
            _manifest(case_count=19),
            load_matched_photo_manifest_config(),
        )


def test_validate_matched_photo_manifest_rejects_phi_and_group_leakage() -> None:
    manifest = _manifest()
    records = manifest["records"]
    assert isinstance(records, list)
    records[0]["phi_reviewed"] = False
    records[0]["patient_name"] = "must-not-appear"
    leaked = dict(records[0])
    leaked["photo_id"] = "case001__angle"
    leaked["variant"] = "angle"
    leaked["split"] = "test"
    records.append(leaked)

    with pytest.raises(ValueError, match="forbidden PHI|PHI review|split leakage"):
        validate_matched_photo_manifest(manifest, load_matched_photo_manifest_config())


def test_validate_matched_photo_manifest_rejects_unsafe_paths() -> None:
    manifest = _manifest()
    records = manifest["records"]
    assert isinstance(records, list)
    records[0]["photo_path"] = "../patient-name.jpg"

    with pytest.raises(ValueError, match="safe relative path"):
        validate_matched_photo_manifest(manifest, load_matched_photo_manifest_config())


def test_validate_matched_photo_manifest_verifies_file_hashes(tmp_path: Path) -> None:
    manifest = _manifest()
    records = manifest["records"]
    assert isinstance(records, list)
    record = records[0]
    photo_path = tmp_path / str(record["photo_path"])
    reference_path = tmp_path / str(record["reference_path"])
    photo_path.parent.mkdir(parents=True)
    reference_path.parent.mkdir(parents=True)
    photo_path.write_bytes(b"photo")
    reference_path.write_bytes(b"reference")
    record["photo_sha256"] = _sha(b"wrong")
    record["reference_sha256"] = _sha(b"reference")

    with pytest.raises(ValueError, match="photo SHA-256 mismatch"):
        validate_matched_photo_manifest(
            manifest,
            load_matched_photo_manifest_config(),
            dataset_root=tmp_path,
            verify_all_files=False,
        )


def test_validate_matched_photo_manifest_rejects_symlink_escape(tmp_path: Path) -> None:
    manifest = _manifest()
    records = manifest["records"]
    assert isinstance(records, list)
    outside = tmp_path.parent / "outside-photo.jpg"
    outside.write_bytes(b"photo")
    photo_path = tmp_path / str(records[0]["photo_path"])
    photo_path.parent.mkdir(parents=True)
    photo_path.symlink_to(outside)

    with pytest.raises(ValueError, match="resolves outside dataset root"):
        validate_matched_photo_manifest(
            manifest,
            load_matched_photo_manifest_config(),
            dataset_root=tmp_path,
            verify_all_files=False,
        )


def test_validation_report_is_aggregate_and_refuses_overwrite(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    report = build_validation_report(
        manifest_path,
        tmp_path,
        Path("configs/matched_photo_manifest_v1.yaml"),
        verify_all_files=False,
    )
    output_path = tmp_path / "report.json"
    write_validation_report(report, output_path)

    assert "records" not in report["validation"]
    assert "photo_path" not in output_path.read_text()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_validation_report(report, output_path)
