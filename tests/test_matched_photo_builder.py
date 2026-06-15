"""Tests for strict Level B collection inventory manifest generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_matched_photo_manifest import write_manifest
from src.evaluation.matched_photo_builder import (
    COLLECTION_FIELDS,
    build_matched_photo_manifest,
    load_collection_inventory,
)
from src.evaluation.matched_photo_manifest import MatchedPhotoManifestConfig


def _config() -> MatchedPhotoManifestConfig:
    return MatchedPhotoManifestConfig(
        version="matched-photo-manifest-v1",
        evidence_status="pre-registered-external",
        minimum_ecg_records=2,
        minimum_capture_sources=2,
        anonymous_case_id_pattern=r"^case[0-9]{3,}$",
        allowed_variants=("front",),
        allowed_reference_types=("flatbed_scan",),
        allowed_splits=("tune", "test"),
        required_splits=("tune", "test"),
    )


def _row(case_id: str, split: str, device_model: str) -> dict[str, str]:
    return {
        "case_id": case_id,
        "photo_id": f"{case_id}__front",
        "variant": "front",
        "layout": "3x4+1R",
        "device_model": device_model,
        "printer_model": "printer-a",
        "photo_path": f"photos/{case_id}__front.jpg",
        "reference_type": "flatbed_scan",
        "reference_path": f"references/{case_id}.png",
        "split": split,
        "phi_reviewed": "true",
    }


def _write_files(root: Path, case_id: str) -> None:
    photo = root / "photos" / f"{case_id}__front.jpg"
    reference = root / "references" / f"{case_id}.png"
    photo.parent.mkdir(parents=True, exist_ok=True)
    reference.parent.mkdir(parents=True, exist_ok=True)
    photo.write_bytes(f"photo-{case_id}".encode())
    reference.write_bytes(f"reference-{case_id}".encode())


def test_build_manifest_hashes_files_and_preserves_registered_splits(tmp_path: Path) -> None:
    _write_files(tmp_path, "case001")
    _write_files(tmp_path, "case002")

    manifest = build_matched_photo_manifest(
        [_row("case001", "tune", "phone-a"), _row("case002", "test", "phone-b")],
        tmp_path,
        _config(),
    )

    assert manifest["records"][0]["split"] == "tune"
    assert len(manifest["records"][0]["photo_sha256"]) == 64
    assert manifest["records"][1]["split"] == "test"


def test_load_inventory_rejects_unexpected_columns(tmp_path: Path) -> None:
    inventory_path = tmp_path / "collection.csv"
    with inventory_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[*COLLECTION_FIELDS, "patient_id"])
        writer.writeheader()
        writer.writerow({**_row("case001", "tune", "phone-a"), "patient_id": "secret"})

    with pytest.raises(ValueError, match="unexpected columns: patient_id"):
        load_collection_inventory(inventory_path)


def test_build_manifest_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset file does not exist"):
        build_matched_photo_manifest(
            [_row("case001", "tune", "phone-a")],
            tmp_path,
            _config(),
        )


def test_write_manifest_requires_explicit_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "manifest.json"
    output_path.write_text("{}")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_manifest({"records": []}, output_path)

    write_manifest({"records": []}, output_path, allow_overwrite=True)
    assert json.loads(output_path.read_text()) == {"records": []}
