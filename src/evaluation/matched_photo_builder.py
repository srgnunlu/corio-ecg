"""Build validated Level B manifests from a strict collection inventory."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from src.evaluation.matched_photo_manifest import (
    MatchedPhotoManifestConfig,
    validate_matched_photo_manifest,
)

COLLECTION_FIELDS = (
    "case_id",
    "photo_id",
    "variant",
    "layout",
    "device_model",
    "printer_model",
    "photo_path",
    "reference_type",
    "reference_path",
    "split",
    "phi_reviewed",
)


def load_collection_inventory(path: Path) -> list[dict[str, str]]:
    """Load a strict CSV inventory without permitting undeclared columns."""
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = tuple(reader.fieldnames or ())
        missing = sorted(set(COLLECTION_FIELDS) - set(fieldnames))
        unexpected = sorted(set(fieldnames) - set(COLLECTION_FIELDS))
        if missing or unexpected:
            problems = []
            if missing:
                problems.append(f"missing columns: {', '.join(missing)}")
            if unexpected:
                problems.append(f"unexpected columns: {', '.join(unexpected)}")
            raise ValueError("; ".join(problems))
        return [dict(row) for row in reader]


def _dataset_file(dataset_root: Path, relative_path: str) -> Path:
    path = PurePosixPath(relative_path)
    if not relative_path or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe dataset path: {relative_path}")
    root = dataset_root.resolve()
    resolved = (root / relative_path).resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"dataset path resolves outside root: {relative_path}")
    if not resolved.is_file():
        raise ValueError(f"dataset file does not exist: {relative_path}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_phi_reviewed(value: object) -> bool:
    if value is True or (isinstance(value, str) and value.strip().lower() == "true"):
        return True
    raise ValueError("phi_reviewed must be explicitly true")


def build_matched_photo_manifest(
    rows: list[dict[str, Any]],
    dataset_root: Path,
    config: MatchedPhotoManifestConfig,
) -> dict[str, Any]:
    """Build and fully validate a manifest from pre-registered inventory rows."""
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        missing = [
            field
            for field in COLLECTION_FIELDS
            if field not in row or not str(row[field]).strip()
        ]
        unexpected = sorted(set(row) - set(COLLECTION_FIELDS))
        if missing or unexpected:
            raise ValueError(
                f"inventory row {index} has invalid fields: "
                f"missing={missing}, unexpected={unexpected}"
            )
        photo_path = str(row["photo_path"])
        reference_path = str(row["reference_path"])
        records.append(
            {
                **{field: str(row[field]).strip() for field in COLLECTION_FIELDS[:-1]},
                "phi_reviewed": _parse_phi_reviewed(row["phi_reviewed"]),
                "photo_sha256": _sha256(_dataset_file(dataset_root, photo_path)),
                "reference_sha256": _sha256(
                    _dataset_file(dataset_root, reference_path)
                ),
            }
        )
    manifest = {
        "manifest_version": config.version,
        "evidence_status": config.evidence_status,
        "records": records,
    }
    validate_matched_photo_manifest(manifest, config, dataset_root=dataset_root)
    return manifest


__all__ = [
    "COLLECTION_FIELDS",
    "build_matched_photo_manifest",
    "load_collection_inventory",
]
