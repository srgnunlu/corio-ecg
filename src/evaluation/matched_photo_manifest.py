"""Validate privacy-safe Level B matched real-photo dataset manifests."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCHED_PHOTO_MANIFEST_CONFIG = (
    PROJECT_ROOT / "configs" / "matched_photo_manifest_v1.yaml"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_RECORD_FIELDS = (
    "case_id",
    "photo_id",
    "variant",
    "layout",
    "device_model",
    "printer_model",
    "photo_path",
    "photo_sha256",
    "reference_type",
    "reference_path",
    "reference_sha256",
    "split",
    "phi_reviewed",
)
FORBIDDEN_RECORD_FIELDS = (
    "patient_name",
    "patient_id",
    "date_of_birth",
    "institution_id",
    "barcode",
)

@dataclass(frozen=True)
class MatchedPhotoManifestConfig:
    """Locked Level B manifest requirements."""

    version: str
    evidence_status: str
    minimum_ecg_records: int
    minimum_capture_sources: int
    anonymous_case_id_pattern: str
    allowed_variants: tuple[str, ...]
    allowed_reference_types: tuple[str, ...]
    allowed_splits: tuple[str, ...]
    required_splits: tuple[str, ...]

    def validate(self) -> None:
        """Validate the manifest contract."""
        if not self.version.strip() or not self.evidence_status.strip():
            raise ValueError("manifest version and evidence status must not be empty")
        if self.minimum_ecg_records < 1 or self.minimum_capture_sources < 1:
            raise ValueError("manifest minimums must be positive")
        if not set(self.required_splits).issubset(self.allowed_splits):
            raise ValueError("required splits must be allowed")
        re.compile(self.anonymous_case_id_pattern)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible resolved configuration."""
        return asdict(self)


def load_matched_photo_manifest_config(
    path: Path = DEFAULT_MATCHED_PHOTO_MANIFEST_CONFIG,
) -> MatchedPhotoManifestConfig:
    """Load and validate the Level B manifest contract."""
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("matched-photo manifest config must contain a mapping")
    try:
        for key in (
            "allowed_variants",
            "allowed_reference_types",
            "allowed_splits",
            "required_splits",
        ):
            payload[key] = tuple(payload[key])
        config = MatchedPhotoManifestConfig(**payload)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid matched-photo manifest config structure: {exc}") from exc
    config.validate()
    return config


def _is_safe_relative_path(value: object) -> bool:
    path = PurePosixPath(str(value))
    return bool(str(value)) and not path.is_absolute() and ".." not in path.parts


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_file(
    record: dict[str, Any],
    *,
    kind: str,
    dataset_root: Path | None,
    verify_all_files: bool,
    errors: list[str],
) -> None:
    path_value = record.get(f"{kind}_path")
    sha_value = str(record.get(f"{kind}_sha256", ""))
    if not _is_safe_relative_path(path_value):
        errors.append(f"{kind}_path must be a safe relative path")
    if not SHA256_PATTERN.fullmatch(sha_value):
        errors.append(f"{kind}_sha256 must be a lowercase SHA-256")
    if dataset_root is None or not _is_safe_relative_path(path_value):
        return
    root = dataset_root.resolve()
    path = (root / str(path_value)).resolve()
    if path != root and root not in path.parents:
        errors.append(f"{kind}_path resolves outside dataset root")
        return
    if not path.exists():
        if verify_all_files:
            errors.append(f"{kind} file does not exist: {path_value}")
        return
    if SHA256_PATTERN.fullmatch(sha_value) and _file_sha256(path) != sha_value:
        errors.append(f"{kind} SHA-256 mismatch: {path_value}")


def _validate_record(
    record: dict[str, Any],
    index: int,
    config: MatchedPhotoManifestConfig,
    dataset_root: Path | None,
    verify_all_files: bool,
) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        return [f"record {index} missing fields: {', '.join(missing)}"]
    forbidden = sorted(set(record) & set(FORBIDDEN_RECORD_FIELDS))
    if forbidden:
        errors.append(f"forbidden PHI fields present: {', '.join(forbidden)}")
    if not re.fullmatch(config.anonymous_case_id_pattern, str(record["case_id"])):
        errors.append("case_id is not anonymous")
    if record["phi_reviewed"] is not True:
        errors.append("PHI review must be explicitly true")
    if record["variant"] not in config.allowed_variants:
        errors.append(f"unsupported variant: {record['variant']}")
    if record["reference_type"] not in config.allowed_reference_types:
        errors.append(f"unsupported reference_type: {record['reference_type']}")
    if record["split"] not in config.allowed_splits:
        errors.append(f"unsupported split: {record['split']}")
    if not str(record["layout"]).strip():
        errors.append("layout must not be empty")
    if not str(record["device_model"]).strip() or not str(record["printer_model"]).strip():
        errors.append("device_model and printer_model must not be empty")
    _validate_file(
        record,
        kind="photo",
        dataset_root=dataset_root,
        verify_all_files=verify_all_files,
        errors=errors,
    )
    _validate_file(
        record,
        kind="reference",
        dataset_root=dataset_root,
        verify_all_files=verify_all_files,
        errors=errors,
    )
    return [f"record {index}: {error}" for error in errors]


def validate_matched_photo_manifest(
    manifest: dict[str, Any],
    config: MatchedPhotoManifestConfig,
    *,
    dataset_root: Path | None = None,
    verify_all_files: bool = True,
) -> dict[str, Any]:
    """Validate a Level B manifest and return a path-free aggregate summary."""
    config.validate()
    errors: list[str] = []
    if manifest.get("manifest_version") != config.version:
        errors.append("manifest_version does not match locked config")
    if manifest.get("evidence_status") != config.evidence_status:
        errors.append("evidence_status does not match locked config")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("matched-photo manifest records must be a non-empty list")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record {index} must be a mapping")
            continue
        errors.extend(_validate_record(record, index, config, dataset_root, verify_all_files))

    valid_records = [record for record in records if isinstance(record, dict)]
    case_splits: dict[str, set[str]] = defaultdict(set)
    case_references: dict[str, set[str]] = defaultdict(set)
    for record in valid_records:
        case_id = str(record.get("case_id"))
        case_splits[case_id].add(str(record.get("split")))
        case_references[case_id].add(str(record.get("reference_sha256")))
    leaked = sorted(case_id for case_id, splits in case_splits.items() if len(splits) > 1)
    if leaked:
        errors.append(f"split leakage detected for cases: {', '.join(leaked)}")
    inconsistent = sorted(
        case_id for case_id, references in case_references.items() if len(references) > 1
    )
    if inconsistent:
        errors.append(f"inconsistent matched references for cases: {', '.join(inconsistent)}")
    case_ids = set(case_splits)
    if len(case_ids) < config.minimum_ecg_records:
        errors.append(f"at least {config.minimum_ecg_records} distinct ECG records are required")
    photo_ids = [str(record.get("photo_id")) for record in valid_records]
    if len(photo_ids) != len(set(photo_ids)):
        errors.append("photo_id values must be unique")
    capture_sources = {
        (str(record.get("device_model")), str(record.get("printer_model")))
        for record in valid_records
    }
    if len(capture_sources) < config.minimum_capture_sources:
        errors.append(f"at least {config.minimum_capture_sources} capture sources are required")
    split_case_counts = Counter(
        next(iter(splits)) for splits in case_splits.values() if len(splits) == 1
    )
    missing_splits = sorted(set(config.required_splits) - set(split_case_counts))
    if missing_splits:
        errors.append(f"required splits are empty: {', '.join(missing_splits)}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "manifest_version": config.version,
        "evidence_status": config.evidence_status,
        "ecg_records": len(case_ids),
        "photos": len(valid_records),
        "references": len({str(record["reference_sha256"]) for record in valid_records}),
        "capture_sources": len(capture_sources),
        "split_case_counts": dict(sorted(split_case_counts.items())),
        "variant_counts": dict(sorted(Counter(str(r["variant"]) for r in valid_records).items())),
        "reference_type_counts": dict(
            sorted(Counter(str(r["reference_type"]) for r in valid_records).items())
        ),
    }
