"""Validate a local Level B matched real-photo dataset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.matched_photo_manifest import (  # noqa: E402
    DEFAULT_MATCHED_PHOTO_MANIFEST_CONFIG,
    load_matched_photo_manifest_config,
    validate_matched_photo_manifest,
)

DEFAULT_DATASET_ROOT = Path("data/level-b-matched")
DEFAULT_OUTPUT = Path("results/level-b-matched/manifest_validation.json")


def build_validation_report(
    manifest_path: Path,
    dataset_root: Path,
    config_path: Path,
    *,
    verify_all_files: bool = True,
) -> dict[str, Any]:
    """Validate a local manifest and return a path-free aggregate report."""
    manifest_bytes = manifest_path.read_bytes()
    config = load_matched_photo_manifest_config(config_path)
    summary = validate_matched_photo_manifest(
        json.loads(manifest_bytes),
        config,
        dataset_root=dataset_root,
        verify_all_files=verify_all_files,
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "config": config.to_dict(),
        "validation": {"passed": True, "summary": summary},
        "limitations": (
            "This validates dataset structure, privacy review declarations, hashes, "
            "and grouped splits. It does not establish digitization or clinical accuracy."
        ),
    }


def write_validation_report(
    report: dict[str, Any],
    output_path: Path,
    *,
    allow_overwrite: bool = False,
) -> None:
    """Write an aggregate report without copying source record paths."""
    if output_path.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite validation report: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_MATCHED_PHOTO_MANIFEST_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest or args.dataset_root / "manifest.json"
    report = build_validation_report(manifest_path, args.dataset_root, args.config)
    write_validation_report(report, args.output, allow_overwrite=args.allow_overwrite)
    print(f"Wrote {args.output}")
    print(f"Validation passed: {report['validation']['passed']}")
    print(f"Summary: {report['validation']['summary']}")


if __name__ == "__main__":
    main()
