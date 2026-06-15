"""Tests for pre-registered PMcardio holdout selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.create_pmcardio_holdout import write_manifest
from src.evaluation.pmcardio_holdout import (
    PMCardioHoldoutConfig,
    build_pmcardio_holdout_manifest,
)


def _metadata(group_count: int = 33) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ECG ID": f"ecg-{group}",
                "Image ID": group,
                "Image page": 0,
                "ECG format": "3x4" if group % 2 else "6x2",
                "Image relative path": f"{category}/img_{group}_page_0.jpeg",
            }
            for group in range(group_count)
            for category in ("phone", "scan")
        ]
    )


def _config() -> PMCardioHoldoutConfig:
    return PMCardioHoldoutConfig(
        version="holdout-v1",
        evidence_status="pre-registered",
        seed=42,
        tune_ecg_count=2,
        test_ecg_count=30,
        categories=("phone", "scan"),
    )


def test_holdout_excludes_development_and_keeps_variants_grouped() -> None:
    development = {"manifest_id": "dev", "records": [{"ECG ID": "ecg-0"}]}

    manifest = build_pmcardio_holdout_manifest(
        _metadata(),
        development,
        _config(),
        metadata_sha256="abc",
    )

    assert manifest["summary"]["ecg_groups"] == 32
    assert manifest["summary"]["images"] == 64
    assert manifest["summary"]["split_ecg_counts"] == {"test": 30, "tune": 2}
    assert all(record["ecg_id"] != "ecg-0" for record in manifest["records"])
    assignments: dict[str, set[str]] = {}
    for record in manifest["records"]:
        assignments.setdefault(record["ecg_id"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in assignments.values())


def test_holdout_is_deterministic() -> None:
    development = {"records": [{"ECG ID": "ecg-0"}]}

    first = build_pmcardio_holdout_manifest(
        _metadata(),
        development,
        _config(),
        metadata_sha256="abc",
    )
    second = build_pmcardio_holdout_manifest(
        _metadata().sample(frac=1),
        development,
        _config(),
        metadata_sha256="abc",
    )

    assert first == second


def test_holdout_rejects_unexpected_available_group_count() -> None:
    with pytest.raises(ValueError, match="requires exactly 32"):
        build_pmcardio_holdout_manifest(
            _metadata(group_count=32),
            {"records": [{"ECG ID": "ecg-0"}]},
            _config(),
            metadata_sha256="abc",
        )


def test_write_manifest_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "holdout.json"
    output.write_text("{}")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_manifest({}, output)
