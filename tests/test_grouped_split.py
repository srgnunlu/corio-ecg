"""Tests for deterministic leakage-safe grouped benchmark splits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.create_quality_gate_split import build_split_artifact
from src.evaluation.grouped_split import (
    GroupedSplitConfig,
    create_grouped_split_manifest,
    load_grouped_split_config,
    validate_grouped_split_manifest,
)


def _records(group_count: int = 10) -> list[dict[str, object]]:
    return [
        {
            "ecg_id": f"ecg-{group_index}",
            "image_id": group_index,
            "category": category,
        }
        for group_index in range(group_count)
        for category in ("scan", "phone", "bent")
    ]


def test_default_grouped_split_config_is_valid() -> None:
    config = load_grouped_split_config()

    assert config.version == "quality-gate-split-v1"
    assert config.seed == 20260614
    assert config.evidence_status == "retroactive-development-only"
    assert config.fractions == {"train": 0.6, "tune": 0.2, "test": 0.2}


def test_grouped_split_config_rejects_invalid_fraction_sum() -> None:
    config = GroupedSplitConfig(
        version="invalid",
        seed=1,
        train_fraction=0.7,
        tune_fraction=0.2,
        test_fraction=0.2,
        evidence_status="retroactive-development-only",
    )

    with pytest.raises(ValueError, match="sum to 1.0"):
        config.validate()


def test_create_grouped_split_manifest_is_deterministic() -> None:
    records = _records()
    config = load_grouped_split_config()

    first = create_grouped_split_manifest(records, config)
    second = create_grouped_split_manifest(list(reversed(records)), config)

    assert first == second


def test_create_grouped_split_keeps_all_ecg_variants_together() -> None:
    manifest = create_grouped_split_manifest(_records(), load_grouped_split_config())
    assignments: dict[str, set[str]] = {}
    for record in manifest["records"]:
        assignments.setdefault(record["ecg_id"], set()).add(record["split"])

    assert all(len(splits) == 1 for splits in assignments.values())
    assert manifest["summary"]["group_counts"] == {"train": 6, "tune": 2, "test": 2}
    assert manifest["summary"]["record_counts"] == {"train": 18, "tune": 6, "test": 6}


def test_create_grouped_split_preserves_balanced_categories() -> None:
    manifest = create_grouped_split_manifest(_records(), load_grouped_split_config())

    assert manifest["summary"]["category_counts"]["train"] == {
        "bent": 6,
        "phone": 6,
        "scan": 6,
    }
    assert manifest["summary"]["category_counts"]["test"] == {
        "bent": 2,
        "phone": 2,
        "scan": 2,
    }


def test_validate_grouped_split_manifest_rejects_leakage() -> None:
    manifest = create_grouped_split_manifest(_records(), load_grouped_split_config())
    leaked_record = dict(manifest["records"][0])
    leaked_record["split"] = "test"
    manifest["records"].append(leaked_record)

    with pytest.raises(ValueError, match="leakage"):
        validate_grouped_split_manifest(manifest)


def test_validate_grouped_split_manifest_rejects_inconsistent_split_lists() -> None:
    manifest = create_grouped_split_manifest(_records(), load_grouped_split_config())
    moved_ecg_id = manifest["splits"]["train"]["ecg_ids"][0]
    manifest["splits"]["test"]["ecg_ids"].append(moved_ecg_id)

    with pytest.raises(ValueError, match="split ECG lists"):
        validate_grouped_split_manifest(manifest)


def test_build_split_artifact_records_source_hash_and_manifest_id(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps({"records": _records()}))

    artifact = build_split_artifact(source_path, load_grouped_split_config())

    assert len(artifact["source_sha256"]) == 64
    assert len(artifact["manifest_id"]) == 64
    assert artifact["summary"]["groups_total"] == 10
