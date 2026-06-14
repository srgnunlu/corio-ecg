"""Tests for leakage-safe quality-gate split evaluation requests."""

from __future__ import annotations

import pytest

from src.evaluation.grouped_split import (
    create_grouped_split_manifest,
    load_grouped_split_config,
)
from src.evaluation.split_selection import (
    EvaluationPurpose,
    EvaluationStage,
    holdout_sample_size_warning,
    select_records_for_split,
    validate_evaluation_request,
)


def _records() -> list[dict[str, object]]:
    return [
        {"ecg_id": f"ecg-{group}", "image_id": group, "category": category}
        for group in range(10)
        for category in ("scan", "phone")
    ]


def test_select_records_for_split_keeps_only_assigned_ecgs() -> None:
    records = _records()
    manifest = create_grouped_split_manifest(records, load_grouped_split_config())

    selected = select_records_for_split(records, manifest, "test")

    expected_ecg_ids = set(manifest["splits"]["test"]["ecg_ids"])
    assert {record["ecg_id"] for record in selected} == expected_ecg_ids
    assert len(selected) == 4


def test_select_records_for_split_rejects_unknown_source_ecg() -> None:
    records = _records()
    manifest = create_grouped_split_manifest(records, load_grouped_split_config())
    records.append({"ecg_id": "unknown", "image_id": 99, "category": "scan"})

    with pytest.raises(ValueError, match="not assigned"):
        select_records_for_split(records, manifest, "test")


@pytest.mark.parametrize(
    (
        "purpose",
        "stage",
        "selected_split",
        "has_manifest",
        "split_evidence_status",
    ),
    [
        (
            EvaluationPurpose.THRESHOLD_TUNING,
            EvaluationStage.DEVELOPMENT,
            "test",
            True,
            "retroactive-development-only",
        ),
        (
            EvaluationPurpose.LOCKED_EVALUATION,
            EvaluationStage.HOLDOUT,
            "tune",
            True,
            "pre-registered",
        ),
        (
            EvaluationPurpose.LOCKED_EVALUATION,
            EvaluationStage.HOLDOUT,
            None,
            False,
            None,
        ),
    ],
)
def test_validate_evaluation_request_rejects_unsafe_combinations(
    purpose: EvaluationPurpose,
    stage: EvaluationStage,
    selected_split: str | None,
    has_manifest: bool,
    split_evidence_status: str | None,
) -> None:
    with pytest.raises(ValueError, match="evaluation request"):
        validate_evaluation_request(
            purpose,
            stage,
            selected_split,
            has_manifest,
            split_evidence_status,
        )


def test_validate_evaluation_request_allows_external_locked_evaluation() -> None:
    validate_evaluation_request(
        EvaluationPurpose.LOCKED_EVALUATION,
        EvaluationStage.EXTERNAL,
        selected_split=None,
        has_manifest=False,
        split_evidence_status=None,
    )


def test_holdout_sample_size_warning_flags_small_group_count() -> None:
    warning = holdout_sample_size_warning(
        [{"ecg_id": "ecg-1"}, {"ecg_id": "ecg-2"}],
        EvaluationStage.HOLDOUT,
    )

    assert warning is not None
    assert "2 ECG groups" in warning
