"""Select benchmark splits and reject unsafe evaluation requests."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from src.evaluation.grouped_split import SPLIT_NAMES, validate_grouped_split_manifest

MIN_MEANINGFUL_HOLDOUT_GROUPS = 30


class EvaluationStage(StrEnum):
    """Evidence stage for a benchmark run."""

    DEVELOPMENT = "development"
    HOLDOUT = "holdout"
    EXTERNAL = "external"


class EvaluationPurpose(StrEnum):
    """Purpose that determines which records a run may access."""

    DEVELOPMENT_BASELINE = "development-baseline"
    THRESHOLD_TUNING = "threshold-tuning"
    LOCKED_EVALUATION = "locked-evaluation"


def validate_evaluation_request(
    purpose: EvaluationPurpose,
    stage: EvaluationStage,
    selected_split: str | None,
    has_manifest: bool,
    split_evidence_status: str | None = None,
) -> None:
    """Reject requests that mix threshold tuning and locked evaluation records."""
    valid = False
    if purpose is EvaluationPurpose.DEVELOPMENT_BASELINE:
        valid = stage is EvaluationStage.DEVELOPMENT and not has_manifest and selected_split is None
    elif purpose is EvaluationPurpose.THRESHOLD_TUNING:
        valid = (
            stage is EvaluationStage.DEVELOPMENT
            and has_manifest
            and selected_split == "tune"
        )
    elif purpose is EvaluationPurpose.LOCKED_EVALUATION:
        valid = (
            stage is EvaluationStage.HOLDOUT
            and has_manifest
            and selected_split == "test"
            and split_evidence_status == "pre-registered"
        ) or (
            stage is EvaluationStage.EXTERNAL
            and not has_manifest
            and selected_split is None
        )
    if not valid:
        raise ValueError(
            "unsafe evaluation request: use development-baseline without a split, "
            "threshold-tuning only on tune, locked holdout evaluation only on test, "
            "locked holdout requires a pre-registered split, or locked external "
            "evaluation must not use an internal split manifest"
        )


def select_records_for_split(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    selected_split: str,
) -> list[dict[str, Any]]:
    """Select records assigned to one split and reject unassigned source ECGs."""
    if selected_split not in SPLIT_NAMES:
        raise ValueError(f"unknown selected split {selected_split!r}")
    validate_grouped_split_manifest(manifest)
    assignments = {
        str(record["ecg_id"]): str(record["split"]) for record in manifest["records"]
    }
    source_group_ids = {str(record.get("ecg_id")) for record in records}
    unassigned = sorted(source_group_ids - assignments.keys())
    if unassigned:
        raise ValueError(f"source ECG identities are not assigned: {', '.join(unassigned)}")
    return [
        record
        for record in records
        if assignments[str(record.get("ecg_id"))] == selected_split
    ]


def holdout_sample_size_warning(
    records: list[dict[str, Any]],
    stage: EvaluationStage,
) -> str | None:
    """Warn when an internal holdout is too small for meaningful evidence."""
    if stage is not EvaluationStage.HOLDOUT:
        return None
    group_count = len({str(record.get("ecg_id")) for record in records})
    if group_count >= MIN_MEANINGFUL_HOLDOUT_GROUPS:
        return None
    return (
        f"Holdout contains only {group_count} ECG groups; this is a pipeline-safety "
        "check, not meaningful external performance evidence."
    )
