# Report builder for deterministic Brugada/Vereckei VT/SVT criteria audits.

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.evaluation.vtsvt_audit_io import load_signal_record
from src.evaluation.vtsvt_audit_schema import CSV_FIELDS, LoadedSignal
from src.vtsvt.criteria import assess_vtsvt
from src.vtsvt.features import TARGET_SAMPLE_RATE
from src.vtsvt.models import LeadMorphology, VTSVTAssessment

SignalLoader = Callable[[dict[str, Any], Path, int], LoadedSignal]
AssessmentFn = Callable[[np.ndarray, np.ndarray | None, int], VTSVTAssessment]


def evaluate_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    signal_loader: SignalLoader = load_signal_record,
    assess_fn: AssessmentFn = assess_vtsvt,
) -> dict[str, Any]:
    """Evaluate every manifest record and return a report."""
    manifest_dir = manifest_path.parent
    sample_rate = int(manifest.get("sample_rate") or TARGET_SAMPLE_RATE)
    records: list[dict[str, Any]] = []

    for record in manifest["records"]:
        try:
            loaded = signal_loader(record, manifest_dir, sample_rate)
            assessment = assess_fn(loaded.signal, loaded.rhythm_strip, loaded.sample_rate)
            records.append(_success_record(record, loaded, assessment))
        except Exception as error:
            records.append(_failure_record(record, error))

    return build_audit_report(manifest, manifest_path, records)


def build_audit_report(
    manifest: dict[str, Any],
    manifest_path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a JSON-serializable VTSVT audit report."""
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "audit_version": "vtsvt-criteria-audit-v1",
        "manifest_version": manifest.get("version"),
        "manifest_path": str(manifest_path),
        "sample_rate": manifest.get("sample_rate", TARGET_SAMPLE_RATE),
        "cohort_description": manifest.get("cohort_description"),
        "limitations": (
            "This deterministic audit measures Brugada/Vereckei criteria trigger "
            "behavior. It is not a standalone clinical diagnosis or locked "
            "sensitivity/specificity claim unless the manifest is adjudicated."
        ),
        "aggregate": aggregate_records(records),
        "records": records,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate criterion trigger counts and optional expected-outcome metrics."""
    successful = [record for record in records if record.get("status") == "success"]
    criterion_counts: Counter[str] = Counter()
    for record in successful:
        criterion_counts.update(record.get("evidence", ()))

    return {
        "total": len(records),
        "successful": len(successful),
        "failed": len(records) - len(successful),
        "in_scope": sum(bool(record.get("in_scope")) for record in successful),
        "supports_vt": sum(bool(record.get("supports_vt")) for record in successful),
        "indeterminate": sum(
            record.get("classification") == "indeterminate_wide_complex_tachycardia"
            for record in successful
        ),
        "terminal_morphology_measurable": sum(
            record.get("terminal_morphology_suggests_vt") is not None
            for record in successful
        ),
        "criterion_counts": dict(sorted(criterion_counts.items())),
        "clinical_labels": _label_counts(successful),
        "confusion": _confusion(successful),
    }


def _success_record(
    record: dict[str, Any],
    loaded: LoadedSignal,
    assessment: VTSVTAssessment,
) -> dict[str, Any]:
    features = assessment.features
    v1 = _lead(features.precordial_leads, "V1")
    v6 = _lead(features.precordial_leads, "V6")
    return {
        "record_id": record.get("record_id"),
        "clinical_label": record.get("clinical_label"),
        "expected_supports_vt": record.get("expected_supports_vt"),
        "source_path": loaded.source_path,
        "status": "success",
        "error": None,
        "classification": assessment.classification,
        "in_scope": assessment.in_scope,
        "supports_vt": assessment.supports_vt,
        "heart_rate_bpm": features.heart_rate_bpm,
        "qrs_ms": features.qrs_ms,
        "regular": features.regular,
        "n_beats": features.n_beats,
        "anchor_lead": features.anchor_lead,
        "max_precordial_rs_interval_ms": features.max_precordial_rs_interval_ms,
        "terminal_morphology_suggests_vt": (
            assessment.brugada.terminal_morphology_suggests_vt
        ),
        "evidence": list(assessment.evidence),
        "limitations": list(assessment.limitations),
        "brugada_criteria": list(assessment.brugada.positive_criteria),
        "vereckei_criteria": list(assessment.vereckei.positive_criteria),
        **_lead_fields("v1", v1),
        **_lead_fields("v6", v6),
    }


def _failure_record(record: dict[str, Any], error: Exception) -> dict[str, Any]:
    row: dict[str, Any] = {field: None for field in CSV_FIELDS}
    row.update(
        {
            "record_id": record.get("record_id"),
            "clinical_label": record.get("clinical_label"),
            "expected_supports_vt": record.get("expected_supports_vt"),
            "source_path": record.get("signal_path") or record.get("record_path"),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    )
    return row


def _lead_fields(prefix: str, lead: LeadMorphology | None) -> dict[str, Any]:
    return {
        f"{prefix}_qrs_pattern": lead.qrs_pattern if lead else None,
        f"{prefix}_r_s_ratio": lead.r_s_ratio if lead else None,
        f"{prefix}_qrs_onset_to_s_nadir_ms": (
            lead.qrs_onset_to_s_nadir_ms if lead else None
        ),
        f"{prefix}_initial_deflection_width_ms": (
            lead.initial_deflection_width_ms if lead else None
        ),
        f"{prefix}_s_downstroke_notched": lead.s_downstroke_notched if lead else None,
    }


def _confusion(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [
        record for record in records if isinstance(record.get("expected_supports_vt"), bool)
    ]
    tp = sum(record["expected_supports_vt"] and record["supports_vt"] for record in evaluated)
    fp = sum(
        (not record["expected_supports_vt"]) and record["supports_vt"]
        for record in evaluated
    )
    tn = sum(
        (not record["expected_supports_vt"]) and not record["supports_vt"]
        for record in evaluated
    )
    fn = sum(
        record["expected_supports_vt"] and not record["supports_vt"]
        for record in evaluated
    )
    return {
        "evaluated": len(evaluated),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
    }


def _label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(record.get("clinical_label", "unknown")) for record in records)
    return dict(sorted(counts.items()))


def _lead(leads: tuple[LeadMorphology, ...], name: str) -> LeadMorphology | None:
    return next((lead for lead in leads if lead.lead == name), None)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator
