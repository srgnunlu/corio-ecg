"""Aggregate paired digitization metrics for experiment regression reports."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from src.training.quality_gate import classify_quality
from src.training.quality_gate_config import QualityGateConfig

Record = dict[str, Any]
METRICS = (
    "median_correlation",
    "median_rmse_mv",
    "median_snr_db",
    "median_gain_error",
    "failure_rate",
    "median_runtime_seconds",
    "quality_gate_reject_rate",
)


def fidelity_value(record: Record, metric: str) -> float | None:
    """Return one comparable fidelity value from a successful record."""
    fidelity = record.get("fidelity")
    if record.get("status") != "success" or not fidelity:
        return None
    if metric == "median_correlation":
        return float(fidelity[metric])
    absolute = fidelity.get("absolute")
    if not absolute:
        return None
    value = float(absolute[metric])
    return abs(value - 1.0) if metric == "median_gain_ratio" else value


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def aggregate_records(records: list[Record], config: QualityGateConfig) -> dict[str, Any]:
    """Aggregate fidelity, runtime, failure, and quality-gate metrics."""
    total = len(records)
    outcomes = [classify_quality(record, config).outcome.value for record in records]
    outcome_counts = Counter(outcomes)
    successful = [record for record in records if record.get("status") == "success"]
    metric_sources = {
        "median_correlation": "median_correlation",
        "median_rmse_mv": "median_rmse_mv",
        "median_snr_db": "median_snr_db",
        "median_gain_error": "median_gain_ratio",
    }
    metrics = {
        output_name: _median(
            [
                value
                for record in successful
                if (value := fidelity_value(record, source_name)) is not None
            ]
        )
        for output_name, source_name in metric_sources.items()
    }
    elapsed = [
        float(record["elapsed_seconds"])
        for record in records
        if record.get("elapsed_seconds") is not None
    ]
    return {
        "total": total,
        "successful": len(successful),
        "failure_rate": (total - len(successful)) / total if total else 0.0,
        **metrics,
        "median_runtime_seconds": _median(elapsed),
        "quality_gate": {
            "outcome_counts": dict(outcome_counts),
            "reject_rate": outcome_counts["reject"] / total if total else 0.0,
        },
    }


def calculate_deltas(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | None]:
    """Calculate candidate-minus-baseline aggregate changes."""
    deltas: dict[str, float | None] = {}
    for metric in METRICS:
        if metric == "quality_gate_reject_rate":
            baseline_value = baseline["quality_gate"]["reject_rate"]
            candidate_value = candidate["quality_gate"]["reject_rate"]
        else:
            baseline_value = baseline.get(metric)
            candidate_value = candidate.get(metric)
        deltas[metric] = optional_delta(baseline_value, candidate_value)
    baseline_runtime = baseline.get("median_runtime_seconds")
    candidate_runtime = candidate.get("median_runtime_seconds")
    deltas["median_runtime_increase_fraction"] = (
        (candidate_runtime - baseline_runtime) / baseline_runtime
        if baseline_runtime and candidate_runtime is not None
        else None
    )
    return deltas


def compare_record(
    baseline: Record,
    candidate: Record,
    config: QualityGateConfig,
) -> Record:
    """Return record-level paired metric and gate changes."""
    return {
        "category": baseline.get("category"),
        "image_id": baseline.get("image_id"),
        "ecg_id": baseline.get("ecg_id"),
        "baseline_status": baseline.get("status"),
        "candidate_status": candidate.get("status"),
        "status_changed": baseline.get("status") != candidate.get("status"),
        "median_correlation_delta": _value_delta(baseline, candidate, "median_correlation"),
        "median_rmse_mv_delta": _value_delta(baseline, candidate, "median_rmse_mv"),
        "median_snr_db_delta": _value_delta(baseline, candidate, "median_snr_db"),
        "median_gain_error_delta": _value_delta(
            baseline, candidate, "median_gain_ratio"
        ),
        "elapsed_seconds_delta": optional_delta(
            baseline.get("elapsed_seconds"), candidate.get("elapsed_seconds")
        ),
        "quality_gate": {
            "baseline": classify_quality(baseline, config).outcome.value,
            "candidate": classify_quality(candidate, config).outcome.value,
        },
    }


def _value_delta(baseline: Record, candidate: Record, metric: str) -> float | None:
    return optional_delta(
        fidelity_value(baseline, metric),
        fidelity_value(candidate, metric),
    )


def optional_delta(baseline: Any, candidate: Any) -> float | None:
    """Return a numeric delta when both values are available."""
    if baseline is None or candidate is None:
        return None
    return float(candidate) - float(baseline)
