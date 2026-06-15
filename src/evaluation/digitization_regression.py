"""Compare candidate digitization artifacts against a frozen baseline."""

from __future__ import annotations

from typing import Any

from src.evaluation.digitization_regression_config import (
    DigitizationRegressionConfig,
    load_digitization_regression_config,
)
from src.evaluation.digitization_regression_metrics import (
    aggregate_records,
    calculate_deltas,
    compare_record,
)
from src.training.quality_gate_config import QualityGateConfig, load_quality_gate_config

Record = dict[str, Any]
RecordKey = tuple[str, str, str]


def _record_key(record: Record) -> RecordKey:
    return (
        str(record.get("ecg_id")),
        str(record.get("category")),
        str(record.get("image_id")),
    )


def _index_records(records: list[Record], label: str) -> dict[RecordKey, Record]:
    indexed = {_record_key(record): record for record in records}
    if len(indexed) != len(records):
        raise ValueError(f"{label} contains duplicate record identities")
    return indexed


def _category_result(
    category: str,
    baseline: list[Record],
    candidate: list[Record],
    config: DigitizationRegressionConfig,
    quality_config: QualityGateConfig,
) -> dict[str, Any]:
    baseline_aggregate = aggregate_records(baseline, quality_config)
    candidate_aggregate = aggregate_records(candidate, quality_config)
    role = (
        "supported"
        if category in config.supported_categories
        else "target"
        if category in config.target_categories
        else "unclassified"
    )
    return {
        "role": role,
        "baseline": baseline_aggregate,
        "candidate": candidate_aggregate,
        "deltas": calculate_deltas(baseline_aggregate, candidate_aggregate),
    }


def _promotion_checks(
    categories: dict[str, dict[str, Any]],
    config: DigitizationRegressionConfig,
) -> list[dict[str, Any]]:
    tolerance = config.tolerances
    rules = {
        "median_correlation": (-tolerance.median_correlation_drop, "minimum"),
        "median_rmse_mv": (tolerance.median_rmse_mv_increase, "maximum"),
        "median_snr_db": (-tolerance.median_snr_db_drop, "minimum"),
        "median_gain_error": (tolerance.median_gain_error_increase, "maximum"),
        "failure_rate": (tolerance.failure_rate_increase, "maximum"),
        "median_runtime_increase_fraction": (
            tolerance.median_runtime_increase_fraction,
            "maximum",
        ),
        "quality_gate_reject_rate": (
            tolerance.quality_gate_reject_rate_increase,
            "maximum",
        ),
    }
    failed: list[dict[str, Any]] = []
    for category in config.supported_categories:
        result = categories.get(category)
        if result is None:
            failed.append({"category": category, "metric": "category", "reason": "missing"})
            continue
        for metric, (limit, direction) in rules.items():
            observed = result["deltas"].get(metric)
            epsilon = 1e-12
            if observed is None or (
                direction == "minimum" and observed < limit - epsilon
            ) or (direction == "maximum" and observed > limit + epsilon):
                failed.append(
                    {
                        "category": category,
                        "metric": metric,
                        "observed_change": observed,
                        "allowed_change": limit,
                        "rule": direction,
                    }
                )
    return failed


def compare_digitization_reports(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    config: DigitizationRegressionConfig | None = None,
    quality_config: QualityGateConfig | None = None,
) -> dict[str, Any]:
    """Compare paired artifacts and return category promotion evidence."""
    resolved_config = config or load_digitization_regression_config()
    resolved_quality_config = quality_config or load_quality_gate_config()
    baseline = _index_records(baseline_report["records"], "baseline")
    candidate = _index_records(candidate_report["records"], "candidate")
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate record sets do not match")
    for key in baseline:
        if not baseline[key].get("source_sha256") or (
            baseline[key].get("source_sha256") != candidate[key].get("source_sha256")
        ):
            raise ValueError(f"source SHA-256 does not match for record {key}")

    categories = sorted({key[1] for key in baseline})
    category_results = {
        category: _category_result(
            category,
            [baseline[key] for key in sorted(baseline) if key[1] == category],
            [candidate[key] for key in sorted(candidate) if key[1] == category],
            resolved_config,
            resolved_quality_config,
        )
        for category in categories
    }
    failed_checks = _promotion_checks(category_results, resolved_config)
    return {
        "config": resolved_config.to_dict(),
        "quality_gate_version": resolved_quality_config.version,
        "promotion": {"passed": not failed_checks, "failed_checks": failed_checks},
        "overall": _category_result(
            "overall",
            list(baseline.values()),
            list(candidate.values()),
            resolved_config,
            resolved_quality_config,
        ),
        "categories": category_results,
        "records": [
            compare_record(baseline[key], candidate[key], resolved_quality_config)
            for key in sorted(baseline)
        ],
    }


__all__ = [
    "compare_digitization_reports",
    "load_digitization_regression_config",
]
