"""Statistical metrics for quality-gate accept, warn, and reject decisions."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Callable

OUTCOMES = ("accept", "warn", "reject")
DEFAULT_BOOTSTRAP_SEED = 20260614
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
MIN_REPORTING_RECORDS = 30

RateFunction = Callable[[list[dict[str, Any]]], float | None]


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _false_accept_rate(rows: list[dict[str, Any]]) -> float | None:
    rejected = [row for row in rows if row["target"] == "reject"]
    return _safe_rate(
        sum(row["prediction"] == "accept" for row in rejected),
        len(rejected),
    )


def _reject_recall(rows: list[dict[str, Any]]) -> float | None:
    rejected = [row for row in rows if row["target"] == "reject"]
    return _safe_rate(
        sum(row["prediction"] == "reject" for row in rejected),
        len(rejected),
    )


def _false_reject_rate(rows: list[dict[str, Any]]) -> float | None:
    non_rejected = [row for row in rows if row["target"] != "reject"]
    return _safe_rate(
        sum(row["prediction"] == "reject" for row in non_rejected),
        len(non_rejected),
    )


def _non_reject_coverage(rows: list[dict[str, Any]]) -> float | None:
    return _safe_rate(
        sum(row["prediction"] != "reject" for row in rows),
        len(rows),
    )


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_interval(
    rows: list[dict[str, Any]],
    rate_function: RateFunction,
    *,
    seed: int,
    resamples: int,
) -> dict[str, float | int | str] | None:
    if not rows or resamples < 1:
        return None
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped_rows[str(row.get("ecg_id") or f"record-{index}")].append(row)
    group_ids = sorted(grouped_rows)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled_group_ids = [group_ids[rng.randrange(len(group_ids))] for _ in group_ids]
        sample = [row for group_id in sampled_group_ids for row in grouped_rows[group_id]]
        estimate = rate_function(sample)
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return None
    estimates.sort()
    return {
        "low": _percentile(estimates, 0.025),
        "high": _percentile(estimates, 0.975),
        "resamples": resamples,
        "resamples_used": len(estimates),
        "resampling_unit": "ecg_id",
    }


def _confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        target: {
            prediction: sum(
                row["target"] == target and row["prediction"] == prediction
                for row in rows
            )
            for prediction in OUTCOMES
        }
        for target in OUTCOMES
    }


def evaluate_quality_gate_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Evaluate quality-gate rows with deterministic bootstrap intervals."""
    target_counts = Counter(str(row["target"]) for row in rows)
    prediction_counts = Counter(str(row["prediction"]) for row in rows)
    false_accepts = sum(
        row["target"] == "reject" and row["prediction"] == "accept" for row in rows
    )
    missed_rejects = sum(
        row["target"] == "reject" and row["prediction"] != "reject" for row in rows
    )
    false_rejects = sum(
        row["target"] != "reject" and row["prediction"] == "reject" for row in rows
    )
    rate_functions: dict[str, RateFunction] = {
        "false_accept_rate": _false_accept_rate,
        "reject_recall": _reject_recall,
        "false_reject_rate": _false_reject_rate,
        "non_reject_coverage": _non_reject_coverage,
    }
    rates = {
        name: function(rows) if function(rows) is not None else 0.0
        for name, function in rate_functions.items()
    }
    missed_records = [
        {
            "ecg_id": row.get("ecg_id"),
            "image_id": row.get("image_id"),
            "category": row.get("category"),
            "prediction": row.get("prediction"),
        }
        for row in rows
        if row["target"] == "reject" and row["prediction"] != "reject"
    ]
    missed_records.sort(
        key=lambda record: (
            str(record["ecg_id"]),
            str(record["category"]),
            str(record["image_id"]),
        )
    )
    warning = None
    independent_groups = len(
        {str(row.get("ecg_id") or f"record-{index}") for index, row in enumerate(rows)}
    )
    if independent_groups < MIN_REPORTING_RECORDS:
        warning = (
            f"Only {independent_groups} ECG groups and {len(rows)} records are available; "
            "confidence intervals and rates are unstable and must not be treated as "
            "external performance evidence."
        )
    return {
        "total": len(rows),
        "independent_groups": independent_groups,
        "target_counts": dict(target_counts),
        "prediction_counts": dict(prediction_counts),
        "confusion_matrix": _confusion_matrix(rows),
        "false_accepts": false_accepts,
        "false_accept_rate": rates["false_accept_rate"],
        "missed_rejects": missed_rejects,
        "reject_recall": rates["reject_recall"],
        "false_rejects": false_rejects,
        "false_reject_rate": rates["false_reject_rate"],
        "non_reject_coverage": rates["non_reject_coverage"],
        "confidence_intervals": {
            name: _bootstrap_interval(
                rows,
                function,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            for name, function in rate_functions.items()
        },
        "missed_reject_records": missed_records,
        "statistical_warning": warning,
    }
