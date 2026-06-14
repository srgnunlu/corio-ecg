"""Interpretable quality gating for ECG digitization outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.training.quality_gate_config import QualityGateConfig, load_quality_gate_config

DEFAULT_QUALITY_GATE_CONFIG = load_quality_gate_config()


class QualityGateOutcome(StrEnum):
    """Possible actions after digitization quality assessment."""

    ACCEPT = "accept"
    WARN = "warn"
    REJECT = "reject"


@dataclass(frozen=True)
class QualityGateDecision:
    """Quality-gate outcome and human-readable reasons."""

    outcome: QualityGateOutcome
    reasons: tuple[str, ...]


def classify_fidelity_target(
    record: dict[str, Any],
    config: QualityGateConfig | None = None,
) -> QualityGateOutcome:
    """Classify matched-reference fidelity for benchmark supervision.

    These targets require a matched digital reference and are not available
    during ordinary photo inference.
    """
    resolved_config = config or DEFAULT_QUALITY_GATE_CONFIG
    thresholds = resolved_config.fidelity
    if record.get("status") != "success" or not record.get("fidelity"):
        return QualityGateOutcome.REJECT

    fidelity = record["fidelity"]
    absolute = fidelity.get("absolute")
    if not absolute:
        return QualityGateOutcome.REJECT

    correlation = float(fidelity["median_correlation"])
    rmse_mv = float(absolute["median_rmse_mv"])
    snr_db = float(absolute["median_snr_db"])
    if (
        correlation < thresholds.reject_correlation_below
        or rmse_mv > thresholds.reject_rmse_mv_above
        or snr_db < thresholds.reject_snr_db_below
    ):
        return QualityGateOutcome.REJECT
    if (
        correlation < thresholds.warn_correlation_below
        or rmse_mv > thresholds.warn_rmse_mv_above
        or snr_db < thresholds.warn_snr_db_below
    ):
        return QualityGateOutcome.WARN
    return QualityGateOutcome.ACCEPT


def classify_quality(
    record: dict[str, Any],
    config: QualityGateConfig | None = None,
) -> QualityGateDecision:
    """Classify digitization using operational features available at inference."""
    resolved_config = config or DEFAULT_QUALITY_GATE_CONFIG
    thresholds = resolved_config.inference
    if record.get("status") != "success":
        return QualityGateDecision(
            QualityGateOutcome.REJECT,
            ("digitization did not complete successfully",),
        )

    diagnostics = record.get("diagnostics") or {}
    required = (
        "layout_cost",
        "detected_leads_count",
        "nonzero_leads_count",
        "einthoven_score",
        "avg_pixel_per_mm",
    )
    if any(key not in diagnostics for key in required):
        return QualityGateDecision(
            QualityGateOutcome.REJECT,
            ("required quality diagnostics are missing",),
        )

    layout_cost = float(diagnostics["layout_cost"])
    detected_leads = int(diagnostics["detected_leads_count"])
    active_leads = int(diagnostics["nonzero_leads_count"])
    einthoven_score = float(diagnostics["einthoven_score"])
    pixel_per_mm = float(diagnostics["avg_pixel_per_mm"])

    critical_reasons: list[str] = []
    if active_leads < thresholds.reject_active_leads_below:
        critical_reasons.append(f"only {active_leads}/12 active leads")
    if detected_leads < thresholds.reject_detected_leads_below:
        critical_reasons.append(f"only {detected_leads} lead labels detected")
    if critical_reasons:
        return QualityGateDecision(QualityGateOutcome.REJECT, tuple(critical_reasons))

    severe_reasons: list[str] = []
    if einthoven_score < thresholds.severe_einthoven_below:
        severe_reasons.append(f"low Einthoven consistency ({einthoven_score:.2f})")
    if layout_cost > thresholds.layout_cost_above:
        severe_reasons.append(f"uncertain layout match ({layout_cost:.2f})")
    if detected_leads < thresholds.severe_detected_leads_below:
        severe_reasons.append(f"only {detected_leads} lead labels detected")
    if pixel_per_mm < thresholds.pixel_per_mm_below:
        severe_reasons.append(f"low grid calibration density ({pixel_per_mm:.2f} px/mm)")
    if active_leads < thresholds.expected_active_leads:
        severe_reasons.append(f"only {active_leads}/12 active leads")
    if len(severe_reasons) >= thresholds.severe_flags_to_reject:
        return QualityGateDecision(QualityGateOutcome.REJECT, tuple(severe_reasons))

    warn_reasons: list[str] = []
    if detected_leads < thresholds.warn_detected_leads_below:
        warn_reasons.append(f"only {detected_leads} lead labels detected")
    if einthoven_score < thresholds.warn_einthoven_below:
        warn_reasons.append(f"reduced Einthoven consistency ({einthoven_score:.2f})")
    if layout_cost > thresholds.layout_cost_above:
        warn_reasons.append(f"uncertain layout match ({layout_cost:.2f})")
    if pixel_per_mm < thresholds.pixel_per_mm_below:
        warn_reasons.append(f"low grid calibration density ({pixel_per_mm:.2f} px/mm)")
    if active_leads < thresholds.expected_active_leads:
        warn_reasons.append(f"only {active_leads}/12 active leads")
    if warn_reasons:
        return QualityGateDecision(QualityGateOutcome.WARN, tuple(warn_reasons))
    return QualityGateDecision(QualityGateOutcome.ACCEPT, ())


def _evaluate_subset(
    records: list[dict[str, Any]],
    config: QualityGateConfig,
) -> dict[str, Any]:
    rows = [
        (
            classify_fidelity_target(record, config),
            classify_quality(record, config).outcome,
        )
        for record in records
    ]
    target_counts = Counter(target.value for target, _ in rows)
    prediction_counts = Counter(prediction.value for _, prediction in rows)
    false_accepts = sum(
        target is QualityGateOutcome.REJECT and prediction is QualityGateOutcome.ACCEPT
        for target, prediction in rows
    )
    missed_rejects = sum(
        target is QualityGateOutcome.REJECT and prediction is not QualityGateOutcome.REJECT
        for target, prediction in rows
    )
    false_rejects = sum(
        target is not QualityGateOutcome.REJECT and prediction is QualityGateOutcome.REJECT
        for target, prediction in rows
    )
    rejected_targets = target_counts[QualityGateOutcome.REJECT.value]
    non_rejected_targets = len(rows) - rejected_targets
    return {
        "total": len(rows),
        "target_counts": dict(target_counts),
        "prediction_counts": dict(prediction_counts),
        "false_accepts": false_accepts,
        "false_accept_rate": false_accepts / rejected_targets if rejected_targets else 0.0,
        "missed_rejects": missed_rejects,
        "reject_recall": (
            (rejected_targets - missed_rejects) / rejected_targets
            if rejected_targets
            else 0.0
        ),
        "false_rejects": false_rejects,
        "false_reject_rate": (
            false_rejects / non_rejected_targets if non_rejected_targets else 0.0
        ),
    }


def evaluate_quality_gate(
    records: list[dict[str, Any]],
    config: QualityGateConfig | None = None,
) -> dict[str, Any]:
    """Evaluate the baseline gate overall and by physical capture category."""
    resolved_config = config or DEFAULT_QUALITY_GATE_CONFIG
    categories = sorted({str(record.get("category", "unknown")) for record in records})
    return {
        **_evaluate_subset(records, resolved_config),
        "categories": {
            category: _evaluate_subset(
                [record for record in records if record.get("category") == category],
                resolved_config,
            )
            for category in categories
        },
    }
