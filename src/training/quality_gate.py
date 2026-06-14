"""Interpretable quality gating for ECG digitization outputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.evaluation.quality_gate_metrics import evaluate_quality_gate_rows
from src.quality.features import QualityFeatureError, extract_quality_features
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

    try:
        features = extract_quality_features(record.get("diagnostics") or {})
    except QualityFeatureError as exc:
        return QualityGateDecision(
            QualityGateOutcome.REJECT,
            (f"invalid quality diagnostics: {exc}",),
        )

    layout_cost = features.layout_cost
    detected_leads = features.detected_leads_count
    active_leads = features.nonzero_leads_count
    einthoven_score = features.einthoven_score
    pixel_per_mm = features.avg_pixel_per_mm

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
        {
            "target": classify_fidelity_target(record, config).value,
            "prediction": classify_quality(record, config).outcome.value,
            "category": record.get("category"),
            "ecg_id": record.get("ecg_id"),
            "image_id": record.get("image_id"),
        }
        for record in records
    ]
    return evaluate_quality_gate_rows(rows)


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
