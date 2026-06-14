"""Runtime-only interpretable quality gate for digitization outputs."""

from __future__ import annotations

from typing import Any

from src.quality.features import QualityFeatureError, QualityFeatures, extract_quality_features
from src.quality.models import (
    QualityGateDecision,
    QualityGateOutcome,
    QualityReason,
    QualityReasonCode,
)
from src.quality.thresholds import DEFAULT_INFERENCE_THRESHOLDS, InferenceThresholds


def _reason(code: QualityReasonCode, message: str) -> QualityReason:
    return QualityReason(code=code, message=message)


def classify_quality(
    record: dict[str, Any],
    thresholds: InferenceThresholds = DEFAULT_INFERENCE_THRESHOLDS,
) -> QualityGateDecision:
    """Classify digitization using only features available during inference."""
    if record.get("status") != "success":
        return QualityGateDecision(
            QualityGateOutcome.REJECT,
            (_reason(QualityReasonCode.DIGITIZATION_FAILED, "digitization failed"),),
        )
    try:
        features = extract_quality_features(record.get("diagnostics") or {})
    except QualityFeatureError as exc:
        return QualityGateDecision(
            QualityGateOutcome.REJECT,
            (_reason(QualityReasonCode.INVALID_FEATURES, f"invalid features: {exc}"),),
        )

    critical: list[QualityReason] = []
    if features.nonzero_leads_count < thresholds.reject_active_leads_below:
        critical.append(
            _reason(
                QualityReasonCode.INSUFFICIENT_ACTIVE_LEADS,
                f"only {features.nonzero_leads_count}/12 active leads",
            )
        )
    if features.detected_leads_count < thresholds.reject_detected_leads_below:
        critical.append(
            _reason(
                QualityReasonCode.INSUFFICIENT_LEAD_LABELS,
                f"only {features.detected_leads_count} lead labels detected",
            )
        )
    if critical:
        return QualityGateDecision(QualityGateOutcome.REJECT, tuple(critical))

    severe = _risk_reasons(features, thresholds, severe=True)
    if len(severe) >= thresholds.severe_flags_to_reject:
        return QualityGateDecision(QualityGateOutcome.REJECT, tuple(severe))
    warnings = _risk_reasons(features, thresholds, severe=False)
    if warnings:
        return QualityGateDecision(QualityGateOutcome.WARN, tuple(warnings))
    return QualityGateDecision(QualityGateOutcome.ACCEPT, ())


def _risk_reasons(
    features: QualityFeatures,
    thresholds: InferenceThresholds,
    *,
    severe: bool,
) -> list[QualityReason]:
    reasons: list[QualityReason] = []
    einthoven_limit = (
        thresholds.severe_einthoven_below if severe else thresholds.warn_einthoven_below
    )
    detected_limit = (
        thresholds.severe_detected_leads_below
        if severe
        else thresholds.warn_detected_leads_below
    )
    if features.einthoven_score < einthoven_limit:
        descriptor = "low" if severe else "reduced"
        reasons.append(
            _reason(
                QualityReasonCode.LOW_EINTHOVEN_CONSISTENCY,
                f"{descriptor} Einthoven consistency ({features.einthoven_score:.2f})",
            )
        )
    if features.layout_cost > thresholds.layout_cost_above:
        reasons.append(
            _reason(
                QualityReasonCode.UNCERTAIN_LAYOUT,
                f"uncertain layout match ({features.layout_cost:.2f})",
            )
        )
    if features.detected_leads_count < detected_limit:
        reasons.append(
            _reason(
                QualityReasonCode.INSUFFICIENT_LEAD_LABELS,
                f"only {features.detected_leads_count} lead labels detected",
            )
        )
    if features.avg_pixel_per_mm < thresholds.pixel_per_mm_below:
        reasons.append(
            _reason(
                QualityReasonCode.LOW_GRID_DENSITY,
                f"low grid calibration density ({features.avg_pixel_per_mm:.2f} px/mm)",
            )
        )
    if features.nonzero_leads_count < thresholds.expected_active_leads:
        reasons.append(
            _reason(
                QualityReasonCode.INSUFFICIENT_ACTIVE_LEADS,
                f"only {features.nonzero_leads_count}/12 active leads",
            )
        )
    return reasons
