"""Interpretable quality gating for ECG digitization outputs."""

from __future__ import annotations

from typing import Any

from src.evaluation.quality_gate_metrics import evaluate_quality_gate_rows
from src.quality.gate import classify_quality as classify_runtime_quality
from src.quality.models import QualityGateDecision, QualityGateOutcome
from src.training.quality_gate_config import QualityGateConfig, load_quality_gate_config

DEFAULT_QUALITY_GATE_CONFIG = load_quality_gate_config()


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
    """Call the runtime gate with thresholds from the benchmark configuration."""
    resolved_config = config or DEFAULT_QUALITY_GATE_CONFIG
    return classify_runtime_quality(record, resolved_config.inference)


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
