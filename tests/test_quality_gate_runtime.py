"""Tests for the runtime-only quality decision API."""

from __future__ import annotations

from pathlib import Path

import src.quality.gate as gate_module
from src.quality.gate import classify_quality
from src.quality.models import QualityGateOutcome, QualityReasonCode


def _record(**changes: object) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "layout_cost": 0.2,
        "detected_leads_count": 10,
        "nonzero_leads_count": 12,
        "einthoven_score": 0.95,
        "avg_pixel_per_mm": 9.0,
        "raw_lines_count": 4,
    }
    diagnostics.update(changes)
    return {"status": "success", "diagnostics": diagnostics}


def test_runtime_gate_exposes_stable_reason_codes() -> None:
    decision = classify_quality(_record(nonzero_leads_count=9))

    assert decision.outcome is QualityGateOutcome.REJECT
    assert decision.reason_codes == (QualityReasonCode.INSUFFICIENT_ACTIVE_LEADS.value,)
    assert decision.reasons == ("only 9/12 active leads",)


def test_runtime_gate_reports_multiple_independent_risk_codes() -> None:
    decision = classify_quality(_record(einthoven_score=0.5, layout_cost=0.8))

    assert decision.outcome is QualityGateOutcome.REJECT
    assert decision.reason_codes == (
        QualityReasonCode.LOW_EINTHOVEN_CONSISTENCY.value,
        QualityReasonCode.UNCERTAIN_LAYOUT.value,
    )


def test_runtime_gate_rejects_invalid_feature_contract() -> None:
    decision = classify_quality(_record(raw_lines_count="invalid"))

    assert decision.outcome is QualityGateOutcome.REJECT
    assert decision.reason_codes == (QualityReasonCode.INVALID_FEATURES.value,)


def test_runtime_gate_module_has_no_training_or_matched_reference_imports() -> None:
    source = Path(gate_module.__file__).read_text()

    assert "src.training" not in source
    assert "classify_fidelity_target" not in source
