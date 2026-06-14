"""Tests for the inference-time quality feature contract."""

from __future__ import annotations

import pytest

from src.quality.features import (
    QualityFeatureError,
    extract_quality_features,
    load_quality_feature_contract,
)


def _diagnostics() -> dict[str, object]:
    return {
        "layout_cost": 0.2,
        "detected_leads_count": 10,
        "nonzero_leads_count": 12,
        "einthoven_score": 0.95,
        "avg_pixel_per_mm": 9.0,
        "raw_lines_count": 4,
    }


def test_quality_feature_contract_documents_active_features() -> None:
    contract = load_quality_feature_contract()

    assert contract.version == "quality-feature-contract-v1"
    assert set(contract.features) == set(_diagnostics())
    for definition in contract.features.values():
        assert definition.description
        assert definition.unit
        assert definition.missing_policy == "reject"
        assert definition.minimum is not None
        assert definition.maximum is not None


def test_extract_quality_features_returns_typed_values() -> None:
    features = extract_quality_features(_diagnostics())

    assert features.layout_cost == pytest.approx(0.2)
    assert features.detected_leads_count == 10
    assert features.raw_lines_count == 4


def test_extract_quality_features_rejects_missing_required_feature() -> None:
    diagnostics = _diagnostics()
    diagnostics.pop("layout_cost")

    with pytest.raises(QualityFeatureError, match="layout_cost.*missing"):
        extract_quality_features(diagnostics)


def test_extract_quality_features_rejects_out_of_range_feature() -> None:
    diagnostics = _diagnostics()
    diagnostics["einthoven_score"] = 1.5

    with pytest.raises(QualityFeatureError, match="einthoven_score.*outside"):
        extract_quality_features(diagnostics)


def test_extract_quality_features_rejects_wrong_type() -> None:
    diagnostics = _diagnostics()
    diagnostics["detected_leads_count"] = 10.5

    with pytest.raises(QualityFeatureError, match="detected_leads_count.*integer"):
        extract_quality_features(diagnostics)
