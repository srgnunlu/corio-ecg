"""Tests for tune-only supported-layout policy evaluation."""

from __future__ import annotations

import pytest

from src.evaluation.layout_scope_experiment import LayoutScopeConfig, evaluate_layout_scope


def _config() -> LayoutScopeConfig:
    return LayoutScopeConfig(
        version="layout-scope-v1",
        supported_layouts=("3x4+1R",),
        require_layout_hint=True,
    )


def test_layout_scope_rejects_unsupported_layouts() -> None:
    quality = {
        "selected_split": "tune",
        "records": [
            {
                "ecg_id": "ecg-1",
                "image_id": 1,
                "category": "scan",
                "target": "reject",
                "prediction": "accept",
            }
        ],
    }
    fidelity = {
        "records": [
            {"ecg_id": "ecg-1", "image_id": 1, "category": "scan", "layout": "6x2"}
        ]
    }

    result = evaluate_layout_scope(quality, fidelity, _config())

    assert result["records"][0]["prediction"] == "reject"
    assert result["unsupported_records_rejected"] == 1
    assert result["aggregate"]["false_accepts"] == 0


def test_layout_scope_preserves_base_gate_inside_scope() -> None:
    quality = {
        "selected_split": "tune",
        "records": [
            {
                "ecg_id": "ecg-1",
                "image_id": 1,
                "category": "scan",
                "target": "reject",
                "prediction": "accept",
            }
        ],
    }
    fidelity = {
        "records": [
            {
                "ecg_id": "ecg-1",
                "image_id": 1,
                "category": "scan",
                "layout": "3x4+1R",
            }
        ]
    }

    result = evaluate_layout_scope(quality, fidelity, _config())

    assert result["records"][0]["prediction"] == "accept"
    assert result["aggregate"]["false_accepts"] == 1


def test_layout_scope_rejects_non_tune_reports() -> None:
    with pytest.raises(ValueError, match="only on the tune split"):
        evaluate_layout_scope(
            {"selected_split": "test", "records": []},
            {"records": []},
            _config(),
        )
