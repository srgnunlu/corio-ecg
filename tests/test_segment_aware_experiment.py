"""Tests for segment-aware experiment decision reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.segment_aware_experiment import (
    evaluate_segment_aware_experiment,
    load_segment_aware_experiment_config,
)


def _method(cosine: float, abs_diff: float, agreement: float) -> dict[str, float]:
    return {
        "mean_cosine_similarity": cosine,
        "mean_abs_probability_difference": abs_diff,
        "mean_agreement_rate": agreement,
    }


def _report(
    *,
    supported_candidate_cosine: float = 0.91,
    target_candidate_cosine: float = 0.72,
) -> dict[str, object]:
    return {
        "aggregate": {
            "categories": {
                "photos_scans": {
                    "total": 10,
                    "successful": 10,
                    "failed": 0,
                    "methods": {
                        "tiled": _method(0.90, 0.05, 0.95),
                        "segment_ensemble": _method(
                            supported_candidate_cosine,
                            0.04,
                            0.96,
                        ),
                    }
                },
                "photos_bents": {
                    "total": 10,
                    "successful": 10,
                    "failed": 0,
                    "methods": {
                        "tiled": _method(0.70, 0.10, 0.90),
                        "segment_ensemble": _method(
                            target_candidate_cosine,
                            0.08,
                            0.92,
                        ),
                    }
                },
            }
        }
    }


def test_segment_aware_experiment_passes_target_gain_without_supported_regression() -> None:
    config = load_segment_aware_experiment_config()
    config = config.with_categories(
        supported_categories=("photos_scans",),
        target_categories=("photos_bents",),
    )

    result = evaluate_segment_aware_experiment(_report(), config=config)

    assert result["research_candidate"]["passed"] is True
    assert result["categories"]["photos_bents"]["deltas"]["mean_cosine_similarity"] == (
        pytest.approx(0.02)
    )
    assert result["categories"]["photos_bents"]["source_counts"]["failed"] == 0
    assert result["production_status"] == "experimental-only"


def test_segment_aware_experiment_blocks_supported_regression() -> None:
    config = load_segment_aware_experiment_config().with_categories(
        supported_categories=("photos_scans",),
        target_categories=("photos_bents",),
    )

    result = evaluate_segment_aware_experiment(
        _report(supported_candidate_cosine=0.88),
        config=config,
    )

    assert result["research_candidate"]["passed"] is False
    assert result["research_candidate"]["failed_checks"][0]["category"] == "photos_scans"


def test_segment_aware_experiment_requires_target_improvement() -> None:
    config = load_segment_aware_experiment_config().with_categories(
        supported_categories=("photos_scans",),
        target_categories=("photos_bents",),
    )

    result = evaluate_segment_aware_experiment(
        _report(target_candidate_cosine=0.705),
        config=config,
    )

    assert result["research_candidate"]["passed"] is False
    assert result["research_candidate"]["failed_checks"][0]["role"] == "target"


def test_load_segment_aware_config_rejects_negative_tolerance(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
version: invalid
baseline_method: tiled
candidate_method: segment_ensemble
supported_categories: [photos_scans]
target_categories: [photos_bents]
tolerances:
  mean_cosine_similarity_drop: -0.01
  mean_abs_probability_difference_increase: 0.0
  mean_agreement_rate_drop: 0.0
  target_mean_cosine_similarity_gain: 0.01
""".strip()
    )

    with pytest.raises(ValueError, match="non-negative"):
        load_segment_aware_experiment_config(config_path)
