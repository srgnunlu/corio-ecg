# Tests for round-trip evaluation comparison helpers.

import numpy as np
import pytest

from src.training.evaluate_roundtrip import (
    _aggregate_probability_vectors,
    _build_layout_column_signals,
    _classification_metric_deltas,
)


def test_classification_metric_deltas_compare_matched_record_sets() -> None:
    baseline = {
        "macro_auroc": 0.9,
        "macro_average_precision": 0.8,
        "micro_f1": 0.7,
        "macro_f1": 0.6,
    }
    roundtrip = {
        "macro_auroc": 0.85,
        "macro_average_precision": 0.7,
        "micro_f1": 0.65,
        "macro_f1": 0.5,
    }

    result = _classification_metric_deltas(baseline, roundtrip)

    assert result == {
        "macro_auroc": -0.05,
        "macro_average_precision": -0.1,
        "micro_f1": -0.05,
        "macro_f1": -0.1,
    }


def test_build_layout_column_signals_preserves_only_3x4_column_leads() -> None:
    signal = np.stack(
        [np.full(20, lead_index + 1, dtype=np.float32) for lead_index in range(12)]
    )

    columns = _build_layout_column_signals(signal, "3x4")

    assert len(columns) == 4
    expected_active_leads = (
        {0, 1, 2},
        {3, 4, 5},
        {6, 7, 8},
        {9, 10, 11},
    )
    for column, expected in zip(columns, expected_active_leads, strict=True):
        active = set(np.flatnonzero(np.max(np.abs(column), axis=1)))
        assert active == expected
        np.testing.assert_array_equal(column[list(expected)], signal[list(expected)])


def test_build_layout_column_signals_supports_6x2_layout() -> None:
    signal = np.ones((12, 20), dtype=np.float32)

    columns = _build_layout_column_signals(signal, "standard_6x2+1R")

    assert len(columns) == 2
    assert set(np.flatnonzero(np.max(np.abs(columns[0]), axis=1))) == set(range(6))
    assert set(np.flatnonzero(np.max(np.abs(columns[1]), axis=1))) == set(range(6, 12))


def test_build_layout_column_signals_rejects_unknown_layout() -> None:
    with pytest.raises(ValueError, match="Unsupported paper layout"):
        _build_layout_column_signals(np.ones((12, 20)), "auto")


def test_aggregate_probability_vectors_supports_mean_and_maximum() -> None:
    vectors = [
        np.array([0.1, 0.8, 0.3]),
        np.array([0.5, 0.2, 0.7]),
    ]

    np.testing.assert_allclose(
        _aggregate_probability_vectors(vectors, "mean"),
        np.array([0.3, 0.5, 0.5]),
    )
    np.testing.assert_allclose(
        _aggregate_probability_vectors(vectors, "max"),
        np.array([0.5, 0.8, 0.7]),
    )


def test_aggregate_probability_vectors_rejects_empty_or_unknown_method() -> None:
    with pytest.raises(ValueError, match="At least one"):
        _aggregate_probability_vectors([], "mean")
    with pytest.raises(ValueError, match="Unsupported probability aggregation"):
        _aggregate_probability_vectors([np.ones(3)], "median")
