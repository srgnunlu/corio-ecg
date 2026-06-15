"""Tests for reference-free reconstruction disagreement metrics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.evaluation.reconstruction_disagreement import evaluate_reconstruction_pair


def _signals() -> NDArray[np.float64]:
    time = np.linspace(0.0, 10.0, 5000, endpoint=False)
    return np.asarray(
        [np.sin(2.0 * np.pi * (lead + 1) * time / 10.0) for lead in range(12)],
        dtype=np.float64,
    )


def test_identical_reconstructions_have_no_disagreement() -> None:
    signals = _signals()

    result = evaluate_reconstruction_pair(
        signals,
        signals.copy(),
        first_calibrated=signals,
        second_calibrated=signals.copy(),
    )

    assert result["median_correlation"] == pytest.approx(1.0)
    assert result["unstable_leads_below_0_8"] == 0
    assert result["calibrated"]["median_rmse_mv"] == pytest.approx(0.0)
    assert result["calibrated"]["median_gain_error"] == pytest.approx(0.0)


def test_metric_tolerates_small_horizontal_shift() -> None:
    signals = _signals()
    shifted = np.roll(signals, 20, axis=1)

    result = evaluate_reconstruction_pair(signals, shifted, max_shift_samples=25)

    assert result["median_correlation"] > 0.999
    assert result["unstable_leads_below_0_8"] == 0


def test_metric_detects_morphology_and_gain_disagreement() -> None:
    signals = _signals()
    changed = signals.copy()
    changed[0] = np.cos(np.linspace(0.0, 40.0, 5000))

    result = evaluate_reconstruction_pair(
        signals,
        changed,
        first_calibrated=signals,
        second_calibrated=changed * 1.5,
    )

    assert result["minimum_correlation"] < 0.2
    assert result["unstable_leads_below_0_8"] == 1
    assert result["calibrated"]["median_gain_error"] == pytest.approx(0.5, abs=0.01)


def test_metric_requires_paired_shapes_and_calibrated_inputs() -> None:
    signals = _signals()

    with pytest.raises(ValueError, match="identical shapes"):
        evaluate_reconstruction_pair(signals, signals[:, :-1])
    with pytest.raises(ValueError, match="both calibrated"):
        evaluate_reconstruction_pair(signals, signals, first_calibrated=signals)
