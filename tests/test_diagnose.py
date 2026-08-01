# Tests for raw-model versus post-processed ECGFounder inference.

from unittest.mock import MagicMock

import numpy as np
import torch

from src.pipeline.diagnose import ECGDiagnoser
from src.utils.ecg_labels import NUM_CLASSES


def _mock_diagnoser() -> ECGDiagnoser:
    diagnoser = object.__new__(ECGDiagnoser)
    diagnoser.device = torch.device("cpu")
    diagnoser.threshold = 0.5
    # These tests cover raw-model behaviour, so calibration stays off.
    diagnoser.calibration = None
    diagnoser.last_estimated_hr_bpm = None
    diagnoser.model = MagicMock(return_value=torch.zeros((1, NUM_CLASSES)))
    diagnoser._apply_rate_consistency_adjustments = MagicMock(
        return_value=np.ones(NUM_CLASSES, dtype=np.float32)
    )
    return diagnoser


def test_diagnose_all_can_return_raw_model_probabilities() -> None:
    diagnoser = _mock_diagnoser()

    results = diagnoser.diagnose_all(
        np.zeros((12, 5000), dtype=np.float32),
        apply_rate_adjustments=False,
    )

    diagnoser._apply_rate_consistency_adjustments.assert_not_called()
    assert all(result.probability == 0.5 for result in results)


def test_diagnose_all_applies_rate_adjustments_by_default() -> None:
    diagnoser = _mock_diagnoser()
    signal = np.zeros((12, 5000), dtype=np.float32)

    results = diagnoser.diagnose_all(signal)

    diagnoser._apply_rate_consistency_adjustments.assert_called_once()
    assert all(result.probability == 1.0 for result in results)


def test_segment_ensemble_runs_one_forward_per_paper_column_and_averages() -> None:
    diagnoser = _mock_diagnoser()
    column_vectors = [
        np.full(NUM_CLASSES, 0.2, dtype=np.float32),
        np.full(NUM_CLASSES, 0.4, dtype=np.float32),
        np.full(NUM_CLASSES, 0.6, dtype=np.float32),
        np.full(NUM_CLASSES, 0.8, dtype=np.float32),
    ]
    diagnoser._forward_probabilities = MagicMock(side_effect=column_vectors)

    results = diagnoser.diagnose_segment_ensemble(
        np.zeros((12, 5000), dtype=np.float32),
        threshold=0.0,
        layout="3x4",
        apply_rate_adjustments=False,
    )

    # 3x4 has four printed columns -> four independent forward passes.
    assert diagnoser._forward_probabilities.call_count == 4
    diagnoser._apply_rate_consistency_adjustments.assert_not_called()
    # mean([0.2, 0.4, 0.6, 0.8]) == 0.5
    assert all(abs(result.probability - 0.5) < 1e-6 for result in results)


def test_segment_ensemble_applies_rate_adjustments_by_default() -> None:
    diagnoser = _mock_diagnoser()
    diagnoser._forward_probabilities = MagicMock(
        return_value=np.full(NUM_CLASSES, 0.3, dtype=np.float32)
    )

    results = diagnoser.diagnose_all_segment_ensemble(
        np.zeros((12, 5000), dtype=np.float32),
        layout="3x4",
    )

    diagnoser._apply_rate_consistency_adjustments.assert_called_once()
    assert all(result.probability == 1.0 for result in results)
