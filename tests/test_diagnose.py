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
