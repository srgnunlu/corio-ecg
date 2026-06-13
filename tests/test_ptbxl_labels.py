# Tests for the explicit PTB-XL SCP-code to ECGFounder label mapping.

import numpy as np

from src.training.ptbxl_labels import (
    MAPPED_ECGFOUNDER_INDICES,
    build_ground_truth_matrix,
    map_scp_codes_to_indices,
)
from src.utils.ecg_labels import ECG_FOUNDER_LABELS, NUM_CLASSES


def test_maps_supported_codes_even_when_confidence_is_zero() -> None:
    indices = map_scp_codes_to_indices({"NORM": 100.0, "SR": 0.0, "UNKNOWN": 50.0})

    assert ECG_FOUNDER_LABELS.index("NORMAL ECG") in indices
    assert ECG_FOUNDER_LABELS.index("SINUS RHYTHM") in indices
    assert len(indices) == 2


def test_multiple_scp_codes_can_map_to_same_ecgfounder_label() -> None:
    indices = map_scp_codes_to_indices({"SVTAC": 100.0, "PSVT": 100.0})

    assert indices == {ECG_FOUNDER_LABELS.index("SUPRAVENTRICULAR TACHYCARDIA")}


def test_build_ground_truth_matrix_uses_all_150_model_outputs() -> None:
    matrix = build_ground_truth_matrix(
        [
            {"AFIB": 100.0, "CRBBB": 100.0},
            {"NORM": 100.0},
        ]
    )

    assert matrix.shape == (2, NUM_CLASSES)
    assert matrix.dtype == np.int8
    assert matrix[0, ECG_FOUNDER_LABELS.index("ATRIAL FIBRILLATION")] == 1
    assert matrix[0, ECG_FOUNDER_LABELS.index("RIGHT BUNDLE BRANCH BLOCK")] == 1
    assert matrix[1, ECG_FOUNDER_LABELS.index("NORMAL ECG")] == 1


def test_mapping_indices_are_unique_and_valid() -> None:
    assert len(MAPPED_ECGFOUNDER_INDICES) == len(set(MAPPED_ECGFOUNDER_INDICES))
    assert all(0 <= index < NUM_CLASSES for index in MAPPED_ECGFOUNDER_INDICES)
