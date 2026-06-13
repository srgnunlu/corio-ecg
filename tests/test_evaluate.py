# Tests for ground-truth PTB-XL evaluation summaries.

import numpy as np

from src.training.evaluate import (
    build_ground_truth_evaluation,
    build_official_ground_truth_evaluation,
)
from src.utils.ecg_labels import ECG_FOUNDER_LABELS, NUM_CLASSES


def test_build_ground_truth_evaluation_reports_mapping_coverage() -> None:
    probabilities = np.full((4, NUM_CLASSES), 0.1, dtype=np.float32)
    normal_index = ECG_FOUNDER_LABELS.index("NORMAL ECG")
    afib_index = ECG_FOUNDER_LABELS.index("ATRIAL FIBRILLATION")
    probabilities[:, normal_index] = [0.9, 0.8, 0.1, 0.2]
    probabilities[:, afib_index] = [0.1, 0.2, 0.9, 0.8]
    scp_codes = [
        {"NORM": 100.0},
        {"NORM": 100.0},
        {"AFIB": 100.0},
        {"AFIB": 100.0, "UNMAPPED": 100.0},
    ]

    result = build_ground_truth_evaluation(
        scp_codes,
        probabilities,
        threshold=0.5,
        minimum_positive_examples=1,
    )

    assert result["records_with_mapped_labels"] == 4
    assert result["mapped_label_assignments"] == 4
    assert result["classification"]["evaluated_classes"] == 2
    assert result["classification"]["macro_auroc"] == 1.0


def test_build_ground_truth_evaluation_handles_no_mapped_labels() -> None:
    probabilities = np.zeros((2, NUM_CLASSES), dtype=np.float32)

    result = build_ground_truth_evaluation(
        [{"UNKNOWN": 100.0}, {}],
        probabilities,
        threshold=0.5,
    )

    assert result["records_with_mapped_labels"] == 0
    assert result["classification"]["evaluated_classes"] == 0


def test_build_official_ground_truth_evaluation_matches_filenames(tmp_path) -> None:
    labels_path = tmp_path / "ptbxl_label.csv"
    first = [0] * NUM_CLASSES
    second = [0] * NUM_CLASSES
    first[2] = 1
    second[5] = 1
    labels_path.write_text(
        'filename_hr,label\n'
        f'records/a_hr,"{str(first)}"\n'
        f'records/b_hr,"{str(second)}"\n'
    )
    probabilities = np.zeros((3, NUM_CLASSES), dtype=np.float32)
    probabilities[0, 2] = 0.9
    probabilities[1, 5] = 0.9

    result = build_official_ground_truth_evaluation(
        ["records/a_hr", "records/b_hr", "records/missing_hr"],
        probabilities,
        labels_path,
        threshold=0.5,
        minimum_positive_examples=1,
    )

    assert result["matched_records"] == 2
    assert result["records_missing_labels"] == 1
    assert result["classification"]["evaluated_classes"] == 2
    assert result["classification"]["macro_auroc"] == 1.0
