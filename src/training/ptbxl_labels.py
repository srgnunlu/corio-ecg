# Explicit mapping from PTB-XL SCP codes to semantically equivalent ECGFounder outputs.

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from src.utils.ecg_labels import ECG_FOUNDER_LABELS, NUM_CLASSES

# Only direct semantic matches are included. PTB-XL codes without a stable
# ECGFounder equivalent are intentionally excluded from ground-truth metrics.
SCP_TO_ECGFOUNDER_LABEL: dict[str, str] = {
    "NORM": "NORMAL ECG",
    "SR": "SINUS RHYTHM",
    "SBRAD": "SINUS BRADYCARDIA",
    "STACH": "SINUS TACHYCARDIA",
    "SARRH": "WITH SINUS ARRHYTHMIA",
    "AFIB": "ATRIAL FIBRILLATION",
    "AFLT": "ATRIAL FLUTTER",
    "SVTAC": "SUPRAVENTRICULAR TACHYCARDIA",
    "PSVT": "SUPRAVENTRICULAR TACHYCARDIA",
    "PVC": "PREMATURE VENTRICULAR COMPLEXES",
    "PAC": "PREMATURE ATRIAL COMPLEXES",
    "PRC(S)": "PREMATURE ECTOPIC COMPLEXES",
    "BIGU": "IN A PATTERN OF BIGEMINY",
    "CRBBB": "RIGHT BUNDLE BRANCH BLOCK",
    "IRBBB": "INCOMPLETE RIGHT BUNDLE BRANCH BLOCK",
    "CLBBB": "LEFT BUNDLE BRANCH BLOCK",
    "ILBBB": "INCOMPLETE LEFT BUNDLE BRANCH BLOCK",
    "LAFB": "LEFT ANTERIOR FASCICULAR BLOCK",
    "LPFB": "LEFT POSTERIOR FASCICULAR BLOCK",
    "1AVB": "WITH 1ST DEGREE AV BLOCK",
    "3AVB": "WITH COMPLETE HEART BLOCK",
    "IVCD": "NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY",
    "WPW": "WOLFF-PARKINSON-WHITE",
    "LNGQT": "PROLONGED QT",
    "LVH": "LEFT VENTRICULAR HYPERTROPHY",
    "RVH": "RIGHT VENTRICULAR HYPERTROPHY",
    "LAO/LAE": "LEFT ATRIAL ENLARGEMENT",
    "RAO/RAE": "RIGHT ATRIAL ENLARGEMENT",
    "LVOLT": "LOW VOLTAGE QRS",
    "VCLVH": "VOLTAGE CRITERIA FOR LEFT VENTRICULAR HYPERTROPHY",
    "NDT": "NONSPECIFIC T WAVE ABNORMALITY",
    "NT_": "NONSPECIFIC T WAVE ABNORMALITY",
    "TAB_": "NONSPECIFIC T WAVE ABNORMALITY",
    "NST_": "NONSPECIFIC ST ABNORMALITY",
    "DIG": "OR DIGITALIS EFFECT",
    "IMI": "INFERIOR INFARCT",
    "ASMI": "ANTEROSEPTAL INFARCT",
    "AMI": "ANTERIOR INFARCT",
    "ALMI": "ANTEROLATERAL INFARCT",
    "LMI": "LATERAL INFARCT",
    "PMI": "POSTERIOR INFARCT",
    "IPMI": "INFERIOR-POSTERIOR INFARCT",
}

_LABEL_TO_INDEX = {label: index for index, label in enumerate(ECG_FOUNDER_LABELS)}

MAPPED_ECGFOUNDER_INDICES: list[int] = sorted(
    {_LABEL_TO_INDEX[label] for label in SCP_TO_ECGFOUNDER_LABEL.values()}
)
MAPPED_ECGFOUNDER_LABELS: list[str] = [
    ECG_FOUNDER_LABELS[index] for index in MAPPED_ECGFOUNDER_INDICES
]


def map_scp_codes_to_indices(scp_codes: Mapping[str, float]) -> set[int]:
    """Map the SCP codes present on one PTB-XL record to ECGFounder indices."""
    return {
        _LABEL_TO_INDEX[SCP_TO_ECGFOUNDER_LABEL[scp_code]]
        for scp_code in scp_codes
        if scp_code in SCP_TO_ECGFOUNDER_LABEL
    }


def build_ground_truth_matrix(
    records_scp_codes: Iterable[Mapping[str, float]],
) -> np.ndarray:
    """Build a binary 150-output ground-truth matrix from PTB-XL SCP codes."""
    records = list(records_scp_codes)
    matrix = np.zeros((len(records), NUM_CLASSES), dtype=np.int8)
    for row_index, scp_codes in enumerate(records):
        for class_index in map_scp_codes_to_indices(scp_codes):
            matrix[row_index, class_index] = 1
    return matrix
