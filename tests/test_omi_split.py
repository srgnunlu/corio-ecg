# Tests for the patient-grouped OMI split and dataset loading helpers.

import numpy as np
import pandas as pd
import pytest

from src.omi.dataset import (
    DIAGNOSIS_LABELS,
    MEDIAN_LEAD_COUNT,
    MEDIAN_SAMPLE_COUNT,
    build_label_matrix,
    load_median_beat,
    select_fold,
)
from src.omi.split import (
    SplitConfig,
    assign_folds,
    summarise_folds,
    verify_no_patient_leakage,
)

RNG = np.random.default_rng(5)


def _synthetic_table(n_patients: int = 400, positive_rate: float = 0.08) -> pd.DataFrame:
    """Build a table where some patients carry several recordings."""
    rows = []
    for patient_index in range(n_patients):
        patient = f"P{patient_index:05d}"
        # Every fifth patient contributes two ECGs, mirroring the real dataset.
        for recording in range(2 if patient_index % 5 == 0 else 1):
            rows.append(
                {
                    "Patient_id": patient,
                    "ecg_row_record": f"{patient_index:05d}_{recording}.dat",
                    "OMI": int(RNG.random() < positive_rate),
                }
            )
    return pd.DataFrame(rows)


class TestAssignFolds:
    """Tests for fold assignment."""

    def test_every_record_gets_a_fold(self) -> None:
        table = _synthetic_table()
        assignment = assign_folds(table, SplitConfig())
        assert len(assignment) == len(table)
        assert assignment.fold.between(0, 4).all()

    def test_patients_never_span_folds(self) -> None:
        """The whole point of grouping — one patient, one fold."""
        table = _synthetic_table()
        assignment = assign_folds(table, SplitConfig())
        folds_per_patient = assignment.groupby("Patient_id").fold.nunique()
        assert folds_per_patient.max() == 1

    def test_assignment_is_deterministic(self) -> None:
        table = _synthetic_table()
        first = assign_folds(table, SplitConfig())
        second = assign_folds(table, SplitConfig())
        pd.testing.assert_frame_equal(first, second)

    def test_seed_changes_assignment(self) -> None:
        table = _synthetic_table()
        default = assign_folds(table, SplitConfig())
        other = assign_folds(table, SplitConfig(seed=999))
        assert not default.fold.equals(other.fold)

    def test_rejects_missing_columns(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            assign_folds(pd.DataFrame({"Patient_id": ["P1"]}), SplitConfig())

    def test_verify_accepts_a_clean_assignment(self) -> None:
        table = _synthetic_table()
        stats = verify_no_patient_leakage(assign_folds(table, SplitConfig()), SplitConfig())
        assert stats["records"] == len(table)
        assert stats["folds"] == 5

    def test_verify_rejects_a_leaking_assignment(self) -> None:
        leaking = pd.DataFrame(
            {
                "ecg_row_record": ["a.dat", "b.dat"],
                "Patient_id": ["P1", "P1"],
                "fold": [0, 1],
            }
        )
        with pytest.raises(ValueError, match="span multiple folds"):
            verify_no_patient_leakage(leaking, SplitConfig())

    def test_summary_covers_all_folds(self) -> None:
        table = _synthetic_table()
        summary = summarise_folds(assign_folds(table, SplitConfig()), table, SplitConfig())
        assert len(summary) == 5
        assert summary.records.sum() == len(table)


class TestSelectFold:
    """Tests for fold selection."""

    def _table(self) -> pd.DataFrame:
        return pd.DataFrame({"ecg_row_record": list("abcd"), "fold": [0, 0, 1, 2]})

    def test_holdout_returns_only_that_fold(self) -> None:
        assert len(select_fold(self._table(), 0)) == 2

    def test_non_holdout_returns_the_rest(self) -> None:
        assert len(select_fold(self._table(), 0, holdout=False)) == 2

    def test_requires_a_fold_column(self) -> None:
        with pytest.raises(ValueError, match="no fold column"):
            select_fold(pd.DataFrame({"ecg_row_record": ["a"]}), 0)


class TestMedianBeat:
    """Tests for decoding the raw median-beat files."""

    def test_decodes_interleaved_microvolts(self, tmp_path) -> None:
        # Lead k is a constant k*100 uV, written sample-major (interleaved).
        expected_leads = np.arange(MEDIAN_LEAD_COUNT, dtype=np.int16) * 100
        interleaved = np.tile(expected_leads, MEDIAN_SAMPLE_COUNT).astype("<i2")
        path = tmp_path / "med_data" / "00001.med"
        path.parent.mkdir(parents=True)
        interleaved.tofile(path)

        median = load_median_beat(tmp_path, "00001.med")
        assert median.shape == (MEDIAN_LEAD_COUNT, MEDIAN_SAMPLE_COUNT)
        # Values come back in millivolts, one constant per lead.
        np.testing.assert_allclose(median[:, 0], expected_leads / 1000.0)

    def test_rejects_wrong_sample_count(self, tmp_path) -> None:
        path = tmp_path / "med_data" / "bad.med"
        path.parent.mkdir(parents=True)
        np.zeros(10, dtype="<i2").tofile(path)
        with pytest.raises(ValueError, match="expected 6000 int16 samples"):
            load_median_beat(tmp_path, "bad.med")


class TestBuildLabelMatrix:
    """Tests for label matrix assembly."""

    def test_stacks_requested_columns(self) -> None:
        table = pd.DataFrame({label: [0, 1] for label in DIAGNOSIS_LABELS})
        matrix = build_label_matrix(table, DIAGNOSIS_LABELS)
        assert matrix.shape == (2, len(DIAGNOSIS_LABELS))

    def test_rejects_missing_label(self) -> None:
        with pytest.raises(ValueError, match="missing label columns"):
            build_label_matrix(pd.DataFrame({"OMI": [0]}), ["OMI", "STEMI"])
