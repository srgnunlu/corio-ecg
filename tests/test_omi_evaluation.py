# Tests for OMI protocol metrics, bootstrap CIs, subgroups and feature preparation.

import numpy as np
import pandas as pd
import pytest

from src.omi.evaluation import (
    SHORT_INTERVAL_MINUTES,
    build_subgroup_masks,
    operating_point_metrics,
    patient_bootstrap_ci,
    ranking_metrics,
    subgroup_report,
)
from src.omi.features import INPUT_MODES, prepare_median_input
from src.utils.wfdb_helpers import TARGET_LENGTH


class TestOperatingPointMetrics:
    """Tests for confusion-derived metrics."""

    def test_known_confusion_matrix(self) -> None:
        # 2 TP, 1 FN, 1 FP, 2 TN at a 0.5 cutoff.
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1])
        metrics = operating_point_metrics(y_true, scores, 0.5)

        assert metrics["true_positive"] == 2
        assert metrics["false_negative"] == 1
        assert metrics["false_positive"] == 1
        assert metrics["true_negative"] == 2
        assert metrics["sensitivity"] == pytest.approx(2 / 3)
        assert metrics["specificity"] == pytest.approx(2 / 3)
        assert metrics["ppv"] == pytest.approx(2 / 3)
        assert metrics["f1"] == pytest.approx(2 / 3)

    def test_no_predicted_positives_gives_zero_f1(self) -> None:
        metrics = operating_point_metrics(np.array([1, 0]), np.array([0.1, 0.1]), 0.5)
        assert metrics["f1"] == 0.0
        assert metrics["ppv"] == 0.0


class TestRankingMetrics:
    """Tests for AUROC/AUPRC guards."""

    def test_perfect_separation(self) -> None:
        metrics = ranking_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        assert metrics["auroc"] == pytest.approx(1.0)

    def test_single_class_returns_none(self) -> None:
        """A subgroup with no positives cannot be ranked."""
        metrics = ranking_metrics(np.zeros(5), np.random.default_rng(0).random(5))
        assert metrics["auroc"] is None
        assert metrics["auprc"] is None


class TestPatientBootstrap:
    """Tests for patient-level bootstrap intervals."""

    def test_interval_brackets_point_estimate(self) -> None:
        rng = np.random.default_rng(1)
        y_true = np.concatenate([np.ones(40), np.zeros(160)])
        scores = np.concatenate([rng.normal(0.7, 0.2, 40), rng.normal(0.3, 0.2, 160)])
        patients = np.arange(200).astype(str)

        from sklearn.metrics import roc_auc_score

        result = patient_bootstrap_ci(
            y_true, scores, patients, lambda t, s: float(roc_auc_score(t, s)), rounds=200
        )
        assert result["lower"] <= result["point"] <= result["upper"]

    def test_repeated_patients_widen_the_interval(self) -> None:
        """Records from one patient must resample together, not independently."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(2)
        y_true = np.concatenate([np.ones(40), np.zeros(160)])
        scores = np.concatenate([rng.normal(0.7, 0.3, 40), rng.normal(0.3, 0.3, 160)])

        independent = np.arange(200).astype(str)
        # Same data, but every 4 records belong to one patient.
        clustered = np.repeat(np.arange(50), 4).astype(str)

        metric = lambda t, s: float(roc_auc_score(t, s))  # noqa: E731
        wide = patient_bootstrap_ci(y_true, scores, clustered, metric, rounds=400)
        narrow = patient_bootstrap_ci(y_true, scores, independent, metric, rounds=400)
        assert (wide["upper"] - wide["lower"]) > (narrow["upper"] - narrow["lower"])

    def test_returns_empty_result_when_never_two_classes(self) -> None:
        result = patient_bootstrap_ci(
            np.zeros(10), np.random.default_rng(3).random(10),
            np.arange(10).astype(str), lambda t, s: 0.0, rounds=50,
        )
        assert result["rounds"] == 0
        assert result["point"] is None


class TestSubgroups:
    """Tests for the pre-registered subgroup definitions."""

    def _table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "STEMI": [1, 0, 0, 0],
                "NSTEMI": [0, 1, 0, 0],
                "Time_Interval": [60, 60, SHORT_INTERVAL_MINUTES + 1, 5000],
                "CTO": [0, 0, 1, 0],
                "Paced": [0, 0, 0, 1],
                "VF_VT": [0, 0, 0, 0],
                "Prior_PCI": [0, 0, 0, 0],
                "age": [50, 70, 64, 65],
                "gender": [0, 1, 1, 0],
                "OMI": [1, 1, 0, 0],
            }
        )

    def test_acs_positive_is_stemi_or_nstemi(self) -> None:
        masks = build_subgroup_masks(self._table())
        assert masks["acs_positive"].tolist() == [True, True, False, False]

    def test_interval_split_is_exclusive_and_complete(self) -> None:
        masks = build_subgroup_masks(self._table())
        short, long = masks["interval_le_12h"], masks["interval_gt_12h"]
        assert not (short & long).any()
        assert (short | long).all()

    def test_age_split_uses_65_as_the_cut(self) -> None:
        masks = build_subgroup_masks(self._table())
        assert masks["age_lt_65"].tolist() == [True, False, True, False]

    def test_report_marks_single_class_subgroups(self) -> None:
        table = self._table()
        report = subgroup_report(
            table, table.OMI.to_numpy(), np.array([0.9, 0.8, 0.2, 0.1]), 0.5
        )
        # CTO row has no OMI positives, so it gets counts but no ranking metrics.
        assert report["cto"]["omi"] == 0
        assert "auroc" not in report["cto"]
        assert report["overall"]["auroc"] == pytest.approx(1.0)


class TestPrepareMedianInput:
    """Tests for turning a 1 s median beat into a model-length input."""

    def test_output_has_model_length(self) -> None:
        median = np.random.default_rng(4).normal(size=(12, 500))
        prepared = prepare_median_input(median)
        assert prepared.shape == (12, TARGET_LENGTH)

    def test_output_is_z_scored(self) -> None:
        median = np.random.default_rng(5).normal(loc=3.0, scale=2.0, size=(12, 500))
        prepared = prepare_median_input(median)
        assert prepared.mean() == pytest.approx(0.0, abs=1e-6)
        assert prepared.std() == pytest.approx(1.0, abs=1e-6)

    def test_beat_is_repeated_not_padded(self) -> None:
        """Tiling keeps morphology everywhere; zero padding would not."""
        median = np.ones((12, 500))
        prepared = prepare_median_input(median)
        # A constant beat z-scores to all zeros, but nothing should be NaN and
        # the tail must not be a flat pad distinguishable from the head.
        assert np.isfinite(prepared).all()
        np.testing.assert_allclose(prepared[:, :500], prepared[:, 500:1000])

    def test_input_modes_are_the_two_supported_variants(self) -> None:
        assert set(INPUT_MODES) == {"raw", "median"}
