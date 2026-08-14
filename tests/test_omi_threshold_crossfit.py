# Tests for out-of-fit threshold selection, where the reported rates must not
# come from the same records that chose the cutoff.

import numpy as np
import pytest

from src.omi.threshold import (
    MAXIMUM_THRESHOLD,
    MINIMUM_THRESHOLD,
    prevalence_weight,
    select_threshold_crossfit,
    select_threshold_fbeta,
)

RNG = np.random.default_rng(7)


def _patient_problem(patients: int = 120, records_per_patient: int = 2):
    """Overlapping scores with several recordings per patient."""
    groups = np.repeat([f"P{i:04d}" for i in range(patients)], records_per_patient)
    labels = np.repeat((np.arange(patients) % 2 == 0).astype(int), records_per_patient)
    centres = np.where(labels == 1, 0.62, 0.38)
    scores = np.clip(centres + RNG.normal(0.0, 0.18, len(labels)), 0.01, 0.99)
    return labels, scores, groups


def _fbeta(sensitivity: float, precision: float, beta: float = 2.0) -> float:
    denominator = (beta**2 * precision) + sensitivity
    return (1 + beta**2) * precision * sensitivity / denominator if denominator else 0.0


class TestCrossFitSelection:
    """Tests for select_threshold_crossfit."""

    def test_separable_problem_reaches_a_perfect_operating_point(self) -> None:
        """With no overlap, a cutoff fitted elsewhere still separates cleanly."""
        groups = np.array([f"P{i:03d}" for i in range(60)])
        labels = np.array([1] * 30 + [0] * 30)
        scores = np.array([0.9] * 30 + [0.1] * 30)

        result = select_threshold_crossfit(labels, scores, groups)

        assert result.sensitivity == 1.0
        assert result.specificity == 1.0

    def test_reported_rates_are_not_the_in_sample_optimum(self) -> None:
        """Out-of-fit rates must not beat the cutoff fitted on everything."""
        labels, scores, groups = _patient_problem()

        in_sample = select_threshold_fbeta(labels, scores, beta=2.0)
        crossfit = select_threshold_crossfit(labels, scores, groups, beta=2.0)

        assert _fbeta(crossfit.sensitivity, crossfit.precision) <= _fbeta(
            in_sample.sensitivity, in_sample.precision
        )

    def test_beta_two_stays_more_sensitive_than_beta_one(self) -> None:
        labels, scores, groups = _patient_problem()

        lenient = select_threshold_crossfit(labels, scores, groups, beta=2.0)
        strict = select_threshold_crossfit(labels, scores, groups, beta=1.0)

        assert lenient.threshold < strict.threshold
        assert lenient.sensitivity > strict.sensitivity

    def test_splits_patients_not_records(self) -> None:
        """One patient cannot fill two folds — proof the split is group-level."""
        labels = np.array([1] * 20 + [0] * 20)
        scores = np.linspace(0.1, 0.9, 40)
        groups = np.full(40, "P0001")

        with pytest.raises(ValueError, match="cannot fill"):
            select_threshold_crossfit(labels, scores, groups, folds=2)

    def test_deployable_threshold_is_the_median_of_the_fits(self) -> None:
        labels, scores, groups = _patient_problem()

        result = select_threshold_crossfit(labels, scores, groups, folds=2, repeats=5)

        assert len(result.fold_thresholds) == 10
        assert result.threshold == pytest.approx(np.median(result.fold_thresholds))
        assert MINIMUM_THRESHOLD <= result.threshold <= MAXIMUM_THRESHOLD

    def test_result_is_reproducible_for_a_fixed_seed(self) -> None:
        labels, scores, groups = _patient_problem()

        first = select_threshold_crossfit(labels, scores, groups, seed=99)
        second = select_threshold_crossfit(labels, scores, groups, seed=99)

        assert first.threshold == second.threshold
        assert first.sensitivity == second.sensitivity

    def test_enrichment_correction_raises_the_cutoff(self) -> None:
        """An enriched subset flatters precision, so it picks too low a cutoff."""
        labels, scores, groups = _patient_problem()
        weight = prevalence_weight(labels, target_prevalence=0.064)

        enriched = select_threshold_crossfit(labels, scores, groups)
        corrected = select_threshold_crossfit(
            labels, scores, groups, negative_weight=weight
        )

        assert corrected.threshold > enriched.threshold
        assert corrected.specificity > enriched.specificity

    def test_result_serializes_for_the_report(self) -> None:
        labels, scores, groups = _patient_problem()

        payload = select_threshold_crossfit(labels, scores, groups).to_dict()

        assert set(payload) == {
            "strategy",
            "threshold",
            "sensitivity",
            "specificity",
            "precision",
            "fold_thresholds",
            "repeats",
            "folds",
        }
        assert payload["strategy"] == "f2_max_crossfit"


class TestPrevalenceWeight:
    """Tests for the enrichment correction."""

    def test_balanced_subset_needs_a_large_multiplier(self) -> None:
        """230 positives against 230 negatives must act like 1 in 15.6."""
        labels = np.array([1] * 230 + [0] * 230)

        assert prevalence_weight(labels, 0.064) == pytest.approx(0.936 / 0.064, rel=1e-6)

    def test_matching_prevalence_needs_no_correction(self) -> None:
        labels = np.array([1] * 64 + [0] * 936)

        assert prevalence_weight(labels, 0.064) == pytest.approx(1.0, rel=1e-3)

    def test_single_class_subset_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="both classes"):
            prevalence_weight(np.ones(10, dtype=int), 0.064)

    def test_impossible_target_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must lie in"):
            prevalence_weight(np.array([1, 0, 1, 0]), 0.0)
