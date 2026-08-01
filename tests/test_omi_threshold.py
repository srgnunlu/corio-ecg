# Tests for sensitivity-weighted threshold selection.

import numpy as np
import pytest

from src.omi.threshold import (
    MAXIMUM_THRESHOLD,
    MINIMUM_THRESHOLD,
    fbeta_score_at,
    select_threshold_at_min_sensitivity,
    select_threshold_at_min_specificity,
    select_threshold_fbeta,
)

RNG = np.random.default_rng(11)


def _graded_problem(n_positive: int = 100, n_negative: int = 400):
    """Overlapping score distributions, so thresholds actually trade off."""
    positives = np.clip(RNG.normal(0.65, 0.20, n_positive), 0.01, 0.99)
    negatives = np.clip(RNG.normal(0.35, 0.20, n_negative), 0.01, 0.99)
    y_true = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
    scores = np.concatenate([positives, negatives])
    return y_true, scores


class TestFbetaScore:
    """Tests for the F-beta helper."""

    def test_beta_one_matches_f1(self) -> None:
        from sklearn.metrics import f1_score

        y_true, scores = _graded_problem()
        expected = f1_score(y_true, scores >= 0.5, zero_division=0)
        assert fbeta_score_at(y_true, scores, 0.5, beta=1.0) == pytest.approx(expected)

    def test_beta_two_matches_sklearn(self) -> None:
        from sklearn.metrics import fbeta_score

        y_true, scores = _graded_problem()
        expected = fbeta_score(y_true, scores >= 0.5, beta=2.0, zero_division=0)
        assert fbeta_score_at(y_true, scores, 0.5, beta=2.0) == pytest.approx(expected)

    def test_no_predictions_scores_zero(self) -> None:
        y_true, scores = _graded_problem()
        assert fbeta_score_at(y_true, scores, 1.5) == 0.0


class TestSelectThresholdFbeta:
    """Tests for F-beta threshold selection."""

    def test_beta_two_is_more_sensitive_than_beta_one(self) -> None:
        """The whole point: weighting recall harder must lower the cutoff."""
        y_true, scores = _graded_problem()
        f1_choice = select_threshold_fbeta(y_true, scores, beta=1.0)
        f2_choice = select_threshold_fbeta(y_true, scores, beta=2.0)

        assert f2_choice.threshold < f1_choice.threshold
        assert f2_choice.sensitivity > f1_choice.sensitivity

    def test_strategy_name_records_beta(self) -> None:
        y_true, scores = _graded_problem()
        assert select_threshold_fbeta(y_true, scores, beta=2.0).strategy == "f2_max"

    def test_threshold_stays_within_bounds(self) -> None:
        y_true, scores = _graded_problem()
        choice = select_threshold_fbeta(y_true, scores)
        assert MINIMUM_THRESHOLD <= choice.threshold <= MAXIMUM_THRESHOLD


class TestConstrainedSelection:
    """Tests for the iso-sensitivity and iso-specificity strategies."""

    def test_meets_the_sensitivity_floor(self) -> None:
        y_true, scores = _graded_problem()
        choice = select_threshold_at_min_sensitivity(y_true, scores, 0.70)
        assert choice.sensitivity >= 0.70

    def test_meets_the_specificity_floor(self) -> None:
        y_true, scores = _graded_problem()
        choice = select_threshold_at_min_specificity(y_true, scores, 0.85)
        assert choice.specificity >= 0.85

    def test_higher_sensitivity_floor_costs_specificity(self) -> None:
        y_true, scores = _graded_problem()
        lenient = select_threshold_at_min_sensitivity(y_true, scores, 0.60)
        strict = select_threshold_at_min_sensitivity(y_true, scores, 0.90)
        assert strict.specificity <= lenient.specificity

    def test_unreachable_floor_is_flagged_not_silently_missed(self) -> None:
        """A floor no cutoff can meet must be visible in the strategy name."""
        y_true = np.concatenate([np.ones(10), np.zeros(90)])
        # Positives score below negatives: sensitivity 1.0 requires taking
        # everything, and specificity 0.99 is then unreachable.
        scores = np.concatenate([np.full(10, 0.1), np.full(90, 0.9)])
        choice = select_threshold_at_min_specificity(y_true, scores, 0.99)
        assert choice.strategy.endswith("_unreachable")

    def test_choice_serializes_for_the_report(self) -> None:
        y_true, scores = _graded_problem()
        payload = select_threshold_fbeta(y_true, scores).to_dict()
        assert set(payload) == {
            "strategy", "threshold", "sensitivity", "specificity", "precision", "score"
        }
