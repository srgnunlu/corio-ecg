# Tests for per-class calibration: Platt fitting, threshold selection, tiering and the artefact.

import numpy as np
import pytest

from src.calibration.artifact import (
    ARTIFACT_VERSION,
    CalibrationArtifact,
    ClassCalibration,
)
from src.calibration.fit import (
    MAXIMUM_LEARNED_THRESHOLD,
    MINIMUM_LEARNED_THRESHOLD,
    fit_calibration,
    fit_platt_scaling,
    select_threshold,
    to_logit,
)
from src.calibration.tiers import (
    MINIMUM_CALIBRATION_POSITIVES,
    TIER_PROVISIONAL,
    TIER_RESEARCH_ONLY,
    TIER_VALIDATED,
    assign_tier,
)
from src.utils.ecg_labels import DEFAULT_THRESHOLD

RNG = np.random.default_rng(7)


def _separable_scores(n_positive: int, n_negative: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a well-separated binary problem with overconfident probabilities."""
    positives = np.clip(RNG.normal(0.85, 0.05, n_positive), 0.01, 0.99)
    negatives = np.clip(RNG.normal(0.15, 0.05, n_negative), 0.01, 0.99)
    y_prob = np.concatenate([positives, negatives])
    y_true = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
    return y_true, y_prob


class TestToLogit:
    """Tests for the logit helper."""

    def test_round_trip(self) -> None:
        """Sigmoid of the logit should recover the original probability."""
        probabilities = np.array([0.1, 0.5, 0.9])
        recovered = 1.0 / (1.0 + np.exp(-to_logit(probabilities)))
        np.testing.assert_allclose(recovered, probabilities, rtol=1e-6)

    def test_saturated_inputs_stay_finite(self) -> None:
        """Probabilities of exactly 0 and 1 must not produce infinities."""
        assert np.all(np.isfinite(to_logit(np.array([0.0, 1.0]))))


class TestFitPlattScaling:
    """Tests for the per-class Platt fit."""

    def test_returns_positive_slope_for_informative_head(self) -> None:
        """A head that ranks correctly should get a positive slope."""
        y_true, y_prob = _separable_scores(60, 240)
        result = fit_platt_scaling(y_true, y_prob)
        assert result is not None
        slope, _ = result
        assert slope > 0

    def test_rejects_inverted_head(self) -> None:
        """A head that ranks worse than chance must not be silently inverted."""
        y_true, y_prob = _separable_scores(60, 240)
        # Flip the labels so high probabilities now mark negatives.
        assert fit_platt_scaling(1 - y_true, y_prob) is None

    def test_returns_none_for_single_outcome_class(self) -> None:
        """An all-negative column has no fit to make."""
        assert fit_platt_scaling(np.zeros(50), RNG.random(50)) is None

    def test_calibration_improves_brier_on_overconfident_head(self) -> None:
        """Fitted parameters should pull inflated probabilities toward the truth."""
        y_true, y_prob = _separable_scores(40, 360)
        slope, intercept = fit_platt_scaling(y_true, y_prob)
        calibrated = 1.0 / (1.0 + np.exp(-(slope * to_logit(y_prob) + intercept)))
        assert np.mean((calibrated - y_true) ** 2) < np.mean((y_prob - y_true) ** 2)


class TestSelectThreshold:
    """Tests for F1-optimal threshold selection."""

    def test_finds_separating_threshold(self) -> None:
        """A separable head should get a cutoff between the two clusters."""
        y_true, y_prob = _separable_scores(50, 200)
        threshold, f1 = select_threshold(y_true, y_prob)
        assert 0.2 < threshold < 0.8
        assert f1 > 0.9

    def test_threshold_stays_within_bounds(self) -> None:
        """Learned cutoffs must never reach the extremes."""
        y_true = np.concatenate([np.ones(5), np.zeros(495)])
        y_prob = np.concatenate([np.full(5, 0.999), np.full(495, 0.001)])
        threshold, _ = select_threshold(y_true, y_prob)
        assert MINIMUM_LEARNED_THRESHOLD <= threshold <= MAXIMUM_LEARNED_THRESHOLD

    def test_all_negative_column_falls_back_to_default(self) -> None:
        """With no positives there is nothing to optimise."""
        threshold, f1 = select_threshold(np.zeros(100), RNG.random(100))
        assert threshold == DEFAULT_THRESHOLD
        assert f1 == 0.0

    def test_rejects_degenerate_predict_everything_threshold(self) -> None:
        """A near-chance head must not get a cutoff that fires on every record.

        With 5% prevalence and uninformative scores, no cutoff can reach the
        precision floor, so the head keeps the default threshold instead of
        collapsing onto the "say yes to everything" operating point.
        """
        rng = np.random.default_rng(3)
        y_true = np.concatenate([np.ones(30), np.zeros(570)])
        y_prob = rng.uniform(0.05, 0.95, 600)
        threshold, _ = select_threshold(y_true, y_prob)
        assert threshold == DEFAULT_THRESHOLD

    def test_keeps_low_cutoff_when_precision_holds(self) -> None:
        """A genuinely separable head may still earn a cutoff well below 0.5."""
        rng = np.random.default_rng(11)
        y_true = np.concatenate([np.ones(40), np.zeros(360)])
        y_prob = np.concatenate([
            rng.uniform(0.15, 0.30, 40),
            rng.uniform(0.00, 0.05, 360),
        ])
        threshold, f1 = select_threshold(y_true, y_prob)
        assert threshold < 0.2
        assert f1 > 0.9

    def test_reported_f1_matches_reported_threshold(self) -> None:
        """The returned F1 must be the score of the cutoff actually shipped."""
        from sklearn.metrics import f1_score

        y_true, y_prob = _separable_scores(30, 170)
        threshold, reported_f1 = select_threshold(y_true, y_prob)
        assert reported_f1 == pytest.approx(
            f1_score(y_true, y_prob >= threshold, zero_division=0)
        )


class TestAssignTier:
    """Tests for evidence tiering rules."""

    def test_strong_head_is_validated(self) -> None:
        assert assign_tier(200, 0.95, 0.70) == TIER_VALIDATED

    def test_weak_discrimination_is_provisional(self) -> None:
        assert assign_tier(200, 0.65, 0.40) == TIER_PROVISIONAL

    def test_weak_operating_point_is_provisional(self) -> None:
        assert assign_tier(200, 0.95, 0.10) == TIER_PROVISIONAL

    def test_low_support_is_research_only(self) -> None:
        """Below the support floor a head cannot be measured, however good it looks."""
        assert assign_tier(MINIMUM_CALIBRATION_POSITIVES - 1, 0.99, 0.99) == TIER_RESEARCH_ONLY

    def test_missing_metrics_is_research_only(self) -> None:
        assert assign_tier(500, None, None) == TIER_RESEARCH_ONLY


class TestCalibrationArtifact:
    """Tests for artefact behaviour and serialization."""

    def _artifact(self) -> CalibrationArtifact:
        return CalibrationArtifact(
            fold=9,
            record_count=2183,
            ground_truth_source="test",
            classes={
                "STRONG": ClassCalibration(
                    label="STRONG",
                    index=0,
                    tier=TIER_VALIDATED,
                    positives=150,
                    threshold=0.2,
                    platt_slope=1.5,
                    platt_intercept=-0.5,
                    fit_auroc=0.95,
                    fit_f1=0.7,
                ),
                "RAW": ClassCalibration(
                    label="RAW",
                    index=1,
                    tier=TIER_RESEARCH_ONLY,
                    positives=3,
                    threshold=DEFAULT_THRESHOLD,
                ),
            },
        )

    def test_threshold_and_tier_lookup(self) -> None:
        artifact = self._artifact()
        assert artifact.threshold_for("STRONG") == 0.2
        assert artifact.tier_for("STRONG") == TIER_VALIDATED

    def test_unknown_label_falls_back_to_default_and_research_only(self) -> None:
        """A head with no recorded state was never measured."""
        artifact = self._artifact()
        assert artifact.threshold_for("MISSING") == DEFAULT_THRESHOLD
        assert artifact.tier_for("MISSING") == TIER_RESEARCH_ONLY

    def test_apply_leaves_uncalibrated_head_untouched(self) -> None:
        artifact = self._artifact()
        probabilities = np.array([[0.3, 0.3]])
        calibrated = artifact.apply(probabilities, ["STRONG", "RAW"])
        assert calibrated[0, 1] == pytest.approx(0.3)
        assert calibrated[0, 0] != pytest.approx(0.3)

    def test_apply_is_monotonic(self) -> None:
        """Calibration must preserve ranking, so AUROC cannot move."""
        artifact = self._artifact()
        ascending = np.linspace(0.01, 0.99, 50).reshape(-1, 1)
        calibrated = artifact.apply(
            np.hstack([ascending, ascending]), ["STRONG", "RAW"]
        )
        assert np.all(np.diff(calibrated[:, 0]) > 0)

    def test_save_load_round_trip(self, tmp_path) -> None:
        artifact = self._artifact()
        path = tmp_path / "calibration.json"
        artifact.save(path)
        loaded = CalibrationArtifact.load(path)
        assert loaded.fold == artifact.fold
        assert loaded.classes["STRONG"].platt_slope == pytest.approx(1.5)
        assert loaded.classes["RAW"].is_calibrated is False

    def test_rejects_unknown_version(self, tmp_path) -> None:
        payload = self._artifact().to_dict()
        payload["version"] = ARTIFACT_VERSION + 1
        with pytest.raises(ValueError, match="Unsupported calibration artefact version"):
            CalibrationArtifact.from_dict(payload)


class TestFitCalibration:
    """Tests for the end-to-end fit over all heads."""

    def test_low_support_head_is_left_raw(self) -> None:
        """Heads under the support floor keep the default threshold and no fit."""
        y_true = np.zeros((300, 2), dtype=np.int64)
        y_true[:120, 0] = 1
        y_true[:2, 1] = 1
        y_prob = np.column_stack(
            [
                np.concatenate([RNG.normal(0.8, 0.05, 120), RNG.normal(0.2, 0.05, 180)]),
                RNG.random(300),
            ]
        ).clip(0.01, 0.99)

        artifact = fit_calibration(
            y_true, y_prob, ["STRONG", "RARE"], fold=9, ground_truth_source="test"
        )
        assert artifact.classes["RARE"].tier == TIER_RESEARCH_ONLY
        assert artifact.classes["RARE"].is_calibrated is False
        assert artifact.classes["RARE"].threshold == DEFAULT_THRESHOLD
        assert artifact.classes["STRONG"].tier == TIER_VALIDATED
        assert artifact.classes["STRONG"].is_calibrated is True

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            fit_calibration(
                np.zeros((10, 2)), np.zeros((10, 3)), ["A", "B"], 9, "test"
            )

    def test_rejects_mismatched_class_names(self) -> None:
        with pytest.raises(ValueError, match="class_names length"):
            fit_calibration(
                np.zeros((10, 2)), np.zeros((10, 2)), ["A"], 9, "test"
            )
