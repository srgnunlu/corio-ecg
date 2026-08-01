# Threshold selection for OMI, where a missed occlusion costs more than a false alarm.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# F1 weights recall and precision equally. For OMI that is the wrong trade:
# a missed occlusion delays reperfusion, a false alarm costs a second look.
# beta=2 weights recall twice as heavily as precision.
DEFAULT_BETA = 2.0

# The published baseline's operating point, used for iso-sensitivity and
# iso-specificity comparisons.
BASELINE_SENSITIVITY = 0.697
BASELINE_SPECIFICITY = 0.873

# Guard against degenerate cutoffs, mirroring src/calibration/fit.py.
MINIMUM_THRESHOLD = 0.001
MAXIMUM_THRESHOLD = 0.999


@dataclass(frozen=True)
class ThresholdChoice:
    """A selected cutoff and the operating point it produced on the fitting set."""

    strategy: str
    threshold: float
    sensitivity: float
    specificity: float
    precision: float
    score: float

    def to_dict(self) -> dict:
        """Serialize for the experiment report."""
        return {
            "strategy": self.strategy,
            "threshold": self.threshold,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "score": self.score,
        }


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Midpoints between observed scores, clipped away from the extremes."""
    unique = np.unique(scores)
    if len(unique) < 2:
        return np.array([0.5])
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    return np.clip(midpoints, MINIMUM_THRESHOLD, MAXIMUM_THRESHOLD)


def _rates(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float, float]:
    """Return (sensitivity, specificity, precision) at one cutoff."""
    predicted = scores >= threshold
    positives = y_true == 1
    true_positive = float(np.sum(predicted & positives))
    false_positive = float(np.sum(predicted & ~positives))
    positive_count = float(positives.sum())
    negative_count = float((~positives).sum())

    sensitivity = true_positive / positive_count if positive_count else 0.0
    specificity = (negative_count - false_positive) / negative_count if negative_count else 0.0
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    return sensitivity, specificity, precision


def fbeta_score_at(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, beta: float = DEFAULT_BETA
) -> float:
    """F-beta at one cutoff; beta>1 favours recall."""
    sensitivity, _, precision = _rates(y_true, scores, threshold)
    denominator = (beta**2 * precision) + sensitivity
    if denominator == 0:
        return 0.0
    return float((1 + beta**2) * precision * sensitivity / denominator)


def select_threshold_fbeta(
    y_true: np.ndarray, scores: np.ndarray, beta: float = DEFAULT_BETA
) -> ThresholdChoice:
    """Pick the cutoff maximising F-beta on the fitting set."""
    candidates = _candidate_thresholds(scores)
    scores_by_candidate = [fbeta_score_at(y_true, scores, t, beta) for t in candidates]
    best = int(np.argmax(scores_by_candidate))
    threshold = float(candidates[best])
    sensitivity, specificity, precision = _rates(y_true, scores, threshold)
    return ThresholdChoice(
        strategy=f"f{beta:g}_max",
        threshold=threshold,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        score=float(scores_by_candidate[best]),
    )


def select_threshold_at_min_sensitivity(
    y_true: np.ndarray, scores: np.ndarray, minimum_sensitivity: float
) -> ThresholdChoice:
    """Maximise specificity subject to a sensitivity floor.

    This is the comparison a clinician can act on: at the same detection rate as
    the published baseline, how many fewer false alarms does the model raise?
    """
    candidates = _candidate_thresholds(scores)
    best_choice: ThresholdChoice | None = None
    for threshold in candidates:
        sensitivity, specificity, precision = _rates(y_true, scores, float(threshold))
        if sensitivity < minimum_sensitivity:
            continue
        if best_choice is None or specificity > best_choice.specificity:
            best_choice = ThresholdChoice(
                strategy=f"max_specificity_at_sens_{minimum_sensitivity:g}",
                threshold=float(threshold),
                sensitivity=sensitivity,
                specificity=specificity,
                precision=precision,
                score=specificity,
            )
    if best_choice is None:
        # No cutoff reaches the floor; fall back to the most sensitive one.
        threshold = float(candidates.min())
        sensitivity, specificity, precision = _rates(y_true, scores, threshold)
        return ThresholdChoice(
            strategy=f"max_specificity_at_sens_{minimum_sensitivity:g}_unreachable",
            threshold=threshold,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            score=specificity,
        )
    return best_choice


def select_threshold_at_min_specificity(
    y_true: np.ndarray, scores: np.ndarray, minimum_specificity: float
) -> ThresholdChoice:
    """Maximise sensitivity subject to a specificity floor.

    The mirror-image comparison: holding the false-alarm burden at the
    baseline's level, how many more occlusions does the model catch?
    """
    candidates = _candidate_thresholds(scores)
    best_choice: ThresholdChoice | None = None
    for threshold in candidates:
        sensitivity, specificity, precision = _rates(y_true, scores, float(threshold))
        if specificity < minimum_specificity:
            continue
        if best_choice is None or sensitivity > best_choice.sensitivity:
            best_choice = ThresholdChoice(
                strategy=f"max_sensitivity_at_spec_{minimum_specificity:g}",
                threshold=float(threshold),
                sensitivity=sensitivity,
                specificity=specificity,
                precision=precision,
                score=sensitivity,
            )
    if best_choice is None:
        threshold = float(candidates.max())
        sensitivity, specificity, precision = _rates(y_true, scores, threshold)
        return ThresholdChoice(
            strategy=f"max_sensitivity_at_spec_{minimum_specificity:g}_unreachable",
            threshold=threshold,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            score=sensitivity,
        )
    return best_choice
