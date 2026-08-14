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


def prevalence_weight(y_true: np.ndarray, target_prevalence: float) -> float:
    """How much each negative must count for a subset to behave like the clinic.

    An enriched subset inflates precision, and every precision-based criterion
    with it — F-beta included. Weighting negatives up to the rate the model will
    actually meet keeps the selected cutoff transferable.

    Args:
        y_true: Binary labels of the subset in hand.
        target_prevalence: Positive rate the cutoff will be deployed at.

    Returns:
        Multiplier for false-positive counts; 1.0 when the subset already
        matches the target rate.
    """
    if not 0.0 < target_prevalence < 1.0:
        raise ValueError(f"target prevalence {target_prevalence} must lie in (0, 1)")
    positives = float(np.sum(y_true == 1))
    negatives = float(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("both classes must be present to reweight")
    target_negatives = positives * (1.0 - target_prevalence) / target_prevalence
    return float(target_negatives / negatives)


def operating_rates(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    negative_weight: float = 1.0,
) -> tuple[float, float, float]:
    """Return (sensitivity, specificity, precision) at one cutoff."""
    return _rates_from_prediction(y_true, scores >= threshold, negative_weight)


def fbeta_score_at(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    beta: float = DEFAULT_BETA,
    negative_weight: float = 1.0,
) -> float:
    """F-beta at one cutoff; beta>1 favours recall."""
    sensitivity, _, precision = operating_rates(y_true, scores, threshold, negative_weight)
    denominator = (beta**2 * precision) + sensitivity
    if denominator == 0:
        return 0.0
    return float((1 + beta**2) * precision * sensitivity / denominator)


def select_threshold_fbeta(
    y_true: np.ndarray,
    scores: np.ndarray,
    beta: float = DEFAULT_BETA,
    negative_weight: float = 1.0,
) -> ThresholdChoice:
    """Pick the cutoff maximising F-beta on the fitting set."""
    candidates = _candidate_thresholds(scores)
    scores_by_candidate = [
        fbeta_score_at(y_true, scores, t, beta, negative_weight) for t in candidates
    ]
    best = int(np.argmax(scores_by_candidate))
    threshold = float(candidates[best])
    sensitivity, specificity, precision = operating_rates(
        y_true, scores, threshold, negative_weight
    )
    return ThresholdChoice(
        strategy=f"f{beta:g}_max",
        threshold=threshold,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        score=float(scores_by_candidate[best]),
    )


@dataclass(frozen=True)
class CrossFitThreshold:
    """A cutoff plus the operating point it reached on records it never saw."""

    strategy: str
    threshold: float
    sensitivity: float
    specificity: float
    precision: float
    fold_thresholds: list[float]
    repeats: int
    folds: int

    def to_dict(self) -> dict:
        """Serialize for the experiment report."""
        return {
            "strategy": self.strategy,
            "threshold": self.threshold,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "fold_thresholds": self.fold_thresholds,
            "repeats": self.repeats,
            "folds": self.folds,
        }


def select_threshold_crossfit(
    y_true: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    beta: float = DEFAULT_BETA,
    folds: int = 2,
    repeats: int = 5,
    seed: int = 20260802,
    negative_weight: float = 1.0,
) -> CrossFitThreshold:
    """Select a cutoff and estimate what it achieves on unseen records.

    Picking the cutoff that maximises F-beta on a set and then reporting that
    same set's sensitivity is circular — the number is an upper bound, not a
    forecast. Here each block of patients is scored by a cutoff fitted on the
    other blocks, so every reported rate is out-of-fit. Splitting by patient
    rather than by record keeps the same person from sitting on both sides.

    Args:
        y_true: Binary labels.
        scores: Model scores.
        groups: Patient identifier per record.
        beta: F-beta weight used inside each fit; beta>1 favours recall.
        folds: Blocks per repeat.
        repeats: Re-splits, to average away one unlucky partition.
        seed: Fixed for reproducibility.
        negative_weight: False-positive multiplier from `prevalence_weight`,
            for subsets enriched above the deployment prevalence.

    Returns:
        The deployable cutoff (median across fits) and its pooled out-of-fit
        sensitivity, specificity and precision averaged over the repeats.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < folds:
        raise ValueError(f"{len(unique_groups)} patients cannot fill {folds} folds")

    rng = np.random.default_rng(seed)
    fold_thresholds: list[float] = []
    pooled_rates: list[tuple[float, float, float]] = []

    for _ in range(repeats):
        held_out_prediction = np.zeros(len(y_true), dtype=bool)
        shuffled = rng.permutation(unique_groups)
        for block in np.array_split(shuffled, folds):
            in_block = np.isin(groups, block)
            fitting = ~in_block
            # A fitting half with one class has no meaningful F-beta curve.
            if y_true[fitting].sum() in (0, int(fitting.sum())):
                continue
            choice = select_threshold_fbeta(
                y_true[fitting], scores[fitting], beta, negative_weight
            )
            fold_thresholds.append(choice.threshold)
            held_out_prediction[in_block] = scores[in_block] >= choice.threshold
        pooled_rates.append(
            _rates_from_prediction(y_true, held_out_prediction, negative_weight)
        )

    if not fold_thresholds:
        raise ValueError("no fold produced a usable threshold — check the labels")

    sensitivity, specificity, precision = (
        float(np.mean([rate[index] for rate in pooled_rates])) for index in range(3)
    )
    return CrossFitThreshold(
        strategy=f"f{beta:g}_max_crossfit",
        threshold=float(np.median(fold_thresholds)),
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        fold_thresholds=[float(t) for t in fold_thresholds],
        repeats=repeats,
        folds=folds,
    )


def _rates_from_prediction(
    y_true: np.ndarray, predicted: np.ndarray, negative_weight: float = 1.0
) -> tuple[float, float, float]:
    """Sensitivity, specificity and precision from a boolean prediction vector.

    `negative_weight` scales false positives, which moves precision to the
    prevalence the cutoff will be deployed at. Sensitivity and specificity are
    ratios within one class, so the weight leaves them untouched.
    """
    positives = y_true == 1
    true_positive = float(np.sum(predicted & positives))
    false_positive = float(np.sum(predicted & ~positives))
    positive_count = float(positives.sum())
    negative_count = float((~positives).sum())

    sensitivity = true_positive / positive_count if positive_count else 0.0
    specificity = (negative_count - false_positive) / negative_count if negative_count else 0.0
    weighted_false_positive = false_positive * negative_weight
    precision = (
        true_positive / (true_positive + weighted_false_positive)
        if (true_positive + weighted_false_positive)
        else 0.0
    )
    return sensitivity, specificity, precision


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
        sensitivity, specificity, precision = operating_rates(y_true, scores, float(threshold))
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
        sensitivity, specificity, precision = operating_rates(y_true, scores, threshold)
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
        sensitivity, specificity, precision = operating_rates(y_true, scores, float(threshold))
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
        sensitivity, specificity, precision = operating_rates(y_true, scores, threshold)
        return ThresholdChoice(
            strategy=f"max_sensitivity_at_spec_{minimum_specificity:g}_unreachable",
            threshold=threshold,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            score=sensitivity,
        )
    return best_choice
