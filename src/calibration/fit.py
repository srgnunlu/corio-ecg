# Fits per-class Platt scaling and F1-optimal thresholds on a held-out validation fold.

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

from src.calibration.artifact import CalibrationArtifact, ClassCalibration
from src.calibration.tiers import MINIMUM_CALIBRATION_POSITIVES, assign_tier
from src.utils.ecg_labels import DEFAULT_THRESHOLD

_PROBABILITY_EPSILON = 1e-6

# Keep learned cutoffs away from the extremes. A head whose "best" F1 sits at
# 0.999 is fitting a single record, not an operating point.
MINIMUM_LEARNED_THRESHOLD = 0.02
MAXIMUM_LEARNED_THRESHOLD = 0.98

# On a head with near-chance AUROC, maximising F1 alone degenerates into
# "say yes to everything": recall goes to 1 and F1 settles at 2p/(1+p), which
# can beat any honest cutoff. Such a threshold would fire on nearly every ECG,
# so we only accept operating points that are right more often than this.
MINIMUM_THRESHOLD_PRECISION = 0.30


def to_logit(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to logits, clipped away from 0 and 1."""
    clipped = np.clip(probabilities, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    return np.log(clipped / (1.0 - clipped))


def fit_platt_scaling(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, float] | None:
    """Fit a 1-D logistic regression mapping model logits to true frequencies.

    Args:
        y_true: Binary ground truth, shape (n_samples,).
        y_prob: Raw model probabilities, shape (n_samples,).

    Returns:
        (slope, intercept) for the sigmoid transform, or None when the fit is
        undefined or non-monotonic — see below.
    """
    positives = int(y_true.sum())
    if positives == 0 or positives == len(y_true):
        return None

    logits = to_logit(y_prob).reshape(-1, 1)
    # L2 regularization is what keeps a head with ~10 positives from learning
    # an extreme slope; that shrinkage is the point, not a limitation.
    model = LogisticRegression(solver="lbfgs")
    model.fit(logits, y_true)
    slope, intercept = float(model.coef_[0][0]), float(model.intercept_[0])

    # A non-positive slope means the head ranks worse than chance on this fold,
    # and the fit "fixes" it by inverting the score. That inversion is an
    # artefact of a broken head, not calibration: it would flip AUROC and claim
    # the model is more confident the *less* evidence it sees. Refuse it and
    # leave the head raw, so tiering can demote it on its own AUROC.
    if slope <= 0.0:
        return None
    return slope, intercept


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Pick the probability cutoff that maximises F1 on this fold.

    Args:
        y_true: Binary ground truth, shape (n_samples,).
        y_prob: Probabilities the cutoff will be applied to, shape (n_samples,).

    Returns:
        (threshold, f1_at_threshold). Falls back to the global default when
        no candidate beats an all-negative prediction, or when no cutoff
        reaches MINIMUM_THRESHOLD_PRECISION.
    """
    if y_true.sum() == 0:
        return DEFAULT_THRESHOLD, 0.0

    default_f1 = float(f1_score(y_true, y_prob >= DEFAULT_THRESHOLD, zero_division=0))

    precision, recall, candidates = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one more precision/recall point than
    # thresholds; drop the trailing (precision=1, recall=0) sentinel.
    precision, recall = precision[:-1], recall[:-1]

    denominator = precision + recall
    f1_scores = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    if f1_scores.size == 0:
        return DEFAULT_THRESHOLD, default_f1

    admissible = precision >= MINIMUM_THRESHOLD_PRECISION
    if not admissible.any():
        return DEFAULT_THRESHOLD, default_f1

    best_index = int(np.argmax(np.where(admissible, f1_scores, -1.0)))
    threshold = float(
        np.clip(candidates[best_index], MINIMUM_LEARNED_THRESHOLD, MAXIMUM_LEARNED_THRESHOLD)
    )
    # Clipping can move the cutoff, so score the threshold we will actually ship.
    achieved_f1 = float(f1_score(y_true, y_prob >= threshold, zero_division=0))
    return threshold, achieved_f1


def fit_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    fold: int,
    ground_truth_source: str,
    minimum_positives: int = MINIMUM_CALIBRATION_POSITIVES,
) -> CalibrationArtifact:
    """Learn per-class calibration and thresholds for every diagnosis head.

    Heads with fewer than `minimum_positives` positives are recorded but left
    uncalibrated at the default threshold, and tiered research-only.

    Args:
        y_true: Binary ground truth, shape (n_samples, n_classes).
        y_prob: Raw model probabilities, shape (n_samples, n_classes).
        class_names: Head labels in column order.
        fold: PTB-XL fold the fit was performed on, recorded for provenance.
        ground_truth_source: Human-readable label source, recorded for provenance.
        minimum_positives: Support floor for attempting a fit.

    Returns:
        A populated CalibrationArtifact.
    """
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_true.shape[1] != len(class_names):
        raise ValueError("class_names length must match the number of classes")

    entries: dict[str, ClassCalibration] = {}
    for column, label in enumerate(class_names):
        class_true = y_true[:, column].astype(np.int64)
        class_prob = y_prob[:, column].astype(np.float64)
        positives = int(class_true.sum())

        if positives < minimum_positives:
            entries[label] = ClassCalibration(
                label=label,
                index=column,
                tier=assign_tier(positives, None, None),
                positives=positives,
                threshold=DEFAULT_THRESHOLD,
            )
            continue

        platt = fit_platt_scaling(class_true, class_prob)
        if platt is None:
            calibrated_prob = class_prob
            slope = intercept = None
        else:
            slope, intercept = platt
            calibrated_prob = 1.0 / (1.0 + np.exp(-(slope * to_logit(class_prob) + intercept)))

        threshold, achieved_f1 = select_threshold(class_true, calibrated_prob)
        auroc = float(roc_auc_score(class_true, class_prob))
        baseline_f1 = float(
            f1_score(class_true, class_prob >= DEFAULT_THRESHOLD, zero_division=0)
        )

        entries[label] = ClassCalibration(
            label=label,
            index=column,
            tier=assign_tier(positives, auroc, achieved_f1),
            positives=positives,
            threshold=threshold,
            platt_slope=slope,
            platt_intercept=intercept,
            fit_auroc=auroc,
            fit_f1=achieved_f1,
            fit_f1_at_default_threshold=baseline_f1,
        )

    return CalibrationArtifact(
        fold=fold,
        record_count=int(y_true.shape[0]),
        ground_truth_source=ground_truth_source,
        classes=entries,
    )
