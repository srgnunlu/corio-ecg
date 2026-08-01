# Evaluation metrics for ECG diagnosis and signal quality assessment

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def compute_auroc_per_class(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[int, float]:
    """Compute AUROC for each class that has both positive and negative samples.

    Args:
        y_true: Binary ground truth labels with shape (n_samples, n_classes).
        y_prob: Predicted probabilities with shape (n_samples, n_classes).

    Returns:
        Dictionary mapping class index to its AUROC score.
        Classes with only positive or only negative samples are skipped.
    """
    n_classes = y_true.shape[1]
    auroc_per_class: dict[int, float] = {}

    for i in range(n_classes):
        # Skip classes without both positive and negative samples
        if y_true[:, i].sum() == 0 or y_true[:, i].sum() == len(y_true):
            continue
        try:
            auroc_per_class[i] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            continue

    return auroc_per_class


def compute_macro_auroc(auroc_per_class: dict[int, float]) -> float:
    """Compute macro-averaged AUROC from per-class values.

    Args:
        auroc_per_class: Dictionary mapping class index to AUROC score.

    Returns:
        Mean AUROC across all provided classes, or 0.0 if empty.
    """
    if not auroc_per_class:
        return 0.0
    return float(np.mean(list(auroc_per_class.values())))


def compute_multilabel_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    threshold: float = 0.5,
    minimum_positive_examples: int = 1,
    per_class_thresholds: dict[str, float] | None = None,
) -> dict:
    """Compute ground-truth metrics for eligible multi-label classes.

    A class is eligible only when the evaluation set contains at least one
    positive and one negative example. This avoids undefined AUROC and
    average-precision values on small subsets.

    Args:
        y_true: Binary ground-truth matrix, shape (n_samples, n_classes).
        y_prob: Predicted probabilities, shape (n_samples, n_classes).
        class_names: Human-readable class names in matrix column order.
        threshold: Probability cutoff used for F1 metrics.
        minimum_positive_examples: Minimum positive support required for a
            class to enter summary and per-class metrics.
        per_class_thresholds: Optional per-class cutoffs keyed by class name.
            Classes absent from the mapping fall back to `threshold`.

    Returns:
        JSON-serializable summary and per-class metrics.
    """
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_true.ndim != 2:
        raise ValueError("y_true and y_prob must be two-dimensional")
    if y_true.shape[1] != len(class_names):
        raise ValueError("class_names length must match the number of classes")

    eligible_indices = [
        index
        for index in range(y_true.shape[1])
        if minimum_positive_examples
        <= int(y_true[:, index].sum())
        < y_true.shape[0]
    ]
    if not eligible_indices:
        return {
            "threshold": threshold,
            "minimum_positive_examples": minimum_positive_examples,
            "evaluated_classes": 0,
            "skipped_classes": y_true.shape[1],
            "macro_auroc": 0.0,
            "macro_average_precision": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
        }

    eligible_true = y_true[:, eligible_indices]
    eligible_prob = y_prob[:, eligible_indices]

    effective_thresholds = np.array(
        [
            (per_class_thresholds or {}).get(class_names[source_index], threshold)
            for source_index in eligible_indices
        ],
        dtype=np.float64,
    )
    eligible_pred = eligible_prob >= effective_thresholds

    per_class: dict[str, dict[str, float | int]] = {}
    for local_index, source_index in enumerate(eligible_indices):
        class_true = eligible_true[:, local_index]
        class_prob = eligible_prob[:, local_index]
        class_pred = eligible_pred[:, local_index]
        per_class[class_names[source_index]] = {
            "positives": int(class_true.sum()),
            "negatives": int(len(class_true) - class_true.sum()),
            "threshold": float(effective_thresholds[local_index]),
            "auroc": float(roc_auc_score(class_true, class_prob)),
            "average_precision": float(average_precision_score(class_true, class_prob)),
            "f1": float(f1_score(class_true, class_pred, zero_division=0)),
            "brier": compute_brier_score(class_true, class_prob),
            "ece": compute_expected_calibration_error(class_true, class_prob),
        }

    return {
        "threshold": threshold,
        "uses_per_class_thresholds": bool(per_class_thresholds),
        "minimum_positive_examples": minimum_positive_examples,
        "evaluated_classes": len(eligible_indices),
        "skipped_classes": y_true.shape[1] - len(eligible_indices),
        "macro_auroc": float(np.mean([item["auroc"] for item in per_class.values()])),
        "macro_average_precision": float(
            np.mean([item["average_precision"] for item in per_class.values()])
        ),
        "micro_f1": float(f1_score(eligible_true, eligible_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(eligible_true, eligible_pred, average="macro", zero_division=0)),
        "macro_brier": float(np.mean([item["brier"] for item in per_class.values()])),
        "macro_ece": float(np.mean([item["ece"] for item in per_class.values()])),
        "per_class": per_class,
    }


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes.

    Lower is better. Unlike AUROC this is sensitive to miscalibration, so a
    model can rank well (high AUROC) yet score poorly here.

    Args:
        y_true: Binary ground truth, shape (n_samples,).
        y_prob: Predicted probabilities in [0, 1], shape (n_samples,).

    Returns:
        Brier score, or 0.0 for an empty input.
    """
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error over equal-width probability bins.

    Each bin contributes the gap between its mean predicted probability and
    its observed positive rate, weighted by how many samples fall in it.

    Args:
        y_true: Binary ground truth, shape (n_samples,).
        y_prob: Predicted probabilities in [0, 1], shape (n_samples,).
        n_bins: Number of equal-width bins spanning [0, 1].

    Returns:
        ECE in [0, 1], or 0.0 for an empty input.
    """
    if y_true.size == 0:
        return 0.0

    # np.digitize with right=True puts p=0.0 in bin 0 and p=1.0 in the last bin.
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)

    total_error = 0.0
    for bin_index in range(n_bins):
        in_bin = bin_indices == bin_index
        bin_size = int(in_bin.sum())
        if bin_size == 0:
            continue
        confidence = float(np.mean(y_prob[in_bin]))
        accuracy = float(np.mean(y_true[in_bin]))
        total_error += (bin_size / y_true.size) * abs(confidence - accuracy)

    return float(total_error)


def compute_snr(clean: np.ndarray, digitized: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB between clean and digitized signals.

    SNR = 10 * log10(power_signal / power_noise)
    where noise = digitized - clean

    Args:
        clean: Reference signal, shape (12, 5000).
        digitized: Digitized signal, shape (12, 5000).

    Returns:
        SNR in decibels. Higher is better.
        Returns float('inf') if noise power is zero (identical signals).
    """
    noise = digitized - clean
    power_signal = np.mean(clean ** 2)
    power_noise = np.mean(noise ** 2)

    # Avoid division by zero when signals are identical
    if power_noise == 0.0:
        return float("inf")

    return float(10.0 * np.log10(power_signal / power_noise))


def compute_pearson_per_lead(
    clean: np.ndarray, digitized: np.ndarray
) -> list[float]:
    """Compute Pearson correlation coefficient for each of 12 leads.

    Args:
        clean: Reference signal, shape (12, 5000).
        digitized: Digitized signal, shape (12, 5000).

    Returns:
        List of 12 Pearson correlation values, one per lead.
    """
    correlations: list[float] = []
    for lead_idx in range(clean.shape[0]):
        clean_lead = clean[lead_idx]
        digitized_lead = digitized[lead_idx]
        finite = np.isfinite(clean_lead) & np.isfinite(digitized_lead)
        if int(finite.sum()) < 2:
            correlations.append(0.0)
            continue

        clean_valid = clean_lead[finite]
        digitized_valid = digitized_lead[finite]
        # Skip constant and near-constant leads — pearsonr is undefined.
        if np.ptp(clean_valid) < 1e-12 or np.ptp(digitized_valid) < 1e-12:
            correlations.append(0.0)
            continue

        r, _ = pearsonr(clean_valid, digitized_valid)
        # Guard against NaN from near-constant signals
        correlations.append(0.0 if np.isnan(r) else float(r))
    return correlations
