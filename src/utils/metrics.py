# Evaluation metrics for ECG diagnosis and signal quality assessment

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score


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
        # Skip constant leads (e.g. zero-padded) — pearsonr is undefined
        if np.std(clean[lead_idx]) == 0.0 or np.std(digitized[lead_idx]) == 0.0:
            correlations.append(0.0)
            continue
        r, _ = pearsonr(clean[lead_idx], digitized[lead_idx])
        # Guard against NaN from near-constant signals
        correlations.append(0.0 if np.isnan(r) else float(r))
    return correlations
