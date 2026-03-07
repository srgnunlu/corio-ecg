# Evaluation metrics for ECG diagnosis — computes per-class and macro AUROC

import numpy as np
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
