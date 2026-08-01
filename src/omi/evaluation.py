# Protocol metrics for OMI: operating-point scores, patient-level bootstrap CIs, subgroups.

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_BOOTSTRAP_ROUNDS = 2000
DEFAULT_ALPHA = 0.05

# Time_Interval is in minutes; 12 hours is the pre-registered cut for
# "the ECG plausibly reflects the angiographic finding".
SHORT_INTERVAL_MINUTES = 12 * 60


def operating_point_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    """Confusion-derived metrics at one cutoff, matching the platform's outputs."""
    predicted = scores >= threshold
    true_positive = int(np.sum(predicted & (y_true == 1)))
    false_positive = int(np.sum(predicted & (y_true == 0)))
    true_negative = int(np.sum(~predicted & (y_true == 0)))
    false_negative = int(np.sum(~predicted & (y_true == 1)))

    def _ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    sensitivity = _ratio(true_positive, true_positive + false_negative)
    precision = _ratio(true_positive, true_positive + false_positive)
    return {
        "sensitivity": sensitivity,
        "specificity": _ratio(true_negative, true_negative + false_positive),
        "ppv": precision,
        "npv": _ratio(true_negative, true_negative + false_negative),
        "accuracy": _ratio(true_positive + true_negative, len(y_true)),
        "f1": _ratio(2 * precision * sensitivity, precision + sensitivity)
        if (precision + sensitivity)
        else 0.0,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """AUROC and average precision, or None when only one class is present."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


def patient_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    patient_ids: np.ndarray,
    metric: callable,
    rounds: int = DEFAULT_BOOTSTRAP_ROUNDS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 20260801,
) -> dict:
    """Percentile bootstrap CI, resampling patients rather than recordings.

    Some patients contribute several ECGs; resampling records would treat those
    as independent and understate the interval.

    Args:
        y_true: Binary labels.
        scores: Model scores.
        patient_ids: Patient identifier per record.
        metric: Callable (y_true, scores) -> float.
        rounds: Bootstrap replicates.
        alpha: Two-sided error rate; 0.05 gives a 95% interval.
        seed: Fixed for reproducibility.

    Returns:
        Point estimate plus lower and upper bounds.
    """
    rng = np.random.default_rng(seed)
    unique_patients, patient_index = np.unique(patient_ids, return_inverse=True)
    rows_by_patient = [np.flatnonzero(patient_index == i) for i in range(len(unique_patients))]

    estimates: list[float] = []
    for _ in range(rounds):
        drawn = rng.integers(0, len(unique_patients), size=len(unique_patients))
        rows = np.concatenate([rows_by_patient[i] for i in drawn])
        sampled_true = y_true[rows]
        if sampled_true.sum() == 0 or sampled_true.sum() == len(sampled_true):
            continue
        estimates.append(metric(sampled_true, scores[rows]))

    if not estimates:
        return {"point": None, "lower": None, "upper": None, "rounds": 0}

    return {
        "point": float(metric(y_true, scores)),
        "lower": float(np.percentile(estimates, 100 * alpha / 2)),
        "upper": float(np.percentile(estimates, 100 * (1 - alpha / 2))),
        "rounds": len(estimates),
    }


def build_subgroup_masks(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pre-registered subgroups from the Gate 2 protocol.

    Defined once here so no subgroup can be added after seeing results.
    """
    is_stemi = table.STEMI == 1
    is_nstemi = table.NSTEMI == 1
    return {
        "overall": np.ones(len(table), dtype=bool),
        "acs_positive": (is_stemi | is_nstemi).to_numpy(),
        "stemi_labelled": is_stemi.to_numpy(),
        "nstemi_labelled": is_nstemi.to_numpy(),
        "interval_le_12h": (table.Time_Interval <= SHORT_INTERVAL_MINUTES).to_numpy(),
        "interval_gt_12h": (table.Time_Interval > SHORT_INTERVAL_MINUTES).to_numpy(),
        "cto": (table.CTO == 1).to_numpy(),
        "paced": (table.Paced == 1).to_numpy(),
        "vf_vt": (table.VF_VT == 1).to_numpy(),
        "prior_pci": (table.Prior_PCI == 1).to_numpy(),
        "age_lt_65": (table.age < 65).to_numpy(),
        "age_ge_65": (table.age >= 65).to_numpy(),
        "female": (table.gender == 0).to_numpy(),
        "male": (table.gender == 1).to_numpy(),
    }


def subgroup_report(
    table: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """Score every pre-registered subgroup at one cutoff."""
    report: dict[str, dict] = {}
    for name, mask in build_subgroup_masks(table).items():
        subset_true = y_true[mask]
        entry: dict = {"records": int(mask.sum()), "omi": int(subset_true.sum())}
        if len(subset_true) and 0 < subset_true.sum() < len(subset_true):
            entry.update(ranking_metrics(subset_true, scores[mask]))
            entry.update(operating_point_metrics(subset_true, scores[mask], threshold))
        report[name] = entry
    return report
