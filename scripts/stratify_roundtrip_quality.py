#!/usr/bin/env python3
# Phase B step 3: is the paper round-trip loss concentrated in badly digitised
# records, or spread evenly? Only the first case can be fixed by abstaining.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.omi.evaluation import operating_point_metrics, ranking_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCORES = PROJECT_ROOT / "results" / "omi" / "roundtrip_pilot_v3_scores.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "omi" / "roundtrip_quality_v1.json"

# The digitiser's own warning levels, from the Gradio debug panel conventions:
# below 0.7 the limb leads no longer satisfy Einthoven, above 1.0 the layout
# match is unreliable.
EINTHOVEN_WARNING = 0.7
LAYOUT_COST_WARNING = 1.0

DEFAULT_THRESHOLD = 0.6423


def _stratum_row(name: str, subset: pd.DataFrame, threshold: float) -> dict:
    """Round-trip damage inside one slice of records."""
    labels = subset.omi.to_numpy()
    clean = subset.clean_score.to_numpy()
    segment = subset.segment_score.to_numpy()
    absolute_error = np.abs(clean - segment)

    row = {
        "stratum": name,
        "records": int(len(subset)),
        "positives": int(labels.sum()),
        "mean_absolute_score_error": float(absolute_error.mean()),
        "median_absolute_score_error": float(np.median(absolute_error)),
        "mean_signed_shift": float((segment - clean).mean()),
        "score_correlation": (
            float(np.corrcoef(clean, segment)[0, 1]) if len(subset) > 2 else None
        ),
    }
    if 0 < labels.sum() < len(labels):
        row["auroc_clean"] = ranking_metrics(labels, clean)["auroc"]
        row["auroc_segment"] = ranking_metrics(labels, segment)["auroc"]
        row["sensitivity_clean"] = operating_point_metrics(labels, clean, threshold)[
            "sensitivity"
        ]
        row["sensitivity_segment"] = operating_point_metrics(labels, segment, threshold)[
            "sensitivity"
        ]
    else:
        # A single-class slice still reports its score damage, just no AUROC.
        row.update(
            {
                "auroc_clean": None,
                "auroc_segment": None,
                "sensitivity_clean": None,
                "sensitivity_segment": None,
            }
        )
    return row


def _quality_strata(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Slices defined by the digitiser's own diagnostics, plus score quartiles."""
    einthoven_quartiles = pd.qcut(
        table.einthoven_score, 4, labels=["q1_worst", "q2", "q3", "q4_best"], duplicates="drop"
    )
    layout_quartiles = pd.qcut(
        table.layout_cost, 4, labels=["q1_best", "q2", "q3", "q4_worst"], duplicates="drop"
    )
    strata = {
        "all": pd.Series(True, index=table.index),
        f"einthoven_below_{EINTHOVEN_WARNING}": table.einthoven_score < EINTHOVEN_WARNING,
        f"einthoven_at_or_above_{EINTHOVEN_WARNING}": table.einthoven_score
        >= EINTHOVEN_WARNING,
        f"layout_cost_above_{LAYOUT_COST_WARNING}": table.layout_cost > LAYOUT_COST_WARNING,
        f"layout_cost_at_or_below_{LAYOUT_COST_WARNING}": table.layout_cost
        <= LAYOUT_COST_WARNING,
    }
    for label in einthoven_quartiles.cat.categories:
        strata[f"einthoven_{label}"] = einthoven_quartiles == label
    for label in layout_quartiles.cat.categories:
        strata[f"layout_cost_{label}"] = layout_quartiles == label
    return strata


def main() -> None:
    """CLI entry point for quality-stratified round-trip analysis."""
    parser = argparse.ArgumentParser(
        description="Split the OMI round-trip loss by digitisation quality"
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    table = pd.read_csv(args.scores)
    print(f"Loaded {len(table)} records, {int(table.omi.sum())} OMI")

    rows = [
        _stratum_row(name, table[mask], args.threshold)
        for name, mask in _quality_strata(table).items()
        if mask.sum() > 0
    ]

    print("\n=== Round-trip damage by digitisation quality ===")
    print(
        f"  {'STRATUM':<38} {'N':>5} {'|ERR|':>7} {'SHIFT':>7} "
        f"{'CORR':>6} {'AUROC↓':>7} {'SENS↓':>7}"
    )
    for row in rows:
        auroc_drop = (
            row["auroc_segment"] - row["auroc_clean"]
            if row["auroc_clean"] is not None
            else None
        )
        sensitivity_drop = (
            row["sensitivity_segment"] - row["sensitivity_clean"]
            if row["sensitivity_clean"] is not None
            else None
        )
        print(
            f"  {row['stratum']:<38} {row['records']:>5} "
            f"{row['mean_absolute_score_error']:>7.4f} {row['mean_signed_shift']:>+7.4f} "
            f"{row['score_correlation'] if row['score_correlation'] is not None else float('nan'):>6.3f} "
            f"{auroc_drop if auroc_drop is not None else float('nan'):>+7.4f} "
            f"{sensitivity_drop if sensitivity_drop is not None else float('nan'):>+7.4f}"
        )

    # The decisive question for an abstention gate: does a quality signal we can
    # read without ground truth predict how much the score moved?
    absolute_error = np.abs(table.clean_score - table.segment_score).to_numpy()
    predictors = {
        "einthoven_score": table.einthoven_score.to_numpy(),
        "layout_cost": table.layout_cost.to_numpy(),
        "detected_leads_count": table.detected_leads_count.to_numpy(),
        "avg_pixel_per_mm": table.avg_pixel_per_mm.to_numpy(),
    }
    correlations = {}
    print("\n=== Can a quality signal predict the damage? (Spearman vs |score error|) ===")
    for name, values in predictors.items():
        if len(np.unique(values)) < 2:
            correlations[name] = {"rho": None, "p_value": None, "note": "constant"}
            print(f"  {name:<24} constant across the subset — no signal")
            continue
        result = spearmanr(values, absolute_error)
        correlations[name] = {
            "rho": float(result.statistic),
            "p_value": float(result.pvalue),
        }
        print(f"  {name:<24} rho {result.statistic:+.3f}   p {result.pvalue:.4f}")

    report = {
        "scores_file": str(args.scores),
        "records": int(len(table)),
        "positives": int(table.omi.sum()),
        "threshold": args.threshold,
        "einthoven_warning_level": EINTHOVEN_WARNING,
        "layout_cost_warning_level": LAYOUT_COST_WARNING,
        "strata": rows,
        "damage_predictors": correlations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
