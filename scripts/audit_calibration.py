#!/usr/bin/env python3
# Single-shot audit of the fold-9 calibration artefact on the held-out PTB-XL test fold.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.calibration.artifact import CalibrationArtifact
from src.calibration.ptbxl_data import load_fold_predictions
from src.calibration.tiers import (
    TIER_ORDER,
    TIER_PROVISIONAL,
    TIER_RESEARCH_ONLY,
    TIER_VALIDATED,
)
from src.training.evaluate import DEFAULT_DATA_DIR, DEFAULT_TEST_FOLD
from src.utils.ecg_labels import DEFAULT_THRESHOLD, ECG_FOUNDER_LABELS
from src.utils.metrics import compute_multilabel_classification_metrics

DEFAULT_METRICS_DIR = Path("results/metrics")
DEFAULT_PREFIX = "ptbxl_baseline"
DEFAULT_ARTIFACT_PATH = Path("configs/calibration/ptbxl_fold9_v1.json")
DEFAULT_OUTPUT_PATH = Path("results/metrics/calibration_audit_fold10.json")
DEFAULT_MINIMUM_POSITIVES = 5


def _scenario_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    minimum_positives: int,
    per_class_thresholds: dict[str, float] | None = None,
) -> dict:
    """Score one probability/threshold configuration on the audit fold."""
    return compute_multilabel_classification_metrics(
        y_true,
        y_prob,
        class_names=ECG_FOUNDER_LABELS,
        threshold=DEFAULT_THRESHOLD,
        minimum_positive_examples=minimum_positives,
        per_class_thresholds=per_class_thresholds,
    )


def _tier_breakdown(scenario: dict, artifact: CalibrationArtifact) -> dict:
    """Aggregate per-class audit metrics by the evidence tier assigned on fold 9."""
    breakdown: dict[str, dict] = {}
    for tier in TIER_ORDER:
        members = {
            label: values
            for label, values in scenario["per_class"].items()
            if artifact.tier_for(label) == tier
        }
        if not members:
            breakdown[tier] = {"classes": 0}
            continue
        breakdown[tier] = {
            "classes": len(members),
            "macro_auroc": float(np.mean([item["auroc"] for item in members.values()])),
            "macro_f1": float(np.mean([item["f1"] for item in members.values()])),
            "macro_brier": float(np.mean([item["brier"] for item in members.values()])),
            "macro_ece": float(np.mean([item["ece"] for item in members.values()])),
            "labels": sorted(members),
        }
    return breakdown


def _print_scenario(name: str, scenario: dict) -> None:
    """Print the headline metrics for one scenario."""
    print(
        f"  {name:<34} AUROC={scenario['macro_auroc']:.4f}  "
        f"AP={scenario['macro_average_precision']:.4f}  "
        f"microF1={scenario['micro_f1']:.4f}  macroF1={scenario['macro_f1']:.4f}  "
        f"Brier={scenario['macro_brier']:.4f}  ECE={scenario['macro_ece']:.4f}"
    )


def main() -> None:
    """CLI entry point for the held-out calibration audit."""
    parser = argparse.ArgumentParser(
        description="Audit fold-9 calibration on the held-out PTB-XL test fold"
    )
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--minimum-positive-examples", type=int, default=DEFAULT_MINIMUM_POSITIVES
    )
    args = parser.parse_args()

    artifact = CalibrationArtifact.load(args.artifact)
    if artifact.fold == DEFAULT_TEST_FOLD:
        raise SystemExit(
            "Artefact was fitted on the audit fold — the audit would be circular."
        )

    y_true, raw_prob, provenance = load_fold_predictions(
        metrics_dir=args.metrics_dir,
        prefix=args.prefix,
        ptbxl_database_path=args.data_dir / "ptbxl_database.csv",
        labels_path=args.data_dir / "ecgfounder_ptbxl_label.csv",
    )
    calibrated_prob = artifact.apply(raw_prob, ECG_FOUNDER_LABELS)

    print(
        f"Audit fold {DEFAULT_TEST_FOLD}: {provenance['records_matched']} labelled records; "
        f"calibration fitted on fold {artifact.fold} ({artifact.record_count} records)\n"
    )

    scenarios = {
        "baseline_raw_global_threshold": _scenario_metrics(
            y_true, raw_prob, args.minimum_positive_examples
        ),
        "calibrated_global_threshold": _scenario_metrics(
            y_true, calibrated_prob, args.minimum_positive_examples
        ),
        "calibrated_per_class_threshold": _scenario_metrics(
            y_true,
            calibrated_prob,
            args.minimum_positive_examples,
            per_class_thresholds=artifact.threshold_map(),
        ),
    }

    print("Scenarios:")
    _print_scenario("1. raw + global 0.5", scenarios["baseline_raw_global_threshold"])
    _print_scenario("2. calibrated + global 0.5", scenarios["calibrated_global_threshold"])
    _print_scenario(
        "3. calibrated + per-class thr", scenarios["calibrated_per_class_threshold"]
    )

    final = scenarios["calibrated_per_class_threshold"]
    baseline = scenarios["baseline_raw_global_threshold"]
    breakdown = _tier_breakdown(final, artifact)

    print("\nBy evidence tier (scenario 3):")
    for tier in TIER_ORDER:
        entry = breakdown[tier]
        if not entry["classes"]:
            print(f"  {tier:<14} no audit-eligible classes")
            continue
        print(
            f"  {tier:<14} classes={entry['classes']:<3} "
            f"AUROC={entry['macro_auroc']:.4f}  macroF1={entry['macro_f1']:.4f}  "
            f"Brier={entry['macro_brier']:.4f}  ECE={entry['macro_ece']:.4f}"
        )

    print("\nLargest per-class F1 gains (scenario 3 vs scenario 1):")
    gains = [
        (label, final["per_class"][label]["f1"] - values["f1"], values["f1"],
         final["per_class"][label]["f1"], artifact.tier_for(label))
        for label, values in baseline["per_class"].items()
        if label in final["per_class"]
    ]
    for label, delta, before, after, tier in sorted(gains, key=lambda item: -item[1])[:12]:
        print(f"  {label[:44]:<44} {before:.3f} → {after:.3f}  ({delta:+.3f}, {tier})")

    regressions = [item for item in gains if item[1] < -0.01]
    print(f"\nClasses whose F1 regressed by >0.01: {len(regressions)}")
    for label, delta, before, after, tier in sorted(regressions, key=lambda item: item[1])[:10]:
        print(f"  {label[:44]:<44} {before:.3f} → {after:.3f}  ({delta:+.3f}, {tier})")

    report = {
        "audit_fold": DEFAULT_TEST_FOLD,
        "calibration_fold": artifact.fold,
        "provenance": provenance,
        "artifact_summary": artifact.summary(),
        "minimum_positive_examples": args.minimum_positive_examples,
        "scenarios": scenarios,
        "tier_breakdown": breakdown,
        "headline_deltas": {
            "macro_auroc": final["macro_auroc"] - baseline["macro_auroc"],
            "macro_average_precision": (
                final["macro_average_precision"] - baseline["macro_average_precision"]
            ),
            "micro_f1": final["micro_f1"] - baseline["micro_f1"],
            "macro_f1": final["macro_f1"] - baseline["macro_f1"],
            "macro_brier": final["macro_brier"] - baseline["macro_brier"],
            "macro_ece": final["macro_ece"] - baseline["macro_ece"],
        },
        "regressed_classes": [
            {"label": label, "before": before, "after": after, "delta": delta, "tier": tier}
            for label, delta, before, after, tier in regressions
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved audit report to {args.output}")


if __name__ == "__main__":
    main()
