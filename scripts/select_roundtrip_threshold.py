#!/usr/bin/env python3
# Phase B step 2: re-select the OMI cutoff on the digitised score distribution.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.omi.evaluation import operating_point_metrics, patient_bootstrap_ci, ranking_metrics
from src.omi.threshold import (
    operating_rates,
    prevalence_weight,
    select_threshold_at_min_specificity,
    select_threshold_crossfit,
    select_threshold_fbeta,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCORES = PROJECT_ROOT / "results" / "omi" / "roundtrip_pilot_v3_scores.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "omi" / "roundtrip_threshold_v1.json"

# The cutoff Gate 2 selected on clean validation signals and the pilot inherited.
INHERITED_THRESHOLD = 0.6423

# The pilot subset is balanced; the clinic is not. Selecting on F-beta without
# correcting for that would pick a cutoff tuned to a 50% OMI ward.
NATURAL_PREVALENCE = 0.064

ARMS = {
    "clean": "clean_score",
    "digitised": "digitised_score",
    "digitised_segment": "segment_score",
}


def _alert_burden(
    sensitivity: float, specificity: float, prevalence: float, cohort: int = 1000
) -> dict:
    """What a cutoff costs a department, per `cohort` patients screened.

    Gate 3 asks for the false-alert burden at the real prevalence, not at the
    enriched subset's — at 6.4% most alerts come from the 93.6%.
    """
    detected = prevalence * sensitivity * cohort
    missed = prevalence * (1.0 - sensitivity) * cohort
    false_alerts = (1.0 - prevalence) * (1.0 - specificity) * cohort
    return {
        "cohort": cohort,
        "detected_omi": detected,
        "missed_omi": missed,
        "false_alerts": false_alerts,
        "false_alerts_per_detected_omi": float(false_alerts / detected) if detected else None,
    }


def _arm_report(
    labels: np.ndarray,
    scores: np.ndarray,
    patients: np.ndarray,
    inherited: float,
    beta: float,
    repeats: int,
    seed: int,
    negative_weight: float,
) -> dict:
    """Everything we want to know about one arm's operating point."""
    crossfit = select_threshold_crossfit(
        labels,
        scores,
        patients,
        beta=beta,
        repeats=repeats,
        seed=seed,
        negative_weight=negative_weight,
    )
    in_sample = select_threshold_fbeta(labels, scores, beta=beta, negative_weight=negative_weight)
    _, _, inherited_precision = operating_rates(labels, scores, inherited, negative_weight)
    return {
        "ranking": ranking_metrics(labels, scores),
        "at_inherited_threshold": {
            # `ppv` is the enriched subset's own value; the corrected one is what
            # a 6.4%-prevalence clinic would see.
            **operating_point_metrics(labels, scores, inherited),
            "precision_at_target_prevalence": inherited_precision,
        },
        "crossfit": crossfit.to_dict(),
        "at_crossfit_threshold": operating_point_metrics(labels, scores, crossfit.threshold),
        "in_sample_optimum": in_sample.to_dict(),
        "median_score_positive": float(np.median(scores[labels == 1])),
        "median_score_negative": float(np.median(scores[labels == 0])),
    }


def _print_arm(name: str, report: dict, inherited_threshold: float) -> None:
    """One block per arm, inherited cutoff versus the re-selected one."""
    inherited = report["at_inherited_threshold"]
    crossfit = report["crossfit"]
    in_sample = report["in_sample_optimum"]
    print(f"\n  {name}  (AUROC {report['ranking']['auroc']:.4f})")
    print(f"    {'':<26} {'CUTOFF':>8} {'SENS':>7} {'SPEC':>7} {'PPV*':>7}")
    print(
        f"    {'inherited (Gate 2)':<26} {inherited_threshold:>8.4f} "
        f"{inherited['sensitivity']:>7.4f} {inherited['specificity']:>7.4f} "
        f"{inherited['precision_at_target_prevalence']:>7.4f}"
    )
    print(
        f"    {'re-selected (out-of-fit)':<26} {crossfit['threshold']:>8.4f} "
        f"{crossfit['sensitivity']:>7.4f} {crossfit['specificity']:>7.4f} {crossfit['precision']:>7.4f}"
    )
    print(
        f"    {'in-sample optimum (bound)':<26} {in_sample['threshold']:>8.4f} "
        f"{in_sample['sensitivity']:>7.4f} {in_sample['specificity']:>7.4f} {in_sample['precision']:>7.4f}"
    )
    print(
        f"    median score  positives {report['median_score_positive']:.4f}"
        f"   negatives {report['median_score_negative']:.4f}"
    )


def main() -> None:
    """CLI entry point for digitised-distribution threshold re-selection."""
    parser = argparse.ArgumentParser(
        description="Re-select the OMI cutoff on digitised scores, out-of-fit"
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inherited-threshold", type=float, default=INHERITED_THRESHOLD)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--target-prevalence", type=float, default=NATURAL_PREVALENCE)
    args = parser.parse_args()

    table = pd.read_csv(args.scores)
    labels = table.omi.to_numpy()
    patients = table.patient_id.to_numpy()
    negative_weight = prevalence_weight(labels, args.target_prevalence)
    print(
        f"Loaded {len(table)} records, {int(labels.sum())} OMI, "
        f"{len(np.unique(patients))} patients"
    )
    print(
        f"Subset prevalence {labels.mean():.1%} → each negative counts "
        f"{negative_weight:.2f}x to reach the deployment rate {args.target_prevalence:.1%}"
    )

    arms = {
        name: _arm_report(
            labels,
            table[column].to_numpy(),
            patients,
            args.inherited_threshold,
            args.beta,
            args.repeats,
            args.seed,
            negative_weight,
        )
        for name, column in ARMS.items()
    }

    print(f"\n=== Threshold re-selection (F{args.beta:g}, patient-grouped cross-fit) ===")
    print(f"    PPV* = precision at {args.target_prevalence:.1%} prevalence, not the subset's")
    for name, report in arms.items():
        _print_arm(name, report, args.inherited_threshold)

    # The pilot's headline loss was sensitivity at a fixed cutoff. Split that
    # loss into the part a new cutoff recovers and the part that survives it.
    clean_sensitivity = arms["clean"]["at_inherited_threshold"]["sensitivity"]
    segment_inherited = arms["digitised_segment"]["at_inherited_threshold"]["sensitivity"]
    segment_reselected = arms["digitised_segment"]["crossfit"]["sensitivity"]
    gap = clean_sensitivity - segment_inherited
    recovered = segment_reselected - segment_inherited
    accounting = {
        "clean_sensitivity_at_inherited": clean_sensitivity,
        "segment_sensitivity_at_inherited": segment_inherited,
        "segment_sensitivity_reselected": segment_reselected,
        "sensitivity_gap": gap,
        "recovered_by_rethresholding": recovered,
        "recovered_fraction": float(recovered / gap) if gap else 0.0,
        "specificity_paid": (
            arms["digitised_segment"]["at_inherited_threshold"]["specificity"]
            - arms["digitised_segment"]["crossfit"]["specificity"]
        ),
        # Re-selection helps the clean arm too. Giving both arms the same
        # procedure is the only comparison that isolates the round-trip itself.
        "matched_procedure": {
            "clean_sensitivity": arms["clean"]["crossfit"]["sensitivity"],
            "clean_specificity": arms["clean"]["crossfit"]["specificity"],
            "segment_sensitivity": arms["digitised_segment"]["crossfit"]["sensitivity"],
            "segment_specificity": arms["digitised_segment"]["crossfit"]["specificity"],
            "sensitivity_gap": (
                arms["clean"]["crossfit"]["sensitivity"]
                - arms["digitised_segment"]["crossfit"]["sensitivity"]
            ),
        },
    }

    # Holding false alarms at the clean arm's level isolates ranking loss from
    # cutoff placement: this is what the digitised arm can do at equal burden.
    iso = select_threshold_at_min_specificity(
        labels,
        table.segment_score.to_numpy(),
        arms["clean"]["at_inherited_threshold"]["specificity"],
    ).to_dict()

    bootstrap = {
        "sensitivity": patient_bootstrap_ci(
            labels,
            table.segment_score.to_numpy(),
            patients,
            lambda y, s: operating_point_metrics(
                y, s, arms["digitised_segment"]["crossfit"]["threshold"]
            )["sensitivity"],
            seed=args.seed,
        ),
        "specificity": patient_bootstrap_ci(
            labels,
            table.segment_score.to_numpy(),
            patients,
            lambda y, s: operating_point_metrics(
                y, s, arms["digitised_segment"]["crossfit"]["threshold"]
            )["specificity"],
            seed=args.seed,
        ),
    }

    segment = arms["digitised_segment"]
    burden = {
        "inherited": _alert_burden(
            segment["at_inherited_threshold"]["sensitivity"],
            segment["at_inherited_threshold"]["specificity"],
            args.target_prevalence,
        ),
        "reselected": _alert_burden(
            segment["crossfit"]["sensitivity"],
            segment["crossfit"]["specificity"],
            args.target_prevalence,
        ),
    }

    print("\n=== Sensitivity accounting (segment-ensemble arm) ===")
    print(f"  clean @ inherited cutoff        {clean_sensitivity:.4f}")
    print(f"  digitised @ inherited cutoff    {segment_inherited:.4f}   (gap {gap:+.4f})")
    print(
        f"  digitised @ re-selected cutoff  {segment_reselected:.4f}   "
        f"(recovers {accounting['recovered_fraction']:.0%} of the gap, "
        f"costs {accounting['specificity_paid']:.4f} specificity)"
    )
    print(
        f"  at the clean arm's specificity ({iso['specificity']:.4f}): "
        f"sensitivity {iso['sensitivity']:.4f}"
    )
    matched = accounting["matched_procedure"]
    print(
        f"  same procedure both arms:       clean {matched['clean_sensitivity']:.4f} "
        f"(spec {matched['clean_specificity']:.4f}) vs digitised "
        f"{matched['segment_sensitivity']:.4f} (spec {matched['segment_specificity']:.4f}) "
        f"→ gap {matched['sensitivity_gap']:+.4f}"
    )
    print(
        f"  bootstrap 95% CI at the re-selected cutoff: sens "
        f"[{bootstrap['sensitivity']['lower']:.4f}, {bootstrap['sensitivity']['upper']:.4f}]"
    )

    print(f"\n=== Alert burden per 1000 screened at {args.target_prevalence:.1%} prevalence ===")
    print(f"  {'CUTOFF':<14} {'DETECTED':>9} {'MISSED':>7} {'FALSE ALERTS':>13} {'FA/DETECTED':>12}")
    for name, values in burden.items():
        print(
            f"  {name:<14} {values['detected_omi']:>9.1f} {values['missed_omi']:>7.1f} "
            f"{values['false_alerts']:>13.1f} {values['false_alerts_per_detected_omi']:>12.1f}"
        )

    report = {
        "scores_file": str(args.scores),
        "records": int(len(table)),
        "positives": int(labels.sum()),
        "patients": int(len(np.unique(patients))),
        "prevalence_note": "balanced subset — not the natural 6.4% rate",
        "beta": args.beta,
        "target_prevalence": args.target_prevalence,
        "negative_weight": negative_weight,
        "inherited_threshold": args.inherited_threshold,
        "crossfit_repeats": args.repeats,
        "arms": arms,
        "sensitivity_accounting": accounting,
        "iso_specificity_segment": iso,
        "bootstrap_at_reselected_threshold": bootstrap,
        "alert_burden_segment": burden,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
