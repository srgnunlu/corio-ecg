#!/usr/bin/env python3
# Summarise a consistency sweep: per-arm mean ± sd, and paired deltas by (fold, seed).
# The pairing is the point — a fold/seed cell is the only unit two arms share,
# so the arm comparison must be made within a cell, not across pooled means.

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

CELL_PATTERN = re.compile(r"consistency_f(?P<fold>\d+)_w(?P<weight>[\d.]+)_s(?P<seed>\d+)\.json")
METRICS = ("auroc", "auprc", "separation")


def _load_cells(results_dir: Path) -> dict[str, dict[tuple[int, int], dict[str, float]]]:
    """{weight: {(fold, seed): {baseline_auroc, auroc, auprc, separation}}}."""
    cells: dict[str, dict[tuple[int, int], dict[str, float]]] = defaultdict(dict)
    for path in sorted(results_dir.glob("consistency_f*_w*_s*.json")):
        match = CELL_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        with open(path) as handle:
            report = json.load(handle)
        key = (int(match["fold"]), int(match["seed"]))
        cells[match["weight"]][key] = {
            "baseline_auroc": report["baseline"]["auroc"],
            "baseline_auprc": report["baseline"]["auprc"],
            "baseline_separation": report["median_score_separation"]["before"],
            "auroc": report["trained"]["auroc"],
            "auprc": report["trained"]["auprc"],
            "separation": report["median_score_separation"]["after"],
        }
    return cells


def _mean_sd(values: list[float]) -> str:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return f"{array.mean():.4f}"
    return f"{array.mean():.4f} ±{array.std(ddof=1):.4f}"


def _print_arms(cells: dict[str, dict[tuple[int, int], dict[str, float]]]) -> None:
    """One line per arm, plus the untrained baseline the arms share."""
    any_arm = next(iter(cells.values()))
    print(f"{'arm':<14}{'n':>3}  {'AUROC':<18}{'AUPRC':<18}{'separation':<18}")
    print(
        f"{'baseline':<14}{len(any_arm):>3}  "
        + "".join(
            f"{_mean_sd([c[f'baseline_{m}'] for c in any_arm.values()]):<18}"
            for m in METRICS
        )
    )
    for weight, arm in sorted(cells.items(), key=lambda item: float(item[0])):
        print(
            f"{'w=' + weight:<14}{len(arm):>3}  "
            + "".join(f"{_mean_sd([c[m] for c in arm.values()]):<18}" for m in METRICS)
        )


def _print_paired(
    cells: dict[str, dict[tuple[int, int], dict[str, float]]], reference: str, arm: str
) -> None:
    """Paired deltas arm − reference over the cells both arms completed."""
    shared = sorted(set(cells[reference]) & set(cells[arm]))
    if not shared:
        print(f"\nNo shared (fold, seed) cells between w={reference} and w={arm}.")
        return
    print(f"\nPaired w={arm} − w={reference} over {len(shared)} cells:")
    print(f"  {'fold':>4} {'seed':>9}  {'ΔAUROC':>8} {'ΔAUPRC':>8} {'Δsep':>8}")
    deltas: dict[str, list[float]] = {m: [] for m in METRICS}
    for fold, seed in shared:
        row = []
        for metric in METRICS:
            delta = cells[arm][(fold, seed)][metric] - cells[reference][(fold, seed)][metric]
            deltas[metric].append(delta)
            row.append(f"{delta:>+8.4f}")
        print(f"  {fold:>4} {seed:>9}  " + " ".join(row))
    for metric in METRICS:
        values = np.asarray(deltas[metric])
        line = f"  mean Δ{metric:<11} {values.mean():+.4f}"
        if len(values) >= 2:
            t_p = stats.ttest_rel(
                [cells[arm][k][metric] for k in shared],
                [cells[reference][k][metric] for k in shared],
            ).pvalue
            line += f"  sd {values.std(ddof=1):.4f}  paired-t p={t_p:.3f}"
            if len(values) >= 5 and np.any(values != 0):
                line += f"  wilcoxon p={stats.wilcoxon(values).pvalue:.3f}"
            line += f"  favouring arm {int((values > 0).sum())}/{len(values)}"
        print(line)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Summarise a consistency sweep directory")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--reference", default="0", help="weight of the reference arm")
    parser.add_argument("--arm", default="1.0", help="weight of the arm to compare")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="take the reference arm from another sweep directory, e.g. to pair "
        "an unlabelled-mix sweep against the plain sweep at the same weight",
    )
    args = parser.parse_args()

    cells = _load_cells(args.results_dir)
    if not cells:
        raise SystemExit(f"No consistency_f*_w*_s*.json under {args.results_dir}")
    print(f"Sweep: {args.results_dir}")
    _print_arms(cells)
    if args.reference_dir is not None:
        # Keyed as "<dir>:w=<weight>" so a cross-directory pairing at the same
        # weight does not collide with the arm being compared.
        reference_key = f"{args.reference_dir.name}:w={args.reference}"
        cells[reference_key] = _load_cells(args.reference_dir)[args.reference]
        print(f"\nReference arm from {args.reference_dir}")
        _print_paired(cells, reference_key, args.arm)
    elif args.reference in cells and args.arm in cells:
        _print_paired(cells, args.reference, args.arm)


if __name__ == "__main__":
    main()
