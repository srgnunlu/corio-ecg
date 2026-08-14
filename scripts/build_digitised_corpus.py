#!/usr/bin/env python3
# Phase B: cache (clean, digitised) signal pairs so training can see paper.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.omi.dataset import load_labels, load_raw_signal
from src.pipeline.digitize import ECGDigitiser
from src.utils.ecg_render import DifficultyLevel, render_ecg_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "omi-chongqing"
DEFAULT_FOLDS = PROJECT_ROOT / "configs" / "omi" / "omi_folds_v1.csv"
DEFAULT_WORK_DIR = PROJECT_ROOT / "data" / "processed" / "omi-roundtrip"
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "processed" / "omi-corpus"

MANIFEST_COLUMNS = [
    "ecg_row_record",
    "patient_id",
    "fold",
    "omi",
    "stemi",
    "nstemi",
    "signal_path",
    "layout_name",
    "layout_cost",
    "einthoven_score",
    "detected_leads_count",
]


def _select_records(
    table: pd.DataFrame, folds: list[int], negatives_per_positive: float, seed: int
) -> pd.DataFrame:
    """Every OMI positive in the folds plus a matched negative sample.

    Digitising is the expensive step, so spending it on a natural-prevalence
    draw would buy mostly negatives. Prevalence is restored at training time
    by weighting, not by paying to digitise 15 negatives per positive.
    """
    subset = table[table.fold.isin(folds)]
    positives = subset[subset.OMI == 1]
    negatives = subset[subset.OMI == 0]
    wanted = min(len(negatives), int(round(len(positives) * negatives_per_positive)))
    sampled = negatives.sample(n=wanted, random_state=seed)
    return (
        pd.concat([positives, sampled])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def _portable_path(path: Path) -> str:
    """Project-relative when possible, so the manifest survives a move."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_manifest(path: Path) -> pd.DataFrame:
    """Existing rows, so a killed run resumes instead of restarting."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def main() -> None:
    """CLI entry point for building a digitised training corpus."""
    parser = argparse.ArgumentParser(
        description="Render and digitise OMI records, caching the signals"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--folds-file", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--difficulty",
        choices=[d.value for d in DifficultyLevel],
        default=DifficultyLevel.CLEAN.value,
    )
    parser.add_argument("--negatives-per-positive", type=float, default=1.0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    table = load_labels(args.data_dir, args.folds_file)
    subset = _select_records(table, args.folds, args.negatives_per_positive, args.seed)
    if args.max_records:
        subset = subset.head(args.max_records)

    corpus_root = args.corpus_dir / args.difficulty
    signal_dir = corpus_root / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus_root / "manifest.csv"
    manifest = _load_manifest(manifest_path)
    already_done = set(manifest.ecg_row_record)

    pending = subset[~subset.ecg_row_record.isin(already_done)]
    print(
        f"Corpus {corpus_root}: {len(subset)} requested, {len(already_done)} cached, "
        f"{len(pending)} to build ({int(subset.OMI.sum())} OMI)"
    )
    if pending.empty:
        print("Nothing to do.")
        return

    image_dir = args.work_dir / "images" / args.difficulty
    image_dir.mkdir(parents=True, exist_ok=True)
    digitiser = ECGDigitiser()

    rows: list[dict] = []
    failures: list[str] = []
    for record in tqdm(list(pending.itertuples(index=False)), desc="digitise"):
        stem = record.ecg_row_record.removesuffix(".dat")
        image_path = image_dir / f"{stem}.png"
        signal_path = signal_dir / f"{stem}.npy"

        try:
            if not image_path.exists():
                render_ecg_image(
                    str(args.data_dir / "row_data" / stem),
                    image_path,
                    difficulty=DifficultyLevel(args.difficulty),
                    seed=args.seed,
                )
            digitised = digitiser.digitize(str(image_path))
            # Verify the clean side loads too, so no pair is half-usable later.
            load_raw_signal(args.data_dir, record.ecg_row_record)
        except Exception as error:  # noqa: BLE001 — one bad record must not stop a 5h build
            failures.append(f"{stem}: {type(error).__name__}: {error}")
            continue

        # float16 halves the corpus on disk; the digitiser's own precision is
        # far coarser than that, so nothing measurable is lost.
        np.save(signal_path, digitised.astype(np.float16))
        info = digitiser.last_info
        rows.append(
            {
                "ecg_row_record": record.ecg_row_record,
                "patient_id": record.Patient_id,
                "fold": int(record.fold),
                "omi": int(record.OMI),
                "stemi": int(record.STEMI),
                "nstemi": int(record.NSTEMI),
                "signal_path": _portable_path(signal_path),
                "layout_name": getattr(info, "layout_name", "unknown"),
                "layout_cost": float(getattr(info, "layout_cost", -1.0)),
                "einthoven_score": float(getattr(info, "einthoven_score", -1.0)),
                "detected_leads_count": int(getattr(info, "detected_leads_count", 0)),
            }
        )

        # Flush every 25 records so a crash costs minutes, not hours.
        if len(rows) % 25 == 0:
            pd.concat([manifest, pd.DataFrame(rows, columns=MANIFEST_COLUMNS)]).to_csv(
                manifest_path, index=False
            )

    manifest = pd.concat([manifest, pd.DataFrame(rows, columns=MANIFEST_COLUMNS)])
    manifest.to_csv(manifest_path, index=False)

    if failures:
        print(f"\n[WARN] {len(failures)} records failed:")
        for failure in failures[:10]:
            print(f"  {failure}")

    print(f"\nCorpus now holds {len(manifest)} records ({int(manifest.omi.sum())} OMI)")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
