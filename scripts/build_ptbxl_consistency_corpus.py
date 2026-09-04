#!/usr/bin/env python3
# Phase B step 6: cache (clean, digitised) pairs from PTB-XL, unlabelled.
# The consistency term needs no diagnosis — only the same recording seen twice —
# so an open dataset with no OMI labels is a valid way to scale it.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.pipeline.digitize import ECGDigitiser
from src.utils.ecg_render import DifficultyLevel, render_ecg_image
from src.utils.wfdb_helpers import read_ecg_signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ptb-xl"
DEFAULT_WORK_DIR = PROJECT_ROOT / "data" / "processed" / "ptbxl-roundtrip"
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "processed" / "ptbxl-corpus"

MANIFEST_COLUMNS = [
    "ecg_id",
    "patient_id",
    "record_path",
    "signal_path",
    "layout_name",
    "layout_cost",
    "einthoven_score",
    "detected_leads_count",
]


def _select_records(table: pd.DataFrame, max_records: int | None, seed: int) -> pd.DataFrame:
    """One recording per patient, shuffled.

    PTB-XL holds several recordings for some patients. Digitising is the
    expensive step, so spending it on near-duplicates of the same heart buys
    less variety than the same budget spread across distinct patients.
    """
    unique = table.sample(frac=1.0, random_state=seed).drop_duplicates("patient_id")
    return unique.head(max_records) if max_records else unique


def _load_manifest(path: Path) -> pd.DataFrame:
    """Existing rows, so a killed run resumes instead of restarting."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def _portable_path(path: Path) -> str:
    """Project-relative when possible, so the manifest survives a move."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    """CLI entry point for building an unlabelled PTB-XL consistency corpus."""
    parser = argparse.ArgumentParser(
        description="Render and digitise PTB-XL records, caching the signals"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--difficulty",
        choices=[d.value for d in DifficultyLevel],
        default=DifficultyLevel.CLEAN.value,
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    table = pd.read_csv(args.data_dir / "ptbxl_database.csv")
    subset = _select_records(table, args.max_records, args.seed)

    corpus_root = args.corpus_dir / args.difficulty
    signal_dir = corpus_root / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus_root / "manifest.csv"
    manifest = _load_manifest(manifest_path)
    already_done = set(manifest.ecg_id)

    pending = subset[~subset.ecg_id.isin(already_done)]
    print(
        f"Corpus {corpus_root}: {len(subset)} requested, {len(already_done)} cached, "
        f"{len(pending)} to build"
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
        # filename_hr is the 500 Hz variant, matching ECGFounder's sample rate.
        record_path = args.data_dir / record.filename_hr
        stem = record_path.name
        image_path = image_dir / f"{stem}.png"
        signal_path = signal_dir / f"{stem}.npy"

        try:
            if not image_path.exists():
                render_ecg_image(
                    str(record_path),
                    image_path,
                    difficulty=DifficultyLevel(args.difficulty),
                    seed=args.seed,
                )
            digitised = digitiser.digitize(str(image_path))
            # Verify the clean side reads too, so no pair is half-usable later.
            read_ecg_signal(str(record_path))
        except Exception as error:  # noqa: BLE001 — one bad record must not stop a long build
            failures.append(f"{stem}: {type(error).__name__}: {error}")
            continue

        np.save(signal_path, digitised.astype(np.float16))
        info = digitiser.last_info
        rows.append(
            {
                "ecg_id": int(record.ecg_id),
                "patient_id": int(record.patient_id),
                "record_path": _portable_path(record_path),
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

    print(f"\nCorpus now holds {len(manifest)} records")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
