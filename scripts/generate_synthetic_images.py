# Batch synthetic ECG image generator — renders PTB-XL test fold records
# as paper ECG photographs at configurable difficulty levels (clean, moderate, hard).

from __future__ import annotations

import argparse
import ast
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.ecg_render import DifficultyLevel, render_ecg_image

DEFAULT_DATA_DIR = Path("data/raw/ptb-xl")
DEFAULT_OUTPUT_DIR = Path("data/processed/images")
ALL_DIFFICULTIES = [d.value for d in DifficultyLevel]


def load_test_fold(data_dir: Path, max_samples: int | None = None) -> pd.DataFrame:
    """Load PTB-XL metadata and filter to test fold (strat_fold == 10).

    Args:
        data_dir: Path to PTB-XL dataset root directory.
        max_samples: Optional limit on number of records.

    Returns:
        DataFrame with test fold records.
    """
    csv_path = data_dir / "ptbxl_database.csv"
    df = pd.read_csv(csv_path, index_col="ecg_id")
    test_records = df[df.strat_fold == 10]

    if max_samples is not None:
        test_records = test_records.head(max_samples)

    return test_records


def generate_images(
    data_dir: Path,
    output_dir: Path,
    difficulties: list[str],
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, int]:
    """Generate synthetic ECG images for each record and difficulty level.

    Args:
        data_dir: Path to PTB-XL dataset root directory.
        output_dir: Base output directory for generated images.
        difficulties: List of difficulty level names to generate.
        max_samples: Optional limit on number of records.
        seed: Random seed for reproducible rendering.

    Returns:
        Summary dict with counts per difficulty: generated, skipped, failed.
    """
    test_records = load_test_fold(data_dir, max_samples)
    total_records = len(test_records)

    # Parse difficulty names into enum values
    difficulty_levels = [DifficultyLevel(d) for d in difficulties]

    summary: dict[str, int] = {
        "total_records": total_records,
        "total_tasks": total_records * len(difficulty_levels),
        "generated": 0,
        "skipped": 0,
        "failed": 0,
    }

    print(f"Records: {total_records} | Difficulties: {difficulties}")
    print(f"Output: {output_dir.resolve()}\n")

    for difficulty in difficulty_levels:
        difficulty_dir = output_dir / difficulty.value
        difficulty_dir.mkdir(parents=True, exist_ok=True)

        description = f"Rendering [{difficulty.value}]"
        progress = tqdm(
            test_records.iterrows(),
            total=total_records,
            desc=description,
        )

        for ecg_id, row in progress:
            output_path = difficulty_dir / f"{ecg_id}.png"

            # Skip already-generated images for resume capability
            if output_path.exists():
                summary["skipped"] += 1
                continue

            try:
                # Build record path — remove .hea extension if present
                record_path = str(data_dir / row.filename_hr)
                if record_path.endswith(".hea"):
                    record_path = record_path[:-4]

                render_ecg_image(
                    record_path=record_path,
                    output_path=output_path,
                    difficulty=difficulty,
                    seed=seed,
                )
                summary["generated"] += 1

            except Exception as error:
                summary["failed"] += 1
                tqdm.write(f"  FAIL ecg_id={ecg_id}: {error}")

    return summary


def print_summary(summary: dict[str, int], elapsed: float) -> None:
    """Print a human-readable summary of the generation run.

    Args:
        summary: Counts from generate_images().
        elapsed: Total wall-clock time in seconds.
    """
    minutes = elapsed / 60
    print("\n" + "=" * 50)
    print("GENERATION SUMMARY")
    print("=" * 50)
    print(f"Total records:    {summary['total_records']}")
    print(f"Total tasks:      {summary['total_tasks']}")
    print(f"Generated (new):  {summary['generated']}")
    print(f"Skipped (exist):  {summary['skipped']}")
    print(f"Failed:           {summary['failed']}")
    print(f"Time elapsed:     {minutes:.1f} min ({elapsed:.0f} sec)")

    if summary["generated"] > 0:
        per_image = elapsed / summary["generated"]
        print(f"Avg per image:    {per_image:.1f} sec")

    print("=" * 50)


def main() -> None:
    """CLI entry point for batch synthetic ECG image generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic paper ECG images from PTB-XL test fold"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path to PTB-XL dataset (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=ALL_DIFFICULTIES,
        default=None,
        help="Generate only one difficulty level (default: all three)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of records to process (default: all ~2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible rendering (default: 42)",
    )

    args = parser.parse_args()

    # Determine which difficulties to generate
    if args.difficulty:
        difficulties = [args.difficulty]
    else:
        difficulties = ALL_DIFFICULTIES

    start_time = time.time()

    summary = generate_images(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        difficulties=difficulties,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    elapsed = time.time() - start_time
    print_summary(summary, elapsed)


if __name__ == "__main__":
    main()
