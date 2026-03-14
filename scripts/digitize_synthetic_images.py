# Batch digitization of synthetic ECG images — converts PNG images back to
# (12, 5000) numpy signals using ECGDigitiser, with resume and error handling.

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.pipeline.digitize import ECGDigitiser

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_DIR = Path("data/processed/images")
DEFAULT_SIGNAL_DIR = Path("data/processed/signals")
VALID_LEVELS = ["clean", "moderate", "hard"]
DEFAULT_LAYOUT_HINT = "3x4+1R"


def collect_image_paths(
    image_dir: Path,
    level: str,
    max_samples: int | None = None,
) -> list[Path]:
    """Collect PNG image paths for a given difficulty level.

    Args:
        image_dir: Base directory containing level subdirectories.
        level: Difficulty level (clean, moderate, hard).
        max_samples: Optional limit on number of images.

    Returns:
        Sorted list of PNG file paths.

    Raises:
        FileNotFoundError: If the level directory does not exist.
    """
    level_dir = image_dir / level
    if not level_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {level_dir}\n"
            f"Run generate_synthetic_images.py first."
        )

    paths = sorted(level_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {level_dir}")

    if max_samples is not None:
        paths = paths[:max_samples]

    return paths


def digitize_batch(
    digitiser: ECGDigitiser,
    image_paths: list[Path],
    output_dir: Path,
    level: str,
    layout_hint: str | None = DEFAULT_LAYOUT_HINT,
) -> dict[str, int]:
    """Digitize a batch of ECG images and save as .npy files.

    Args:
        digitiser: Initialized ECGDigitiser instance (model loaded once).
        image_paths: List of PNG image paths to process.
        output_dir: Base output directory for signal files.
        level: Difficulty level name (used for subdirectory).
        layout_hint: Optional layout constraint for the digitizer.

    Returns:
        Summary dict with counts: digitized, skipped, failed.
    """
    level_output = output_dir / level
    level_output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {
        "total": len(image_paths),
        "digitized": 0,
        "skipped": 0,
        "failed": 0,
    }

    progress = tqdm(
        image_paths,
        desc=f"Digitizing [{level}]",
        unit="img",
    )

    for image_path in progress:
        ecg_id = image_path.stem
        output_path = level_output / f"{ecg_id}.npy"

        # Skip already-digitized files for resume capability
        if output_path.exists():
            summary["skipped"] += 1
            continue

        try:
            signal = digitiser.digitize(image_path, layout_hint=layout_hint)
            np.save(output_path, signal)
            summary["digitized"] += 1
        except Exception as error:
            summary["failed"] += 1
            tqdm.write(f"  FAIL {ecg_id}: {error}")
            logger.warning("Failed to digitize %s: %s", ecg_id, error)

    return summary


def print_summary(summary: dict[str, int], elapsed: float) -> None:
    """Print a human-readable summary of the digitization run.

    Args:
        summary: Counts from digitize_batch().
        elapsed: Total wall-clock time in seconds.
    """
    minutes = elapsed / 60
    print("\n" + "=" * 50)
    print("DIGITIZATION SUMMARY")
    print("=" * 50)
    print(f"Total images:     {summary['total']}")
    print(f"Digitized (new):  {summary['digitized']}")
    print(f"Skipped (exist):  {summary['skipped']}")
    print(f"Failed:           {summary['failed']}")
    print(f"Time elapsed:     {minutes:.1f} min ({elapsed:.0f} sec)")

    if summary["digitized"] > 0:
        per_image = elapsed / summary["digitized"]
        print(f"Avg per image:    {per_image:.1f} sec")

    print("=" * 50)


def main() -> None:
    """CLI entry point for batch digitization of synthetic ECG images."""
    parser = argparse.ArgumentParser(
        description="Digitize synthetic ECG images to numpy signal arrays"
    )
    parser.add_argument(
        "--level",
        type=str,
        required=True,
        choices=VALID_LEVELS,
        help="Difficulty level to digitize (clean, moderate, hard)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Base directory for input images (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SIGNAL_DIR,
        help=f"Base directory for output signals (default: {DEFAULT_SIGNAL_DIR})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of images to process (default: all)",
    )
    parser.add_argument(
        "--layout-hint",
        type=str,
        default=DEFAULT_LAYOUT_HINT,
        help=(
            "Layout substring to constrain the digitizer "
            f"(default: {DEFAULT_LAYOUT_HINT}). Use 'auto' to disable."
        ),
    )

    args = parser.parse_args()

    # Collect images before loading the model (fail fast if no images)
    image_paths = collect_image_paths(args.image_dir, args.level, args.max_samples)
    print(f"Found {len(image_paths)} images for level '{args.level}'")
    print(f"Output: {(args.output_dir / args.level).resolve()}\n")

    # Load model once, reuse for all images
    print("Loading ECGDigitiser model...")
    digitiser = ECGDigitiser()

    start_time = time.time()

    summary = digitize_batch(
        digitiser=digitiser,
        image_paths=image_paths,
        output_dir=args.output_dir,
        level=args.level,
        layout_hint=None if args.layout_hint.lower() == "auto" else args.layout_hint,
    )

    elapsed = time.time() - start_time
    print_summary(summary, elapsed)


if __name__ == "__main__":
    main()
