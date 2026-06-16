# Batch digitization of synthetic ECG images — converts PNG images back to
# (12, 5000) numpy signals using ECGDigitiser, with resume and error handling.

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.pipeline.digitize import (
    MIN_REQUIRED_NONZERO_LEADS,
    NUM_LEADS,
    TARGET_LENGTH,
    DigitizeInfo,
    ECGDigitiser,
)

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_DIR = Path("data/processed/images")
DEFAULT_SIGNAL_DIR = Path("data/processed/signals")
VALID_LEVELS = ["clean", "moderate", "hard"]
DEFAULT_LAYOUT_HINT = "3x4+1R"
DIGITIZATION_PIPELINE_VERSION = 4


# Bulky DigitizeInfo arrays kept only for in-memory reprojection probes —
# they are huge (H×W ink maps, full-width pixel traces) and have no place in
# a per-record audit file, so we drop them before serializing.
_NON_AUDIT_DIAGNOSTIC_FIELDS: tuple[str, ...] = ("signal_probability", "raw_lines")


def _json_default(value: object) -> object:
    """Convert vendor/PyTorch diagnostic values into JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _audit_diagnostics(info: DigitizeInfo) -> dict:
    """Return DigitizeInfo as a JSON-friendly dict without bulky array fields."""
    diagnostics = asdict(info)
    for field_name in _NON_AUDIT_DIAGNOSTIC_FIELDS:
        diagnostics.pop(field_name, None)
    return diagnostics


def _metadata_path(signal_path: Path) -> Path:
    """Return the audit metadata path paired with a saved signal."""
    return signal_path.with_suffix(".json")


def _is_current_saved_signal(signal_path: Path) -> bool:
    """Return whether an existing signal is usable and from this pipeline version."""
    metadata_path = _metadata_path(signal_path)
    if not signal_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        signal = np.load(signal_path, mmap_mode="r")
    except (OSError, ValueError, json.JSONDecodeError):
        return False

    if metadata.get("pipeline_version") != DIGITIZATION_PIPELINE_VERSION:
        return False
    if signal.shape != (NUM_LEADS, TARGET_LENGTH) or not np.isfinite(signal).all():
        return False
    active_leads = int(np.sum(np.max(np.abs(signal), axis=1) > 5e-2))
    return active_leads >= MIN_REQUIRED_NONZERO_LEADS


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

    paths = sorted(
        level_dir.glob("*.png"),
        key=lambda path: (
            0,
            int(path.stem),
        )
        if path.stem.isdigit()
        else (1, path.stem),
    )
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
    overwrite: bool = False,
    max_new: int | None = None,
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
        "invalidated": 0,
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

        # Resume only from audited outputs produced by the current pipeline.
        if not overwrite and _is_current_saved_signal(output_path):
            summary["skipped"] += 1
            continue
        if max_new is not None and summary["digitized"] + summary["failed"] >= max_new:
            break
        if output_path.exists():
            summary["invalidated"] += 1
            output_path.unlink()
        _metadata_path(output_path).unlink(missing_ok=True)

        try:
            signal = digitiser.digitize(image_path, layout_hint=layout_hint)
            np.save(output_path, signal)
            metadata = {
                "pipeline_version": DIGITIZATION_PIPELINE_VERSION,
                "source_image": str(image_path),
                "layout_hint": layout_hint,
                "dewarping_retry_enabled": digitiser.enable_dewarping_retry,
                "orientation_retry_enabled": digitiser.enable_orientation_retry,
                "diagnostics": _audit_diagnostics(digitiser.last_info),
            }
            _metadata_path(output_path).write_text(
                json.dumps(metadata, indent=2, default=_json_default)
            )
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
    print(f"Invalidated old:  {summary['invalidated']}")
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-digitize all selected images even when audited outputs exist",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Process at most this many non-reusable images before exiting",
    )
    parser.add_argument(
        "--ecg-id",
        type=str,
        default=None,
        help="Process only one image stem/ecg_id",
    )

    args = parser.parse_args()

    # Collect images before loading the model (fail fast if no images)
    image_paths = collect_image_paths(args.image_dir, args.level, args.max_samples)
    if args.ecg_id is not None:
        image_paths = [path for path in image_paths if path.stem == args.ecg_id]
        if not image_paths:
            raise FileNotFoundError(
                f"No image with ecg_id={args.ecg_id} found for level {args.level}"
            )
    print(f"Found {len(image_paths)} images for level '{args.level}'")
    print(f"Output: {(args.output_dir / args.level).resolve()}\n")

    # Load model once, reuse for all images
    print("Loading ECGDigitiser model...")
    # Synthetic benchmark images have controlled flat geometry. Dewarping retry
    # adds no value here and can hang on a few otherwise usable records.
    digitiser = ECGDigitiser(
        enable_dewarping_retry=False,
        enable_orientation_retry=False,
    )

    start_time = time.time()

    summary = digitize_batch(
        digitiser=digitiser,
        image_paths=image_paths,
        output_dir=args.output_dir,
        level=args.level,
        layout_hint=None if args.layout_hint.lower() == "auto" else args.layout_hint,
        overwrite=args.overwrite,
        max_new=args.max_new,
    )

    elapsed = time.time() - start_time
    print_summary(summary, elapsed)


if __name__ == "__main__":
    main()
