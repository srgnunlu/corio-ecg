"""Digitize a single image in an isolated process and save both signal arms.

The CPU digitizer does not return its multi-GB working set to the OS between
calls, so running many digitizations in one process accumulates RSS and gets
OOM-killed. Callers that need several digitizations (e.g. the perturbation
pilot) invoke this worker once per image so the OS reclaims memory on exit.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.digitize import ECGDigitiser  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--calibrated-output", type=Path, required=True)
    parser.add_argument("--layout-hint", type=str, default=None)
    parser.add_argument(
        "--no-retries",
        action="store_true",
        help="Disable orientation/dewarping retries for a single cheap pass.",
    )
    parser.add_argument(
        "--rhythm-output",
        type=Path,
        default=None,
        help="Optional path to save the uncropped full-duration rhythm strip.",
    )
    args = parser.parse_args()

    digitiser = ECGDigitiser(
        enable_dewarping_retry=not args.no_retries,
        enable_orientation_retry=not args.no_retries,
    )
    signals = digitiser.digitize_with_calibrated(args.image, layout_hint=args.layout_hint)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.model_output, signals.model_input)
    args.calibrated_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.calibrated_output, signals.calibrated_millivolts)
    if args.rhythm_output is not None and signals.rhythm_strip is not None:
        args.rhythm_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.rhythm_output, signals.rhythm_strip)


if __name__ == "__main__":
    main()
