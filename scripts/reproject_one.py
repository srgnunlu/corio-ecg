"""Digitize one image in an isolated process and print re-projection fidelity.

Mirrors digitize_one.py's isolation rationale (the CPU digitizer does not return
its multi-GB working set between calls). Outputs a JSON line with the image-space
overlay scores so an orchestrator can collect them across many records without
holding several digitizations in one process.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.digitize import ECGDigitiser  # noqa: E402
from src.quality.reprojection import reprojection_fidelity  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--layout-hint", type=str, default=None)
    parser.add_argument("--probability-threshold", type=float, default=0.25)
    parser.add_argument("--precision-tolerance", type=int, default=3)
    parser.add_argument("--recall-tolerance", type=int, default=8)
    args = parser.parse_args()

    digitiser = ECGDigitiser(enable_dewarping_retry=False, enable_orientation_retry=False)
    digitiser.digitize_with_calibrated(args.image, layout_hint=args.layout_hint)
    info = digitiser.last_info

    if info.signal_probability is None or info.raw_lines is None:
        raise RuntimeError(
            "digitizer did not expose re-projection artifacts; "
            "is the Open-ECG-Digitizer vendor patch applied?"
        )

    fidelity = reprojection_fidelity(
        info.signal_probability,
        info.raw_lines,
        extraction_crop_x0=info.extraction_crop_x0,
        probability_threshold=args.probability_threshold,
        precision_tolerance=args.precision_tolerance,
        recall_tolerance=args.recall_tolerance,
    )
    print(
        json.dumps(
            {
                "precision": fidelity.precision,
                "recall": fidelity.recall,
                "f1": fidelity.f1,
                "residual": fidelity.residual,
                "ink_pixels": fidelity.ink_pixels,
                "reconstructed_pixels": fidelity.reconstructed_pixels,
                "trace_count": fidelity.trace_count,
                "layout_name": info.layout_name,
                "layout_cost": info.layout_cost,
            }
        )
    )


if __name__ == "__main__":
    main()
