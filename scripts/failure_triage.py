"""Phase 0 failure triage for the surviving PMcardio tune false accepts.

Classifies why a digitization is wrong without re-digitizing: it reuses the
cached digitized signals (shape-only and calibrated) plus the matched printed
reference, and answers two questions per record:

1. **Lead permutation?** Build the 12x12 shifted-correlation matrix between the
   z-scored digitized leads and the z-scored reference leads. If a digitized
   lead correlates best with a *different* reference lead (off-diagonal argmax),
   the assignment is wrong.
2. **Global amplitude-scale error?** Take the per-lead calibrated gain ratios
   (digitized RMS / reference RMS) and check whether they cluster tightly around
   a single constant k != 1 (calibration-scale error) or scatter (not a clean
   scale error).

The output is descriptive JSON, printed to stdout, so the orchestrator can
decide whether Track A (calibration) or Track B (assignment) is the primary win.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.digitize import LEAD_NAMES, NUM_LEADS  # noqa: E402
from src.training.reference_fidelity import shifted_correlation  # noqa: E402

DEFAULT_DATA_DIR = Path("data/reference/pmcardio-holdout")
MAX_SHIFT_SAMPLES = 50  # 100 ms at 500 Hz


def _zscore(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    std = float(np.std(signal))
    if std < 1e-12:
        return np.zeros_like(signal)
    return (signal - float(np.mean(signal))) / std


def cross_correlation_matrix(
    reference: np.ndarray,
    digitized: np.ndarray,
    *,
    max_shift: int,
) -> np.ndarray:
    """Return a 12x12 best-shifted-correlation matrix.

    Entry ``[i, j]`` is the best correlation of digitized lead ``i`` against
    reference lead ``j`` over a bounded horizontal shift. Both signals are
    z-scored inside ``shifted_correlation`` so only morphology matters.
    """
    segment_length = reference.shape[0]
    matrix = np.zeros((NUM_LEADS, NUM_LEADS), dtype=np.float64)
    for digit_index in range(NUM_LEADS):
        predicted_lead = digitized[digit_index, :segment_length]
        for ref_index in range(NUM_LEADS):
            reference_lead = reference[:, ref_index]
            result = shifted_correlation(
                reference_lead,
                predicted_lead,
                max_shift=max_shift,
            )
            matrix[digit_index, ref_index] = result.correlation
    return matrix


def gain_ratios(
    reference: np.ndarray,
    calibrated: np.ndarray,
    *,
    max_shift: int,
) -> list[float]:
    """Per-lead calibrated RMS gain ratio (digitized / reference)."""
    segment_length = reference.shape[0]
    ratios: list[float] = []
    for lead_index in range(NUM_LEADS):
        reference_lead = reference[:, lead_index].astype(np.float64)
        predicted_lead = calibrated[lead_index, :segment_length].astype(np.float64)
        reference_rms = float(np.sqrt(np.mean(reference_lead**2)))
        predicted_rms = float(np.sqrt(np.mean(predicted_lead**2)))
        ratios.append(predicted_rms / reference_rms if reference_rms > 1e-12 else 0.0)
    return ratios


def triage_record(
    reference: np.ndarray,
    digitized: np.ndarray,
    calibrated: np.ndarray,
) -> dict[str, Any]:
    """Classify a single record as scale, permutation, or morphology error."""
    matrix = cross_correlation_matrix(
        reference, digitized, max_shift=MAX_SHIFT_SAMPLES
    )
    diagonal = np.diag(matrix)
    best_match_index = matrix.argmax(axis=1)  # for each digitized lead, best ref
    best_match_corr = matrix.max(axis=1)

    permuted_leads = []
    for digit_index in range(NUM_LEADS):
        best_ref = int(best_match_index[digit_index])
        # A permutation is only meaningful when the off-diagonal match is both
        # the argmax AND clearly better than the on-diagonal score.
        if best_ref != digit_index and (
            best_match_corr[digit_index] - diagonal[digit_index] > 0.10
        ):
            permuted_leads.append(
                {
                    "digitized_lead": LEAD_NAMES[digit_index],
                    "diagonal_corr": round(float(diagonal[digit_index]), 4),
                    "best_match_lead": LEAD_NAMES[best_ref],
                    "best_match_corr": round(float(best_match_corr[digit_index]), 4),
                }
            )

    ratios = gain_ratios(reference, calibrated, max_shift=MAX_SHIFT_SAMPLES)
    ratios_array = np.asarray(ratios)
    # Coefficient of variation: tight cluster => consistent single scale.
    ratio_median = float(np.median(ratios_array))
    ratio_cv = (
        float(np.std(ratios_array) / np.mean(ratios_array))
        if np.mean(ratios_array) > 1e-12
        else float("inf")
    )

    return {
        "diagonal_corr_median": round(float(np.median(diagonal)), 4),
        "diagonal_corr_min": round(float(np.min(diagonal)), 4),
        "permutation": {
            "n_permuted_leads": len(permuted_leads),
            "permuted_leads": permuted_leads,
        },
        "scale": {
            "gain_ratio_median": round(ratio_median, 4),
            "gain_ratio_cv": round(ratio_cv, 4),
            "gain_ratios": [round(r, 4) for r in ratios],
            "deviation_from_unity": round(abs(ratio_median - 1.0), 4),
        },
        "cross_correlation_matrix": [
            [round(float(v), 3) for v in row] for row in matrix
        ],
    }


def load_record(
    data_dir: Path,
    *,
    reference_key: str,
    category: str,
    image_stem: str,
) -> dict[str, Any]:
    references = np.load(data_dir / "leads.npz")
    reference = references[reference_key]
    references.close()
    digitized = np.load(data_dir / "digitized" / category / f"{image_stem}.npy")
    calibrated = np.load(
        data_dir / "digitized-calibrated" / category / f"{image_stem}.npy"
    )
    result = triage_record(reference, digitized, calibrated)
    result["reference_key"] = reference_key
    result["category"] = category
    result["image_stem"] = image_stem
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--records",
        nargs="+",
        required=True,
        help="reference_key:category:image_stem triples",
    )
    args = parser.parse_args()

    results = []
    for spec in args.records:
        reference_key, category, image_stem = spec.split(":")
        results.append(
            load_record(
                args.data_dir,
                reference_key=reference_key,
                category=category,
                image_stem=image_stem,
            )
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
