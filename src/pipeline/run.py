# CLI entry point for the ECG diagnosis pipeline
# Reads a WFDB signal, runs ECGFounder inference, and prints/saves results

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from src.pipeline.diagnose import DiagnosisResult, ECGDiagnoser
from src.utils.wfdb_helpers import read_ecg_signal

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/ecgfounder/base/12_lead_ECGFounder.pth")

# Block characters for probability bar visualization (8 levels)
_BAR_FULL = "\u2588"
_BAR_WIDTH = 20


def run_signal_diagnosis(
    record_path: str,
    model_path: str | Path,
    threshold: float,
) -> list[DiagnosisResult]:
    """Read a WFDB record, run ECGFounder diagnosis, and return results.

    Args:
        record_path: Path to WFDB record (without .dat/.hea extension).
        model_path: Path to ECGFounder model checkpoint.
        threshold: Sigmoid probability cutoff for filtering diagnoses.

    Returns:
        List of DiagnosisResult above threshold, sorted by probability descending.
    """
    logger.info("Reading ECG signal from %s", record_path)
    signal, metadata = read_ecg_signal(record_path)
    logger.info(
        "Signal loaded — record=%s, duration=%.1fs, rate=%dHz",
        metadata["record_name"],
        metadata["duration_seconds"],
        metadata["original_sample_rate"],
    )

    diagnoser = ECGDiagnoser(checkpoint_path=model_path, threshold=threshold)
    results = diagnoser.diagnose(signal, threshold=threshold)

    logger.info("Diagnosis complete — %d findings above threshold %.2f", len(results), threshold)
    return results


def print_results(results: list[DiagnosisResult]) -> None:
    """Pretty-print diagnosis results with visual probability bars.

    Args:
        results: List of DiagnosisResult to display.
    """
    if not results:
        print("\nNo diagnoses above threshold.")
        return

    print(f"\n{'=' * 60}")
    print(f"  ECG DIAGNOSIS RESULTS — {len(results)} finding(s)")
    print(f"{'=' * 60}\n")

    for result in results:
        filled = int(result.probability * _BAR_WIDTH)
        bar = _BAR_FULL * filled + " " * (_BAR_WIDTH - filled)
        print(f"  {result.probability:.3f} {bar} {result.label}")

    print()


def _save_results_json(results: list[DiagnosisResult], output_path: Path) -> None:
    """Save diagnosis results to a JSON file.

    Args:
        results: List of DiagnosisResult to save.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Results saved to %s", output_path)


def main() -> None:
    """Parse CLI arguments and run the ECG diagnosis pipeline."""
    parser = argparse.ArgumentParser(
        description="Run ECGFounder diagnosis on a WFDB ECG record.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m src.pipeline.run "
            "--signal data/raw/ptb-xl/records500/00000/00001_hr "
            "--threshold 0.3"
        ),
    )
    parser.add_argument(
        "--signal",
        required=True,
        help="Path to WFDB record (without .dat/.hea extension)",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to ECGFounder checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for filtering diagnoses (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results as JSON to this file path",
    )

    args = parser.parse_args()

    # Configure logging to stderr so stdout stays clean for results
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    results = run_signal_diagnosis(
        record_path=args.signal,
        model_path=args.model,
        threshold=args.threshold,
    )

    print_results(results)

    if args.output:
        _save_results_json(results, Path(args.output))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
