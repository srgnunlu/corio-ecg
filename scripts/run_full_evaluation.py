# Full Phase 2 evaluation orchestrator — runs the complete pipeline:
# 1. Generate synthetic ECG images from PTB-XL test fold
# 2. Digitize images back to signals
# 3. Run round-trip evaluation (clean vs digitized diagnosis)
# 4. Generate comparison plots

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Default paths matching individual scripts
DEFAULT_DATA_DIR = Path("data/raw/ptb-xl")
DEFAULT_IMAGE_DIR = Path("data/processed/images")
DEFAULT_SIGNAL_DIR = Path("data/processed/signals")
DEFAULT_MODEL_PATH = Path("models/ecgfounder/base/12_lead_ECGFounder.pth")
DEFAULT_RESULTS_PATH = Path("results/metrics/roundtrip_comparison.json")
DEFAULT_PLOTS_DIR = Path("results/figures")
DEFAULT_THRESHOLD = 0.5
DEFAULT_DIGITIZER_TIMEOUT_SECONDS = 90

DIFFICULTY_LEVELS = ["clean", "moderate", "hard"]


def _run_script(
    cmd: list[str],
    description: str,
    timeout_seconds: int | None = None,
) -> None:
    """Run a Python script as a subprocess, streaming output to stdout.

    Args:
        cmd: Command list to execute.
        description: Human-readable name for error messages.

    Raises:
        RuntimeError: If the subprocess exits with non-zero code.
    """
    import subprocess

    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{description} timed out after {timeout_seconds} seconds"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {result.returncode}")


def step_generate(data_dir: Path, image_dir: Path, max_samples: int | None) -> None:
    """Step 1: Generate synthetic ECG images at all difficulty levels."""
    import sys

    print("\n" + "=" * 60)
    print("STEP 1: GENERATING SYNTHETIC ECG IMAGES")
    print("=" * 60)

    cmd = [sys.executable, "scripts/generate_synthetic_images.py"]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]
    _run_script(cmd, "Image generation")


def step_digitize(
    image_dir: Path,
    signal_dir: Path,
    max_samples: int | None,
    timeout_seconds: int = DEFAULT_DIGITIZER_TIMEOUT_SECONDS,
) -> None:
    """Step 2: Digitize synthetic images back to numpy signals."""
    import sys

    print("\n" + "=" * 60)
    print("STEP 2: DIGITIZING SYNTHETIC IMAGES")
    print("=" * 60)

    for level in DIFFICULTY_LEVELS:
        print(f"\n--- Digitizing [{level}] ---")
        level_dir = image_dir / level
        if not level_dir.exists():
            print(f"  Skipping {level}: directory not found")
            continue
        from digitize_synthetic_images import (
            _is_current_saved_signal,
            collect_image_paths,
        )

        image_paths = collect_image_paths(image_dir, level, max_samples)
        pending = [
            path
            for path in image_paths
            if not _is_current_saved_signal(signal_dir / level / f"{path.stem}.npy")
        ]
        print(f"  Pending {level}: {len(pending)}")
        failures: list[str] = []
        for image_path in pending:
            cmd = [
                sys.executable,
                "scripts/digitize_synthetic_images.py",
                "--level",
                level,
                "--ecg-id",
                image_path.stem,
            ]
            if max_samples is not None:
                cmd += ["--max-samples", str(max_samples)]
            try:
                _run_script(
                    cmd,
                    f"Digitization [{level}] ecg_id={image_path.stem}",
                    timeout_seconds=timeout_seconds,
                )
            except RuntimeError as error:
                failures.append(image_path.stem)
                print(f"  Warning: {error} — continuing with next record")
        if failures:
            print(f"  Failed/timed out {level} records: {', '.join(failures)}")


def step_evaluate(
    max_samples: int | None,
    threshold: float,
) -> None:
    """Step 3: Run round-trip evaluation comparing clean vs digitized diagnoses."""
    import sys

    print("\n" + "=" * 60)
    print("STEP 3: ROUND-TRIP EVALUATION")
    print("=" * 60)

    cmd = [sys.executable, "-m", "src.training.evaluate_roundtrip"]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]
    cmd += ["--threshold", str(threshold)]
    _run_script(cmd, "Round-trip evaluation")


def step_plot(results_path: Path, plots_dir: Path) -> None:
    """Step 4: Generate comparison bar charts from results.

    Args:
        results_path: Path to roundtrip_comparison.json.
        plots_dir: Output directory for PNG plot files.
    """
    from src.utils.plot_results import plot_scenario_comparison

    print("\n" + "=" * 60)
    print("STEP 4: GENERATING COMPARISON PLOTS")
    print("=" * 60)

    plot_scenario_comparison(results_path, plots_dir)


def print_final_summary(results_path: Path, total_elapsed: float) -> None:
    """Print a brief final summary of the full pipeline run.

    Args:
        results_path: Path to the results JSON.
        total_elapsed: Total wall-clock time in seconds.
    """
    print("\n" + "=" * 60)
    print("FULL EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_elapsed / 60:.1f} min ({total_elapsed:.0f} sec)")
    print(f"Results:    {results_path}")
    print("Plots:      results/figures/")

    # Show a quick summary if results exist
    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)

        print(f"\nRecords evaluated: {results['n_records']}")
        for name, data in results["scenarios"].items():
            if name == "baseline":
                continue
            cos_sim = data.get("mean_cosine_similarity_vs_baseline", "N/A")
            agree = data.get("agreement_rate_vs_baseline", "N/A")
            cos_str = f"{cos_sim:.4f}" if isinstance(cos_sim, float) else cos_sim
            agr_str = f"{agree:.4f}" if isinstance(agree, float) else agree
            print(f"  {name}: CosSim={cos_str}, Agreement={agr_str}")

    print("=" * 60)


def main() -> None:
    """CLI entry point for the full Phase 2 evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the complete Phase 2 evaluation pipeline"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of records to process (default: all)",
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip image generation (use existing images)",
    )
    parser.add_argument(
        "--skip-digitization", action="store_true",
        help="Skip digitization (use existing signals)",
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true",
        help="Skip evaluation (use existing results, only regenerate plots)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Probability threshold for agreement rate (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--digitizer-timeout",
        type=int,
        default=DEFAULT_DIGITIZER_TIMEOUT_SECONDS,
        help="Maximum seconds allowed for each isolated image digitization",
    )
    args = parser.parse_args()

    total_start = time.time()

    # Step 1: Generate synthetic images
    if not args.skip_generation:
        step_generate(DEFAULT_DATA_DIR, DEFAULT_IMAGE_DIR, args.max_samples)
    else:
        print("\n[SKIP] Image generation (--skip-generation)")

    # Step 2: Digitize images to signals
    if not args.skip_digitization:
        step_digitize(
            DEFAULT_IMAGE_DIR,
            DEFAULT_SIGNAL_DIR,
            args.max_samples,
            timeout_seconds=args.digitizer_timeout,
        )
    else:
        print("[SKIP] Digitization (--skip-digitization)")

    # Step 3: Run round-trip evaluation
    if not args.skip_evaluation:
        step_evaluate(max_samples=args.max_samples, threshold=args.threshold)
    else:
        print("[SKIP] Evaluation (--skip-evaluation)")

    # Step 4: Generate plots (always runs unless results don't exist)
    step_plot(DEFAULT_RESULTS_PATH, DEFAULT_PLOTS_DIR)

    total_elapsed = time.time() - total_start
    print_final_summary(DEFAULT_RESULTS_PATH, total_elapsed)


if __name__ == "__main__":
    main()
