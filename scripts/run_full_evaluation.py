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

DIFFICULTY_LEVELS = ["clean", "moderate", "hard"]


def step_generate(data_dir: Path, image_dir: Path, max_samples: int | None) -> None:
    """Step 1: Generate synthetic ECG images at all difficulty levels.

    Args:
        data_dir: Path to PTB-XL dataset root.
        image_dir: Output directory for generated images.
        max_samples: Optional limit on number of records.
    """
    from scripts.generate_synthetic_images import generate_images, print_summary

    print("\n" + "=" * 60)
    print("STEP 1: GENERATING SYNTHETIC ECG IMAGES")
    print("=" * 60)

    start = time.time()
    summary = generate_images(
        data_dir=data_dir,
        output_dir=image_dir,
        difficulties=DIFFICULTY_LEVELS,
        max_samples=max_samples,
    )
    print_summary(summary, time.time() - start)


def step_digitize(
    image_dir: Path, signal_dir: Path, max_samples: int | None
) -> None:
    """Step 2: Digitize synthetic images back to numpy signals.

    Args:
        image_dir: Directory containing synthetic ECG images.
        signal_dir: Output directory for digitized signal files.
        max_samples: Optional limit on number of images per level.
    """
    from scripts.digitize_synthetic_images import (
        collect_image_paths,
        digitize_batch,
        print_summary,
    )
    from src.pipeline.digitize import ECGDigitiser

    print("\n" + "=" * 60)
    print("STEP 2: DIGITIZING SYNTHETIC IMAGES")
    print("=" * 60)

    # Load model once, reuse across all difficulty levels
    print("Loading ECGDigitiser model...")
    digitiser = ECGDigitiser()

    for level in DIFFICULTY_LEVELS:
        print(f"\n--- Digitizing [{level}] ---")
        start = time.time()

        try:
            image_paths = collect_image_paths(image_dir, level, max_samples)
        except FileNotFoundError as error:
            print(f"  Skipping {level}: {error}")
            continue

        summary = digitize_batch(
            digitiser=digitiser,
            image_paths=image_paths,
            output_dir=signal_dir,
            level=level,
        )
        print_summary(summary, time.time() - start)


def step_evaluate(
    data_dir: Path,
    signal_dir: Path,
    model_path: Path,
    threshold: float,
    max_samples: int | None,
    results_path: Path,
) -> dict:
    """Step 3: Run round-trip evaluation comparing clean vs digitized diagnoses.

    Args:
        data_dir: Path to PTB-XL dataset root.
        signal_dir: Path to digitized signals directory.
        model_path: Path to ECGFounder checkpoint.
        threshold: Probability threshold for agreement rate.
        max_samples: Optional limit on number of records.
        results_path: Where to save the results JSON.

    Returns:
        Results dictionary.
    """
    from src.training.evaluate_roundtrip import evaluate_roundtrip, _print_comparison_table

    print("\n" + "=" * 60)
    print("STEP 3: ROUND-TRIP EVALUATION")
    print("=" * 60)

    start = time.time()
    results = evaluate_roundtrip(
        data_dir=data_dir,
        digitized_dir=signal_dir,
        model_path=model_path,
        threshold=threshold,
        max_samples=max_samples,
    )
    elapsed = time.time() - start

    # Save results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Evaluation time: {elapsed:.1f} sec")

    _print_comparison_table(results)
    return results


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
    print(f"Plots:      results/figures/")

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
    args = parser.parse_args()

    total_start = time.time()

    # Step 1: Generate synthetic images
    if not args.skip_generation:
        step_generate(DEFAULT_DATA_DIR, DEFAULT_IMAGE_DIR, args.max_samples)
    else:
        print("\n[SKIP] Image generation (--skip-generation)")

    # Step 2: Digitize images to signals
    if not args.skip_digitization:
        step_digitize(DEFAULT_IMAGE_DIR, DEFAULT_SIGNAL_DIR, args.max_samples)
    else:
        print("[SKIP] Digitization (--skip-digitization)")

    # Step 3: Run round-trip evaluation
    if not args.skip_evaluation:
        step_evaluate(
            data_dir=DEFAULT_DATA_DIR,
            signal_dir=DEFAULT_SIGNAL_DIR,
            model_path=DEFAULT_MODEL_PATH,
            threshold=args.threshold,
            max_samples=args.max_samples,
            results_path=DEFAULT_RESULTS_PATH,
        )
    else:
        print("[SKIP] Evaluation (--skip-evaluation)")

    # Step 4: Generate plots (always runs unless results don't exist)
    step_plot(DEFAULT_RESULTS_PATH, DEFAULT_PLOTS_DIR)

    total_elapsed = time.time() - total_start
    print_final_summary(DEFAULT_RESULTS_PATH, total_elapsed)


if __name__ == "__main__":
    main()
