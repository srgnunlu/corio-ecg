# Visualization utilities for round-trip evaluation results
# Generates comparison bar charts from roundtrip_comparison.json

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# Scenario display names for cleaner chart labels
SCENARIO_LABELS: dict[str, str] = {
    "clean_roundtrip": "Clean",
    "moderate_roundtrip": "Moderate",
    "hard_roundtrip": "Hard",
}

# Color palette — blue shades (darker = easier scenario)
SCENARIO_COLORS: dict[str, str] = {
    "clean_roundtrip": "#1a5276",
    "moderate_roundtrip": "#2e86c1",
    "hard_roundtrip": "#85c1e9",
}


def _load_results(results_path: Path) -> dict[str, Any]:
    """Load roundtrip comparison JSON file.

    Args:
        results_path: Path to roundtrip_comparison.json.

    Returns:
        Parsed results dictionary.

    Raises:
        FileNotFoundError: If the results file does not exist.
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r") as f:
        return json.load(f)


def _get_roundtrip_scenarios(results: dict[str, Any]) -> list[str]:
    """Extract roundtrip scenario names (excluding baseline) from results.

    Args:
        results: Parsed results dictionary.

    Returns:
        List of scenario keys that have comparison metrics.
    """
    return [
        name for name in results["scenarios"]
        if name != "baseline" and "mean_cosine_similarity_vs_baseline" in results["scenarios"][name]
    ]


def _setup_style() -> None:
    """Apply a clean academic matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        # Fallback if seaborn style is not available
        plt.style.use("ggplot")


def _create_bar_chart(
    scenario_names: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    fmt: str = ".4f",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Create and save a single bar chart comparing scenarios.

    Args:
        scenario_names: List of scenario keys.
        values: Corresponding metric values.
        title: Chart title.
        ylabel: Y-axis label.
        output_path: Where to save the PNG file.
        fmt: Format string for value annotations on bars.
        ylim: Optional (min, max) for y-axis.
    """
    labels = [SCENARIO_LABELS.get(name, name) for name in scenario_names]
    colors = [SCENARIO_COLORS.get(name, "#5dade2") for name in scenario_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = np.arange(len(labels))
    bars = ax.bar(x_positions, values, color=colors, width=0.5, edgecolor="white")

    # Add value annotations on top of each bar
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_scenario_comparison(results_path: Path, output_dir: Path) -> None:
    """Generate all comparison plots from roundtrip results.

    Creates four bar charts comparing scenarios:
    - Cosine similarity vs baseline
    - Agreement rate vs baseline
    - Mean Pearson correlation (signal quality)
    - Mean absolute probability difference vs baseline

    All saved as PNG to output_dir at 300 DPI.

    Args:
        results_path: Path to roundtrip_comparison.json.
        output_dir: Directory where PNG plots will be saved.
    """
    _setup_style()

    results = _load_results(results_path)
    scenarios = _get_roundtrip_scenarios(results)

    if not scenarios:
        print("No roundtrip scenarios with comparison metrics found. Skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_data = results["scenarios"]

    print(f"Generating plots for {len(scenarios)} scenarios...")

    # 1. Cosine similarity by scenario
    cosine_values = [
        scenario_data[s]["mean_cosine_similarity_vs_baseline"] for s in scenarios
    ]
    _create_bar_chart(
        scenario_names=scenarios,
        values=cosine_values,
        title="Cosine Similarity vs Baseline by Scenario",
        ylabel="Cosine Similarity",
        output_path=output_dir / "cosine_similarity_by_scenario.png",
        ylim=(0.0, 1.05),
    )

    # 2. Agreement rate by scenario
    agreement_values = [
        scenario_data[s]["agreement_rate_vs_baseline"] for s in scenarios
    ]
    _create_bar_chart(
        scenario_names=scenarios,
        values=agreement_values,
        title="Agreement Rate vs Baseline by Scenario",
        ylabel="Agreement Rate",
        output_path=output_dir / "agreement_rate_by_scenario.png",
        ylim=(0.0, 1.05),
    )

    # 3. Mean Pearson correlation by scenario
    pearson_values = [
        scenario_data[s]["mean_pearson_correlation"] for s in scenarios
    ]
    _create_bar_chart(
        scenario_names=scenarios,
        values=pearson_values,
        title="Mean Pearson Correlation by Scenario",
        ylabel="Pearson Correlation",
        output_path=output_dir / "pearson_correlation_by_scenario.png",
        ylim=(0.0, 1.05),
    )

    # 4. Mean absolute probability difference by scenario
    abs_diff_values = [
        scenario_data[s]["mean_abs_prob_diff_vs_baseline"] for s in scenarios
    ]
    _create_bar_chart(
        scenario_names=scenarios,
        values=abs_diff_values,
        title="Mean Absolute Probability Difference vs Baseline",
        ylabel="Mean |P_roundtrip - P_baseline|",
        output_path=output_dir / "abs_prob_diff_by_scenario.png",
        fmt=".6f",
    )

    print(f"All plots saved to {output_dir}")
