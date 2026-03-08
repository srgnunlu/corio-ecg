# ECG paper-style visualization for digitized 12-lead signals
# Renders signals on a medical-standard grid (25mm/s, 10mm/mV)
# Layout: 4 columns x 3 rows + rhythm strip (Lead II)

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from src.utils.ecg_labels import LEAD_NAMES

# Standard ECG paper settings
PAPER_SPEED_MM_S: float = 25.0
VOLTAGE_GAIN_MM_MV: float = 10.0
SAMPLE_RATE: int = 500

# Grid layout: standard 12-lead arrangement
# Each column shows 2.5 seconds of a different lead
GRID_LAYOUT: list[list[int]] = [
    [0, 3, 6, 9],     # I, aVR, V1, V4
    [1, 4, 7, 10],    # II, aVL, V2, V5
    [2, 5, 8, 11],    # III, aVF, V3, V6
]

SECONDS_PER_COLUMN: float = 2.5
SAMPLES_PER_COLUMN: int = int(SECONDS_PER_COLUMN * SAMPLE_RATE)

# Colors matching real ECG paper
COLOR_GRID_MAJOR: str = "#E8B4B4"
COLOR_GRID_MINOR: str = "#F2DCDC"
COLOR_SIGNAL: str = "#1A1A2E"
COLOR_LABEL: str = "#2563EB"
COLOR_BG: str = "#FDF6F0"


def plot_ecg_paper(
    signal: np.ndarray,
    title: str = "Corio ECG — Digitized Signal",
) -> Figure:
    """Render a 12-lead ECG signal on paper-style grid.

    Args:
        signal: Shape (12, 5000) — z-score normalized signal.
        title: Plot title.

    Returns:
        matplotlib Figure ready for display.
    """
    display_signal = signal.copy()

    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [1, 1, 1, 0.8]},
    )
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # Draw 3 rows of 4 leads each
    for row_idx in range(3):
        ax = axes[row_idx]
        _setup_grid(ax)
        lead_indices = GRID_LAYOUT[row_idx]

        for col_idx, lead_idx in enumerate(lead_indices):
            start = col_idx * SAMPLES_PER_COLUMN
            end = start + SAMPLES_PER_COLUMN
            segment = display_signal[lead_idx, start:end]

            x = np.arange(len(segment)) + (col_idx * SAMPLES_PER_COLUMN)
            ax.plot(x, segment, color=COLOR_SIGNAL, linewidth=0.7)

            # Lead label at start of each column
            label_x = col_idx * SAMPLES_PER_COLUMN + 20
            label_y = ax.get_ylim()[1] * 0.85
            ax.text(
                label_x, label_y,
                LEAD_NAMES[lead_idx],
                fontsize=9, fontweight="bold",
                color=COLOR_LABEL,
            )

            # Vertical separator between columns
            if col_idx > 0:
                sep_x = col_idx * SAMPLES_PER_COLUMN
                ax.axvline(x=sep_x, color=COLOR_GRID_MAJOR, linewidth=1.2)

    # Rhythm strip: Lead II across full 10 seconds
    ax_rhythm = axes[3]
    _setup_grid(ax_rhythm)
    rhythm_signal = display_signal[1, :5000]
    x_rhythm = np.arange(len(rhythm_signal))
    ax_rhythm.plot(x_rhythm, rhythm_signal, color=COLOR_SIGNAL, linewidth=0.7)
    ax_rhythm.text(
        20, ax_rhythm.get_ylim()[1] * 0.85,
        "II (rhythm strip)",
        fontsize=9, fontweight="bold", color=COLOR_LABEL,
    )

    # Footer
    fig.text(
        0.02, 0.01,
        f"Paper speed: {PAPER_SPEED_MM_S:.0f} mm/s, "
        f"Voltage gain: {VOLTAGE_GAIN_MM_MV:.0f} mm/mV",
        fontsize=7, color="#666666",
    )
    fig.text(
        0.98, 0.01,
        "Visualization by Corio ECG",
        fontsize=7, color="#666666", ha="right",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


def _setup_grid(ax: plt.Axes) -> None:
    """Configure ECG paper grid on an axis."""
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, 5000)
    ax.set_ylim(-4, 4)

    # Minor grid (1mm equivalent)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(SAMPLE_RATE * 0.04))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.grid(which="minor", color=COLOR_GRID_MINOR, linewidth=0.3)

    # Major grid (5mm equivalent)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(SAMPLE_RATE * 0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.grid(which="major", color=COLOR_GRID_MAJOR, linewidth=0.6)

    # Hide axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)


def fig_to_pil(fig: Figure) -> Image.Image:
    """Convert matplotlib Figure to PIL Image for Gradio display."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
