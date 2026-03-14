# ECG paper-style visualization for digitized 12-lead signals
# Renders signals on a medical-standard grid (25mm/s, 10mm/mV)
# Layout: 6 rows x 2 columns + Lead II rhythm strip (6x2+1R)

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

# 6x2+1R layout: limb leads left column, precordial leads right column
# Each lead shows 5 seconds (half of the 10-second recording)
ROW_TO_LEADS: list[list[int]] = [
    [0, 6],    # Row 0: I,   V1
    [1, 7],    # Row 1: II,  V2
    [2, 8],    # Row 2: III, V3
    [3, 9],    # Row 3: aVR, V4
    [4, 10],   # Row 4: aVL, V5
    [5, 11],   # Row 5: aVF, V6
]

SAMPLES_PER_COLUMN: int = int(5.0 * SAMPLE_RATE)  # 2500 (5 seconds per column)

# Colors matching real ECG paper
COLOR_GRID_MAJOR: str = "#E8B4B4"
COLOR_GRID_MINOR: str = "#F2DCDC"
COLOR_SIGNAL: str = "#1A1A2E"
COLOR_LABEL: str = "#2563EB"
COLOR_BG: str = "#FDF6F0"


def _extract_lead_segment(signal: np.ndarray, lead_idx: int, col_idx: int) -> np.ndarray:
    """Extract the 5-second segment of a lead from its column position.

    In 6x2 layout, column 0 gets samples 0-2499 (first 5 seconds),
    column 1 gets samples 2500-4999 (last 5 seconds).
    """
    start = col_idx * SAMPLES_PER_COLUMN
    end = start + SAMPLES_PER_COLUMN
    segment = signal[lead_idx, start:end]

    # If this segment is mostly zero/NaN, try finding data elsewhere
    if np.std(segment) < 0.01 and np.std(signal[lead_idx]) > 0.01:
        full_lead = signal[lead_idx]
        best_start = 0
        best_energy = 0.0
        for s in range(0, len(full_lead) - SAMPLES_PER_COLUMN + 1, SAMPLES_PER_COLUMN // 4):
            chunk = full_lead[s : s + SAMPLES_PER_COLUMN]
            energy = np.sum(chunk ** 2)
            if energy > best_energy:
                best_energy = energy
                best_start = s
        segment = full_lead[best_start : best_start + SAMPLES_PER_COLUMN]

    return segment


def plot_ecg_paper(
    signal: np.ndarray,
    title: str = "Corio ECG — Digitized Signal",
) -> Figure:
    """Render a 12-lead ECG signal in 6x2+1R paper-style layout.

    Args:
        signal: Shape (12, 5000) — z-score normalized signal.
        title: Plot title.

    Returns:
        matplotlib Figure ready for display.
    """
    n_rows = 6
    fig, axes = plt.subplots(
        n_rows + 1, 1,
        figsize=(14, 14),
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1, 0.8]},
    )
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)

    # Collect all segments to compute global amplitude range
    all_segments: list[np.ndarray] = []
    for row_idx in range(n_rows):
        for col_idx, lead_idx in enumerate(ROW_TO_LEADS[row_idx]):
            seg = _extract_lead_segment(signal, lead_idx, col_idx)
            all_segments.append(seg)

    nonflat_segments = [s for s in all_segments if np.std(s) > 0.01]
    if nonflat_segments:
        all_values = np.concatenate(nonflat_segments)
        y_margin = np.percentile(np.abs(all_values), 98) * 1.3
    else:
        y_margin = 4.0
    y_margin = max(y_margin, 1.0)

    total_width = SAMPLES_PER_COLUMN * 2  # 5000 samples total

    # Draw 6 rows of 2 leads each
    seg_idx = 0
    for row_idx in range(n_rows):
        ax = axes[row_idx]
        lead_indices = ROW_TO_LEADS[row_idx]

        for col_idx, lead_idx in enumerate(lead_indices):
            segment = all_segments[seg_idx]
            seg_idx += 1

            x_offset = col_idx * SAMPLES_PER_COLUMN
            x = np.arange(len(segment)) + x_offset
            ax.plot(x, segment, color=COLOR_SIGNAL, linewidth=0.7)

            # Lead label
            ax.text(
                x_offset + 10, y_margin * 0.85,
                LEAD_NAMES[lead_idx],
                fontsize=9, fontweight="bold",
                color=COLOR_LABEL,
            )

            # Vertical separator between columns
            if col_idx > 0:
                ax.axvline(x=x_offset, color=COLOR_GRID_MAJOR, linewidth=1.2)

        _setup_grid(ax, x_max=total_width, y_range=y_margin)

    # Rhythm strip: Lead II full 10 seconds
    ax_rhythm = axes[n_rows]
    rhythm = signal[1, :]  # Lead II
    if np.std(rhythm) < 0.01:
        for i in range(12):
            if np.std(signal[i]) > 0.01:
                rhythm = signal[i]
                break

    x_rhythm = np.arange(len(rhythm))
    ax_rhythm.plot(x_rhythm, rhythm, color=COLOR_SIGNAL, linewidth=0.7)
    ax_rhythm.text(
        10, y_margin * 0.85,
        "II (rhythm strip)",
        fontsize=9, fontweight="bold", color=COLOR_LABEL,
    )
    _setup_grid(ax_rhythm, x_max=len(rhythm), y_range=y_margin)

    # Footer
    fig.text(
        0.02, 0.005,
        f"Paper speed: {PAPER_SPEED_MM_S:.0f} mm/s, "
        f"Voltage gain: {VOLTAGE_GAIN_MM_MV:.0f} mm/mV",
        fontsize=7, color="#666666",
    )
    fig.text(
        0.98, 0.005,
        "Visualization by Corio ECG",
        fontsize=7, color="#666666", ha="right",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    return fig


def _setup_grid(
    ax: plt.Axes,
    x_max: int = 5000,
    y_range: float = 4.0,
) -> None:
    """Configure ECG paper grid on an axis."""
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-y_range, y_range)

    # Minor grid (1mm = 0.04s = 20 samples at 500Hz)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(SAMPLE_RATE * 0.04))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_range / 20))
    ax.grid(which="minor", color=COLOR_GRID_MINOR, linewidth=0.3)

    # Major grid (5mm = 0.2s = 100 samples at 500Hz)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(SAMPLE_RATE * 0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_range / 4))
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
    image = Image.open(buf)
    image.load()
    buf.close()
    return image
