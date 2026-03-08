# ECG paper-style visualization for digitized 12-lead signals
# Handles the Open-ECG-Digitizer's output where 4 row-level traces
# need to be split into 12 individual leads for proper display.
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

# Standard 12-lead ECG paper layout (4 columns x 3 rows)
# Each row maps to lead indices: [col0, col1, col2, col3]
ROW_TO_LEADS: list[list[int]] = [
    [0, 3, 6, 9],     # Row 0: I, aVR, V1, V4
    [1, 4, 7, 10],    # Row 1: II, aVL, V2, V5
    [2, 5, 8, 11],    # Row 2: III, aVF, V3, V6
]

# Colors matching real ECG paper
COLOR_GRID_MAJOR: str = "#E8B4B4"
COLOR_GRID_MINOR: str = "#F2DCDC"
COLOR_SIGNAL: str = "#1A1A2E"
COLOR_LABEL: str = "#2563EB"
COLOR_BG: str = "#FDF6F0"


def _find_active_leads(signal: np.ndarray) -> list[int]:
    """Find which lead indices have actual signal (not flat/zero)."""
    active = []
    for i in range(signal.shape[0]):
        lead = signal[i]
        if np.std(lead) > 0.01:
            active.append(i)
    return active


def _split_rows_to_leads(signal: np.ndarray) -> dict[int, np.ndarray]:
    """Split 4 row-level traces into 12 individual lead segments.

    The Open-ECG-Digitizer detects 4 physical rows from paper ECGs.
    Each row contains 4 concatenated leads (2.5s each). This function
    splits them into individual leads using the standard 4x3+1 layout.

    Returns:
        Dict mapping lead index (0-11) to its signal segment.
        Also includes key -1 for the rhythm strip (Lead II, full trace).
    """
    active = _find_active_leads(signal)

    # If all 12 leads have signal, no splitting needed
    if len(active) >= 10:
        leads = {}
        for i in range(12):
            leads[i] = signal[i]
        leads[-1] = signal[1]  # Rhythm strip = Lead II
        return leads

    # If exactly 4 leads have signal, assume they're row-level traces
    # The digitizer typically maps them to consecutive indices (e.g. 6-9)
    if len(active) == 4:
        row_traces = [signal[idx] for idx in active]
    elif len(active) == 3:
        # 3 rows detected (no rhythm strip)
        row_traces = [signal[idx] for idx in active]
    else:
        # Unexpected number of active leads — show as-is
        leads = {}
        for i in range(12):
            leads[i] = signal[i]
        leads[-1] = signal[1]
        return leads

    samples_per_lead = len(row_traces[0]) // 4
    leads: dict[int, np.ndarray] = {}

    # Split each row into 4 lead segments
    for row_idx in range(min(len(row_traces), 3)):
        trace = row_traces[row_idx]
        lead_indices = ROW_TO_LEADS[row_idx]
        for col_idx, lead_idx in enumerate(lead_indices):
            start = col_idx * samples_per_lead
            end = start + samples_per_lead
            leads[lead_idx] = trace[start:end]

    # Rhythm strip: 4th row if available, otherwise use row 1 (contains Lead II)
    if len(row_traces) >= 4:
        leads[-1] = row_traces[3]
    elif 1 in leads:
        leads[-1] = leads[1]
    else:
        leads[-1] = np.zeros(len(row_traces[0]))

    return leads


def plot_ecg_paper(
    signal: np.ndarray,
    title: str = "Corio ECG — Digitized Signal",
) -> Figure:
    """Render a 12-lead ECG signal on paper-style grid.

    Automatically detects whether the signal needs row-splitting
    (when digitizer outputs 4 row-level traces instead of 12 leads).

    Args:
        signal: Shape (12, 5000) — z-score normalized signal.
        title: Plot title.

    Returns:
        matplotlib Figure ready for display.
    """
    leads = _split_rows_to_leads(signal)

    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [1, 1, 1, 0.8]},
    )
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # Compute global amplitude range for consistent Y-axis across rows
    all_values = np.concatenate(
        [v for k, v in leads.items() if k >= 0 and np.std(v) > 0.01]
    )
    if len(all_values) > 0:
        y_margin = np.percentile(np.abs(all_values), 99) * 1.3
    else:
        y_margin = 4.0

    # Draw 3 rows of 4 leads each
    for row_idx in range(3):
        ax = axes[row_idx]
        lead_indices = ROW_TO_LEADS[row_idx]

        for col_idx, lead_idx in enumerate(lead_indices):
            segment = leads.get(lead_idx, np.zeros(100))
            seg_len = len(segment)

            # X position: offset by column
            total_samples = seg_len * 4
            x_offset = col_idx * seg_len
            x = np.arange(seg_len) + x_offset
            ax.plot(x, segment, color=COLOR_SIGNAL, linewidth=0.7)

            # Lead label
            label_x = x_offset + 10
            ax.text(
                label_x, y_margin * 0.85,
                LEAD_NAMES[lead_idx],
                fontsize=9, fontweight="bold",
                color=COLOR_LABEL,
            )

            # Vertical separator between columns
            if col_idx > 0:
                ax.axvline(x=x_offset, color=COLOR_GRID_MAJOR, linewidth=1.2)

        _setup_grid(ax, x_max=total_samples, y_range=y_margin, seg_len=seg_len)

    # Rhythm strip: Lead II across full duration
    ax_rhythm = axes[3]
    rhythm = leads.get(-1, np.zeros(100))
    x_rhythm = np.arange(len(rhythm))
    ax_rhythm.plot(x_rhythm, rhythm, color=COLOR_SIGNAL, linewidth=0.7)
    ax_rhythm.text(
        10, y_margin * 0.85,
        "II (rhythm strip)",
        fontsize=9, fontweight="bold", color=COLOR_LABEL,
    )
    _setup_grid(ax_rhythm, x_max=len(rhythm), y_range=y_margin, seg_len=len(rhythm))

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


def _setup_grid(
    ax: plt.Axes,
    x_max: int = 5000,
    y_range: float = 4.0,
    seg_len: int = 1250,
) -> None:
    """Configure ECG paper grid on an axis."""
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-y_range, y_range)

    # Minor grid — scale tick spacing to segment length
    minor_x_step = max(seg_len // 25, 1)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_x_step))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_range / 20))
    ax.grid(which="minor", color=COLOR_GRID_MINOR, linewidth=0.3)

    # Major grid
    major_x_step = max(seg_len // 5, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(major_x_step))
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
    return Image.open(buf)
