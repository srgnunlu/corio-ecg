# ECG paper-style visualization for calibrated digitized 12-lead signals
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
MINOR_VOLTAGE_MV: float = 0.1
MAJOR_VOLTAGE_MV: float = 0.5
DEFAULT_Y_RANGE_MV: float = 1.0
FIGURE_SIZE_IN: tuple[float, float] = (18.0, 9.0)
RENDER_DPI: int = 200
CALIBRATION_MV: float = 1.0
CALIBRATION_WIDTH_SAMPLES: int = int(SAMPLE_RATE * 0.2)
CALIBRATION_GAP_SAMPLES: int = int(SAMPLE_RATE * 0.08)
CALIBRATION_RIGHT_PAD_SAMPLES: int = int(SAMPLE_RATE * 0.08)

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

# High-contrast blue ECG paper style, tuned for screen/PDF readability.
COLOR_GRID_MAJOR: str = "#6EA0E8"
COLOR_GRID_MINOR: str = "#B9D0F6"
COLOR_SIGNAL: str = "#111111"
COLOR_LABEL: str = "#0F5BD7"
COLOR_BG: str = "#FFFFFF"


def _extract_lead_segment(signal: np.ndarray, lead_idx: int, col_idx: int) -> np.ndarray:
    """Extract the 5-second segment of a lead from its column position.

    After canonical segment expansion in the digitize pipeline, all leads
    contain tiled data across 5000 samples. Simple column-based slicing is
    sufficient: col 0 → [0:2500], col 1 → [2500:5000].
    """
    start = col_idx * SAMPLES_PER_COLUMN
    end = start + SAMPLES_PER_COLUMN
    return signal[lead_idx, start:end]


def plot_ecg_paper(
    signal: np.ndarray,
    title: str = "",
) -> Figure:
    """Render a 12-lead ECG signal in 6x2+1R paper-style layout.

    Args:
        signal: Shape (12, 5000) — calibrated millivolt signal.
        title: Plot title.

    Returns:
        matplotlib Figure ready for display.
    """
    n_rows = 6
    fig, axes = plt.subplots(
        n_rows + 1, 1,
        figsize=FIGURE_SIZE_IN,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1, 1]},
    )
    fig.patch.set_facecolor(COLOR_BG)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.985)

    # Collect all segments to compute global amplitude range
    all_segments: list[np.ndarray] = []
    for row_idx in range(n_rows):
        for col_idx, lead_idx in enumerate(ROW_TO_LEADS[row_idx]):
            seg = _extract_lead_segment(signal, lead_idx, col_idx)
            all_segments.append(seg)

    y_margin = _compute_y_range_mv(all_segments)

    signal_width = SAMPLES_PER_COLUMN * 2  # 5000 samples total
    x_max = (
        signal_width
        + CALIBRATION_GAP_SAMPLES
        + CALIBRATION_WIDTH_SAMPLES
        + CALIBRATION_RIGHT_PAD_SAMPLES
    )

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
            ax.plot(x, segment, color=COLOR_SIGNAL, linewidth=0.9)

            # Lead label
            ax.text(
                x_offset + 10, y_margin * 0.70,
                LEAD_NAMES[lead_idx],
                fontsize=9, fontweight="bold",
                color=COLOR_LABEL,
            )

            # Vertical separator between columns
            if col_idx > 0:
                ax.axvline(x=x_offset, color=COLOR_GRID_MAJOR, linewidth=1.2)

        _draw_calibration_pulse(ax, signal_width + CALIBRATION_GAP_SAMPLES)
        _setup_grid(ax, x_max=x_max, y_range=y_margin)

    # Rhythm strip: Lead II full 10 seconds
    ax_rhythm = axes[n_rows]
    rhythm = signal[1, :]  # Lead II
    if np.std(rhythm) < 0.01:
        for i in range(12):
            if np.std(signal[i]) > 0.01:
                rhythm = signal[i]
                break

    x_rhythm = np.arange(len(rhythm))
    ax_rhythm.plot(x_rhythm, rhythm, color=COLOR_SIGNAL, linewidth=0.9)
    ax_rhythm.text(
        10, y_margin * 0.70,
        "II (rhythm strip)",
        fontsize=9, fontweight="bold", color=COLOR_LABEL,
    )
    _draw_calibration_pulse(ax_rhythm, signal_width + CALIBRATION_GAP_SAMPLES)
    _setup_grid(ax_rhythm, x_max=x_max, y_range=y_margin)

    # Footer
    fig.text(
        0.02, 0.005,
        f"Paper speed: {PAPER_SPEED_MM_S:.1f} mm/s, "
        f"Voltage gain: {VOLTAGE_GAIN_MM_MV:.1f} mm/mV",
        fontsize=8, color=COLOR_LABEL,
    )
    fig.text(
        0.98, 0.005,
        "Visualization by Corio ECG",
        fontsize=8, color="#334155", ha="right",
    )

    top = 0.95 if title else 0.985
    fig.subplots_adjust(
        left=0.035,
        right=0.992,
        top=top,
        bottom=0.055,
        hspace=0.0,
    )
    return fig


def _setup_grid(
    ax: plt.Axes,
    x_max: int = 5000,
    y_range: float = DEFAULT_Y_RANGE_MV,
) -> None:
    """Configure ECG paper grid on an axis."""
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-y_range, y_range)

    # Minor grid (1mm = 0.04s = 20 samples at 500Hz)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(SAMPLE_RATE * 0.04))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(MINOR_VOLTAGE_MV))
    ax.grid(which="minor", color=COLOR_GRID_MINOR, linewidth=0.3)

    # Major grid (5mm = 0.2s = 100 samples at 500Hz)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(SAMPLE_RATE * 0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(MAJOR_VOLTAGE_MV))
    ax.grid(which="major", color=COLOR_GRID_MAJOR, linewidth=0.6)

    # Hide axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_calibration_pulse(ax: plt.Axes, x_start: int) -> None:
    """Draw a standard 1 mV, 200 ms calibration pulse at the right edge."""
    baseline = -CALIBRATION_MV / 2.0
    top = baseline + CALIBRATION_MV
    x_end = x_start + CALIBRATION_WIDTH_SAMPLES
    ax.plot(
        [x_start, x_start, x_end, x_end],
        [baseline, top, top, baseline],
        color=COLOR_LABEL,
        linewidth=1.0,
        label="1 mV calibration",
    )


def _compute_y_range_mv(segments: list[np.ndarray]) -> float:
    """Choose a mV display window without changing ECG paper grid scale."""
    nonflat_segments = [s for s in segments if np.std(s) > 0.01]
    if not nonflat_segments:
        return DEFAULT_Y_RANGE_MV

    values = np.concatenate(nonflat_segments)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return DEFAULT_Y_RANGE_MV

    robust_peak = float(np.percentile(np.abs(finite_values), 99.5))
    required = max(DEFAULT_Y_RANGE_MV, robust_peak * 1.15)
    return float(np.ceil(required / MAJOR_VOLTAGE_MV) * MAJOR_VOLTAGE_MV)


def fig_to_pil(fig: Figure) -> Image.Image:
    """Convert matplotlib Figure to PIL Image for Gradio display."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=RENDER_DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    image.load()
    buf.close()
    return image
