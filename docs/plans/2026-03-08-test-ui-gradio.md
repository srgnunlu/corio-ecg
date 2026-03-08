# Test UI (Gradio) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Gradio web UI that accepts an ECG photo, digitizes it, runs 150-class diagnosis, and displays results with a PMcardio-style ECG visualization.

**Architecture:** Single-page Gradio app with two modules — `ecg_plot.py` for matplotlib ECG paper rendering and `app.py` for the Gradio interface. Models (ECGDigitiser + ECGDiagnoser) load once at startup. Image upload triggers the full pipeline: digitize → diagnose → plot → display.

**Tech Stack:** Gradio 4.x, matplotlib (already installed), existing ECGDigitiser + ECGDiagnoser classes.

---

### Task 1: Install Gradio

**Step 1: Install the web optional dependency**

Run: `cd "/Users/sergenunlu/Developer/active/apps/Corio ECG" && source .venv/bin/activate && pip install -e ".[web]"`

**Step 2: Verify installation**

Run: `python -c "import gradio; print(gradio.__version__)"`
Expected: `4.x.x` (any 4+ version)

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: install gradio for test UI"
```

---

### Task 2: Create ECG Paper Plot Module

**Files:**
- Create: `src/web/ecg_plot.py`

**Step 1: Write the ECG paper grid plotter**

Creates a matplotlib figure that mimics standard ECG paper:
- Pink/red grid (25mm major, 5mm minor boxes)
- 4×3 layout (I/aVR/V1/V4, II/aVL/V2/V5, III/aVF/V3/V6)
- Lead II rhythm strip at bottom
- Lead labels in blue
- "Paper speed: 25 mm/s, Voltage gain: 10 mm/mV" footer

```python
# ECG paper-style visualization for digitized 12-lead signals
# Renders signals on a medical-standard grid (25mm/s, 10mm/mV)
# Layout: 4 columns × 3 rows + rhythm strip (Lead II)

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from src.utils.ecg_labels import LEAD_NAMES

# Standard ECG paper settings
PAPER_SPEED_MM_S: float = 25.0   # 25 mm per second
VOLTAGE_GAIN_MM_MV: float = 10.0  # 10 mm per mV
SAMPLE_RATE: int = 500

# Grid layout: standard 12-lead arrangement
# Each column shows 2.5 seconds of a different lead
GRID_LAYOUT: list[list[int]] = [
    [0, 3, 6, 9],     # I, aVR, V1, V4
    [1, 4, 7, 10],    # II, aVL, V2, V5
    [2, 5, 8, 11],    # III, aVF, V3, V6
]

# Seconds per column (2.5s each, 4 columns = 10s total)
SECONDS_PER_COLUMN: float = 2.5
SAMPLES_PER_COLUMN: int = int(SECONDS_PER_COLUMN * SAMPLE_RATE)

# Colors matching real ECG paper
COLOR_GRID_MAJOR: str = "#E8B4B4"   # Dark pink for 5mm boxes
COLOR_GRID_MINOR: str = "#F2DCDC"   # Light pink for 1mm boxes
COLOR_SIGNAL: str = "#1A1A2E"       # Dark navy for signal trace
COLOR_LABEL: str = "#2563EB"        # Blue for lead labels
COLOR_BG: str = "#FDF6F0"           # Warm off-white paper background


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
    # Denormalize z-score back to approximate mV for display
    # Z-score normalized signals have mean~0, std~1
    # We scale to reasonable ECG amplitude range for visualization
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

            # X offset for each column (in samples)
            x = np.arange(len(segment)) + (col_idx * SAMPLES_PER_COLUMN)
            # Y offset: center each row
            y = segment

            ax.plot(x, y, color=COLOR_SIGNAL, linewidth=0.7)

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
    rhythm_signal = display_signal[1, :5000]  # Lead II
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

    # Hide axis labels (ECG paper doesn't show them)
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
```

**Step 2: Run quick smoke test**

Run: `cd "/Users/sergenunlu/Developer/active/apps/Corio ECG" && source .venv/bin/activate && python -c "
from src.web.ecg_plot import plot_ecg_paper, fig_to_pil
import numpy as np
sig = np.random.randn(12, 5000).astype(np.float32)
fig = plot_ecg_paper(sig)
img = fig_to_pil(fig)
print(f'Plot size: {img.size}')
"`
Expected: `Plot size: (XXXX, XXXX)` — any valid dimensions, no errors.

**Step 3: Commit**

```bash
git add src/web/ecg_plot.py
git commit -m "feat: add ECG paper-style plot renderer for digitized signals"
```

---

### Task 3: Create Gradio App

**Files:**
- Create: `src/web/app.py`
- Modify: `src/web/__init__.py` (add docstring)

**Step 1: Write the Gradio application**

```python
# Gradio web UI for testing the Corio ECG pipeline
# Upload a paper ECG photo → digitize → diagnose → display results
# Run: python -m src.web.app

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from src.pipeline.diagnose import DiagnosisResult, ECGDiagnoser
from src.pipeline.digitize import ECGDigitiser
from src.utils.ecg_labels import CRITICAL_DIAGNOSIS_INDICES, ECG_FOUNDER_LABELS
from src.web.ecg_plot import fig_to_pil, plot_ecg_paper

logger = logging.getLogger(__name__)

MODEL_PATH: Path = Path("models/ecgfounder/base/12_lead_ECGFounder.pth")

# Lazy-loaded global instances (loaded once on first request)
_digitiser: ECGDigitiser | None = None
_diagnoser: ECGDiagnoser | None = None


def _get_digitiser() -> ECGDigitiser:
    """Get or create the digitiser singleton."""
    global _digitiser
    if _digitiser is None:
        logger.info("Loading ECGDigitiser (first request)...")
        _digitiser = ECGDigitiser()
    return _digitiser


def _get_diagnoser() -> ECGDiagnoser:
    """Get or create the diagnoser singleton."""
    global _diagnoser
    if _diagnoser is None:
        logger.info("Loading ECGDiagnoser (first request)...")
        _diagnoser = ECGDiagnoser(checkpoint_path=MODEL_PATH)
    return _diagnoser


def analyze_ecg(
    image: Image.Image | None,
    threshold: float,
) -> tuple[Image.Image | None, str, str]:
    """Full pipeline: image → digitize → diagnose → display.

    Args:
        image: Uploaded ECG photo (PIL Image from Gradio).
        threshold: Probability cutoff for diagnosis filtering.

    Returns:
        Tuple of (ecg_visualization, critical_findings_html, all_diagnoses_html).
    """
    if image is None:
        return None, _info_html("Upload an ECG image to begin analysis."), ""

    # Save uploaded image to temp file (digitiser needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp, format="PNG")
        tmp_path = Path(tmp.name)

    try:
        # Step 1: Digitize image to signal
        digitiser = _get_digitiser()
        signal = digitiser.digitize(tmp_path)

        # Step 2: Diagnose signal
        diagnoser = _get_diagnoser()
        results = diagnoser.diagnose(signal, threshold=threshold)
        all_results = diagnoser.diagnose_all(signal)

        # Step 3: Generate ECG paper visualization
        fig = plot_ecg_paper(signal)
        ecg_image = fig_to_pil(fig)

        # Step 4: Format results
        critical_html = _format_critical(results)
        diagnoses_html = _format_diagnoses(all_results, threshold)

        return ecg_image, critical_html, diagnoses_html

    except RuntimeError as exc:
        error_msg = (
            f"<div style='padding:16px; background:#FEF2F2; border-radius:8px; "
            f"border-left:4px solid #EF4444;'>"
            f"<b>Analysis Failed</b><br>{exc}</div>"
        )
        return None, error_msg, ""
    finally:
        tmp_path.unlink(missing_ok=True)


def _format_critical(results: list[DiagnosisResult]) -> str:
    """Format critical findings as HTML alert boxes."""
    critical = [r for r in results if r.index in CRITICAL_DIAGNOSIS_INDICES]

    if not critical:
        return (
            "<div style='padding:12px 16px; background:#F0FDF4; border-radius:8px; "
            "border-left:4px solid #22C55E; color:#166534;'>"
            "<b>No critical findings detected</b></div>"
        )

    items = []
    for r in critical:
        items.append(
            f"<div style='padding:12px 16px; margin-bottom:8px; background:#FEF2F2; "
            f"border-radius:8px; border-left:4px solid #EF4444;'>"
            f"<span style='color:#DC2626; font-size:18px;'>&#9888;</span> "
            f"<b style='color:#991B1B;'>{r.label}</b>"
            f"<span style='float:right; color:#666; font-size:14px;'>"
            f"prob: {r.probability:.2f}</span></div>"
        )
    return "\n".join(items)


def _format_diagnoses(
    all_results: list[DiagnosisResult],
    threshold: float,
) -> str:
    """Format all diagnoses as an HTML table with probability bars."""
    # Show top 20 diagnoses (sorted by probability descending)
    top = sorted(all_results, key=lambda r: r.probability, reverse=True)[:20]

    rows = []
    for r in top:
        bar_width = int(r.probability * 100)
        is_above = r.probability >= threshold
        is_critical = r.index in CRITICAL_DIAGNOSIS_INDICES

        # Color based on status
        if is_critical and is_above:
            bar_color = "#EF4444"
            text_weight = "bold"
        elif is_above:
            bar_color = "#3B82F6"
            text_weight = "bold"
        else:
            bar_color = "#D1D5DB"
            text_weight = "normal"

        rows.append(
            f"<tr>"
            f"<td style='padding:6px 8px; font-weight:{text_weight};'>{r.label}</td>"
            f"<td style='padding:6px 8px; width:200px;'>"
            f"<div style='background:#F3F4F6; border-radius:4px; overflow:hidden;'>"
            f"<div style='background:{bar_color}; height:20px; width:{bar_width}%; "
            f"border-radius:4px; transition:width 0.3s;'></div></div></td>"
            f"<td style='padding:6px 8px; text-align:right; font-family:monospace;'>"
            f"{r.probability:.3f}</td>"
            f"</tr>"
        )

    return (
        f"<div style='margin-top:8px;'>"
        f"<table style='width:100%; border-collapse:collapse; font-size:14px;'>"
        f"<thead><tr style='border-bottom:2px solid #E5E7EB;'>"
        f"<th style='text-align:left; padding:8px;'>Diagnosis</th>"
        f"<th style='text-align:left; padding:8px;'>Probability</th>"
        f"<th style='text-align:right; padding:8px;'>Value</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table></div>"
    )


def _info_html(message: str) -> str:
    """Wrap a message in a styled info box."""
    return (
        f"<div style='padding:16px; background:#EFF6FF; border-radius:8px; "
        f"border-left:4px solid #3B82F6; color:#1E40AF;'>{message}</div>"
    )


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(
        title="Corio ECG — AI ECG Interpreter",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# 🫀 Corio ECG — AI ECG Interpreter\n"
            "> *For Educational and Research Use Only*\n\n"
            "Upload a paper ECG photograph to get AI-powered diagnosis."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload ECG Image",
                    type="pil",
                    height=300,
                )
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Diagnosis Threshold",
                    info="Higher = fewer but more confident diagnoses",
                )
                analyze_btn = gr.Button(
                    "Analyze ECG",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Visualization"):
                        ecg_output = gr.Image(
                            label="Digitized ECG Signal",
                            type="pil",
                        )
                    with gr.TabItem("Original"):
                        original_output = gr.Image(
                            label="Original Upload",
                            type="pil",
                        )

        gr.Markdown("### Findings")
        critical_output = gr.HTML(
            value=_info_html("Upload an ECG image and click Analyze."),
        )

        gr.Markdown("### All Diagnoses (Top 20)")
        diagnoses_output = gr.HTML()

        # Wire up the analyze button
        analyze_btn.click(
            fn=analyze_ecg,
            inputs=[image_input, threshold_slider],
            outputs=[ecg_output, critical_output, diagnoses_output],
        )

        # Also mirror uploaded image to Original tab
        image_input.change(
            fn=lambda img: img,
            inputs=[image_input],
            outputs=[original_output],
        )

    return app


def main() -> None:
    """Launch the Gradio app."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
```

**Step 2: Update `src/web/__init__.py`**

```python
# Corio ECG web UI — Gradio-based test interface
```

**Step 3: Test the app launches (press Ctrl+C after verifying)**

Run: `cd "/Users/sergenunlu/Developer/active/apps/Corio ECG" && source .venv/bin/activate && timeout 15 python -m src.web.app 2>&1 || true`
Expected: Should show "Running on http://0.0.0.0:7860" (or similar), then timeout.

**Step 4: Commit**

```bash
git add src/web/app.py src/web/__init__.py
git commit -m "feat: add Gradio test UI for ECG photo analysis"
```

---

### Task 4: Smoke Test Full Pipeline via UI

**Step 1: Launch the app**

Run: `cd "/Users/sergenunlu/Developer/active/apps/Corio ECG" && source .venv/bin/activate && python -m src.web.app`

**Step 2: Manual test**

1. Open http://localhost:7860 in browser
2. Upload a paper ECG image (from the PTB-XL synthetic images or a real photo)
3. Click "Analyze ECG"
4. Verify: ECG visualization appears, diagnoses show in table, critical findings highlighted if any

**Step 3: Fix any issues found during manual testing**

Iterate until the pipeline works end-to-end.

**Step 4: Final commit**

```bash
git commit -m "fix: address issues found during UI smoke testing"
```

---

### Summary

| Task | What | Est. |
|------|------|------|
| 1 | Install Gradio | 2 min |
| 2 | ECG paper plot module | 5 min |
| 3 | Gradio app | 5 min |
| 4 | Smoke test | 5 min |

**Total: 4 tasks, ~17 min**
