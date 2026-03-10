# Gradio web UI for testing the Corio ECG pipeline
# Upload a paper ECG photo -> digitize -> diagnose -> display results
# Run: python -m src.web.app

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from src.pipeline.diagnose import DiagnosisResult, ECGDiagnoser
from src.pipeline.digitize import DigitizeInfo, ECGDigitiser
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


LAYOUT_CHOICES: list[str] = [
    "Auto-detect",
    "3x4 (standard)",
    "3x4+1R (with rhythm strip)",
    "6x2",
]

# Map UI labels to layout_should_include_substring values
_LAYOUT_MAP: dict[str, str | None] = {
    "Auto-detect": None,
    "3x4 (standard)": "3x4",
    "3x4+1R (with rhythm strip)": "3x4+1R",
    "6x2": "6x2",
}


def analyze_ecg(
    image: Image.Image | None,
    threshold: float,
    layout_choice: str,
) -> tuple[Image.Image | None, str, str, str]:
    """Full pipeline: image -> digitize -> diagnose -> display.

    Returns:
        Tuple of (ecg_visualization, critical_findings_html,
                  all_diagnoses_html, debug_info_html).
    """
    import time as _time

    if image is None:
        return None, _info_html("Upload an ECG image to begin analysis."), "", ""

    layout_hint = _LAYOUT_MAP.get(layout_choice)
    total_start = _time.time()

    # Save uploaded image to temp file (digitiser needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp, format="PNG")
        tmp_path = Path(tmp.name)

    try:
        # Step 1: Digitize image to signal
        digitiser = _get_digitiser()
        signal = digitiser.digitize(tmp_path, layout_hint=layout_hint)
        debug_info = digitiser.last_info

        # Step 2: Diagnose signal
        diagnoser = _get_diagnoser()
        results = diagnoser.diagnose(signal, threshold=threshold)
        all_results = diagnoser.diagnose_all(signal)

        # Step 3: Generate ECG paper visualization
        fig = plot_ecg_paper(signal)
        ecg_image = fig_to_pil(fig)

        # Step 4: Collect timing from the wrapper
        wrapper_times = getattr(digitiser._wrapper, "times", {})
        wrapper_times["Total (end-to-end)"] = _time.time() - total_start

        # Step 5: Format results
        critical_html = _format_critical(results)
        diagnoses_html = _format_diagnoses(all_results, threshold)
        debug_html = _format_debug_info(debug_info, wrapper_times)

        return ecg_image, critical_html, diagnoses_html, debug_html

    except RuntimeError as exc:
        error_msg = (
            "<div style='padding:16px; background:#FEF2F2; border-radius:8px; "
            f"border-left:4px solid #EF4444;'>"
            f"<b>Analysis Failed</b><br>{exc}</div>"
        )
        return None, error_msg, "", ""
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
            "<div style='padding:12px 16px; margin-bottom:8px; background:#FEF2F2; "
            "border-radius:8px; border-left:4px solid #EF4444;'>"
            "<span style='color:#DC2626; font-size:18px;'>&#9888;</span> "
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
    top = sorted(all_results, key=lambda r: r.probability, reverse=True)[:20]

    rows = []
    for r in top:
        bar_width = int(r.probability * 100)
        is_above = r.probability >= threshold
        is_critical = r.index in CRITICAL_DIAGNOSIS_INDICES

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
        "<div style='margin-top:8px;'>"
        "<table style='width:100%; border-collapse:collapse; font-size:14px;'>"
        "<thead><tr style='border-bottom:2px solid #E5E7EB;'>"
        "<th style='text-align:left; padding:8px;'>Diagnosis</th>"
        "<th style='text-align:left; padding:8px;'>Probability</th>"
        "<th style='text-align:right; padding:8px;'>Value</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _format_debug_info(
    info: DigitizeInfo,
    timing: dict[str, float] | None = None,
) -> str:
    """Format diagnostic info as HTML for the debug panel."""
    # Quality indicator based on layout cost
    if info.layout_cost < 0.5:
        quality_badge = (
            "<span style='background:#22C55E; color:white; padding:2px 8px; "
            "border-radius:4px; font-size:12px;'>GOOD</span>"
        )
    elif info.layout_cost < 1.5:
        quality_badge = (
            "<span style='background:#F59E0B; color:white; padding:2px 8px; "
            "border-radius:4px; font-size:12px;'>FAIR</span>"
        )
    else:
        quality_badge = (
            "<span style='background:#EF4444; color:white; padding:2px 8px; "
            "border-radius:4px; font-size:12px;'>POOR</span>"
        )

    # Warnings
    warnings = []
    if info.layout_cost > 1.0:
        warnings.append("Layout matching cost is high — leads may be misassigned")
    if info.detected_leads_count < 4:
        warnings.append(
            f"Only {info.detected_leads_count} lead labels detected "
            "(expected 8+) — U-Net may have failed on this image"
        )
    if info.avg_pixel_per_mm < 1.0 and info.avg_pixel_per_mm > 0:
        warnings.append(
            f"Very low pixel density ({info.avg_pixel_per_mm:.1f} px/mm) "
            "— grid detection may have failed, causing amplitude distortion"
        )
    if info.avg_pixel_per_mm > 30.0:
        warnings.append(
            f"Very high pixel density ({info.avg_pixel_per_mm:.1f} px/mm) "
            "— grid detection may have failed"
        )
    if info.raw_lines_count < 3:
        warnings.append(
            f"Only {info.raw_lines_count} signal traces extracted "
            "(expected 4+) — signal segmentation may have failed"
        )

    warnings_html = ""
    if warnings:
        items = "".join(
            f"<div style='padding:6px 10px; margin-bottom:4px; background:#FEF3C7; "
            f"border-radius:4px; border-left:3px solid #F59E0B; font-size:13px;'>"
            f"&#9888; {w}</div>"
            for w in warnings
        )
        warnings_html = f"<div style='margin-bottom:12px;'>{items}</div>"

    # Per-lead energy table
    lead_names = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ]
    energy_cells = ""
    for i, energy in enumerate(info.per_lead_energy):
        name = lead_names[i] if i < len(lead_names) else f"L{i}"
        # Flag leads with very low or very high energy
        if energy < 0.05:
            color = "#EF4444"  # red — nearly flat/dead lead
        elif energy > 3.0:
            color = "#F59E0B"  # amber — possibly distorted
        else:
            color = "#166534"  # green — normal
        energy_cells += (
            f"<td style='padding:4px 8px; text-align:center;'>"
            f"<b>{name}</b><br>"
            f"<span style='color:{color}; font-family:monospace;'>{energy:.2f}</span>"
            f"</td>"
        )

    # Timing breakdown
    timing_html = ""
    if timing:
        total = sum(timing.values())
        timing_rows = "".join(
            f"<tr><td style='padding:2px 8px;'>{name}</td>"
            f"<td style='padding:2px 8px; text-align:right; font-family:monospace;'>"
            f"{duration:.1f}s</td></tr>"
            for name, duration in timing.items()
        )
        timing_html = (
            f"<div style='margin-top:12px;'><b>Timing breakdown:</b>"
            f"<table style='border-collapse:collapse; font-size:12px; margin-top:4px;'>"
            f"{timing_rows}"
            f"<tr style='border-top:1px solid #CBD5E1; font-weight:bold;'>"
            f"<td style='padding:4px 8px;'>Total</td>"
            f"<td style='padding:4px 8px; text-align:right; font-family:monospace;'>"
            f"{total:.1f}s</td></tr></table></div>"
        )

    return (
        f"{warnings_html}"
        f"<div style='background:#F8FAFC; border:1px solid #E2E8F0; "
        f"border-radius:8px; padding:16px; font-size:13px;'>"
        f"<div style='display:flex; gap:24px; flex-wrap:wrap; margin-bottom:12px;'>"
        f"<div><b>Quality:</b> {quality_badge}</div>"
        f"<div><b>Image:</b> {info.image_size[1]}x{info.image_size[0]} px</div>"
        f"<div><b>Layout:</b> {info.layout_name}</div>"
        f"<div><b>Layout cost:</b> {info.layout_cost:.2f}</div>"
        f"<div><b>Flipped:</b> {'Yes' if info.layout_flipped else 'No'}</div>"
        f"<div><b>Signal traces:</b> {info.raw_lines_count}</div>"
        f"<div><b>Detected leads:</b> {info.detected_leads_count}/12</div>"
        f"<div><b>Pixel density:</b> {info.avg_pixel_per_mm:.1f} px/mm</div>"
        f"</div>"
        f"<div style='margin-bottom:8px;'><b>Detected:</b> "
        f"{', '.join(info.detected_leads) if info.detected_leads else '<i>none</i>'}</div>"
        f"<div><b>Per-lead energy (RMS after z-score):</b></div>"
        f"<table style='width:100%; border-collapse:collapse; margin-top:4px;'>"
        f"<tr>{energy_cells}</tr></table>"
        f"{timing_html}"
        f"</div>"
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
    ) as app:
        gr.Markdown(
            "# Corio ECG — AI ECG Interpreter\n"
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
                layout_dropdown = gr.Dropdown(
                    choices=LAYOUT_CHOICES,
                    value="Auto-detect",
                    label="ECG Layout",
                    info="Select layout format or let the AI auto-detect",
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

        with gr.Accordion("Digitization Debug Info", open=False):
            debug_output = gr.HTML(
                value=_info_html("Debug info will appear after analysis."),
            )

        # Example images for quick testing
        example_dir = Path("data/processed/images")
        example_images = []
        for category in ["clean", "moderate", "hard"]:
            cat_dir = example_dir / category
            if cat_dir.exists():
                imgs = sorted(cat_dir.glob("*.png"))[:3]
                for img in imgs:
                    example_images.append([str(img), 0.5, "Auto-detect"])

        if example_images:
            gr.Examples(
                examples=example_images,
                inputs=[image_input, threshold_slider, layout_dropdown],
                label="Sample ECG Images",
            )

        # Wire up the analyze button
        analyze_btn.click(
            fn=analyze_ecg,
            inputs=[image_input, threshold_slider, layout_dropdown],
            outputs=[ecg_output, critical_output, diagnoses_output, debug_output],
        )

        # Mirror uploaded image to Original tab
        image_input.change(
            fn=lambda img: img,
            inputs=[image_input],
            outputs=[original_output],
        )

    return app


def main() -> None:
    """Launch the Gradio app.

    Pass --share to create a public Gradio tunnel (useful for VPS access).
    """
    import argparse

    parser = argparse.ArgumentParser(description="Corio ECG Web UI")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
