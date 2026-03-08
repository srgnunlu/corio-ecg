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
    """Full pipeline: image -> digitize -> diagnose -> display.

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
            "<div style='padding:16px; background:#FEF2F2; border-radius:8px; "
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

        # Mirror uploaded image to Original tab
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
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
