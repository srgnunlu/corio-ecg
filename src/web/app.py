# Gradio web UI for testing the Corio ECG pipeline
# Upload a paper ECG photo -> digitize -> diagnose -> display results
# Run: python -m src.web.app

from __future__ import annotations

import gc
import logging
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
import torch

from src.measurement.intervals import IntervalMeasurements, interpret_intervals
from src.measurement.rhythm_analysis import analyze_rhythm
from src.pipeline.diagnose import DiagnosisResult, ECGDiagnoser
from src.pipeline.digitize import DigitizeInfo, ECGDigitiser
from src.report.structured_report import ECGReport, build_report
from src.utils.ecg_labels import CRITICAL_DIAGNOSIS_INDICES, ECG_FOUNDER_LABELS
from src.web.ecg_plot import fig_to_pil, plot_ecg_paper

logger = logging.getLogger(__name__)

MODEL_PATH: Path = Path("models/ecgfounder/base/12_lead_ECGFounder.pth")

# Lazy-loaded global instances (loaded once on first request)
_digitiser: ECGDigitiser | None = None
_diagnoser: ECGDiagnoser | None = None


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _get_digitiser() -> ECGDigitiser:
    """Get or create the digitiser singleton."""
    global _digitiser
    if _digitiser is None:
        logger.info("Loading ECGDigitiser (first request)...")
        _digitiser = ECGDigitiser(
            # Dewarping retry runs the entire pipeline twice, doubling peak
            # memory (15+ GB extra). Disabled by default on 24 GB Macs.
            enable_dewarping_retry=_env_flag("CORIO_GRADIO_ENABLE_DEWARP_RETRY", False),
        )
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
    "3x4+1R (standard)",
    "3x4+3R",
    "6x2",
    "6x2+1R",
]

# Map UI labels to layout_should_include_substring values
_LAYOUT_MAP: dict[str, str | None] = {
    "Auto-detect": None,
    "3x4+1R (standard)": "3x4+1R",
    "3x4+3R": "3x4+3R",
    "6x2": "standard_6x2",
    "6x2+1R": "standard_6x2+1R",
}

# Segment-ensemble diagnosis is the production default: it diagnoses each
# printed paper column independently and averages, recovering the accuracy
# lost when all 12 (temporally misaligned, tiled) leads are fed at once
# (PTB-XL round-trip macro AUROC ~0.75 tiled -> ~0.87 segment-ensemble).
# Set CORIO_SEGMENT_ENSEMBLE=0 to fall back to the single-pass tiled path.
SEGMENT_ENSEMBLE_ENABLED: bool = _env_flag("CORIO_SEGMENT_ENSEMBLE", True)


def _resolve_segment_layout(
    layout_choice: str,
    detected_layout_name: str | None,
) -> str | None:
    """Resolve a segment-ensemble column layout ("3x4"/"6x2") or None.

    Prefers the user's explicit layout choice, then the digitizer's detected
    layout. Returns None when neither resolves to a supported column layout,
    signalling the caller to fall back to the tiled single-pass path.
    """
    hint = _LAYOUT_MAP.get(layout_choice)
    for candidate in (hint, detected_layout_name):
        if not candidate:
            continue
        if "3x4" in candidate:
            return "3x4"
        if "6x2" in candidate:
            return "6x2"
    return None


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
    timing_breakdown: dict[str, float] = {}

    # Save uploaded image to temp file (digitiser needs a file path)
    save_start = _time.time()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp, format="PNG")
        tmp_path = Path(tmp.name)
    timing_breakdown["Save upload"] = _time.time() - save_start
    response: tuple[Image.Image | None, str, str, str]

    try:
        # Step 1: Digitize image to signal
        digitiser_load_start = _time.time()
        digitiser_was_loaded = _digitiser is not None
        digitiser = _get_digitiser()
        if not digitiser_was_loaded:
            timing_breakdown["Load digitizer"] = _time.time() - digitiser_load_start

        digitize_start = _time.time()
        signal = digitiser.digitize(tmp_path, layout_hint=layout_hint)
        timing_breakdown["Digitization"] = _time.time() - digitize_start
        debug_info = digitiser.last_info

        # Step 2: Diagnose signal
        diagnoser_load_start = _time.time()
        diagnoser_was_loaded = _diagnoser is not None
        diagnoser = _get_diagnoser()
        if not diagnoser_was_loaded:
            timing_breakdown["Load diagnoser"] = _time.time() - diagnoser_load_start

        diagnose_start = _time.time()
        segment_layout = (
            _resolve_segment_layout(layout_choice, debug_info.layout_name)
            if SEGMENT_ENSEMBLE_ENABLED
            else None
        )
        # Heart rate comes from the uncropped full-duration rhythm strip when
        # the layout printed one — the tiled diagnosis signal lost the real RR.
        rhythm_strip = debug_info.rhythm_strip
        if segment_layout is not None:
            all_results = diagnoser.diagnose_all_segment_ensemble(
                signal, layout=segment_layout, rhythm_strip=rhythm_strip
            )
        else:
            # Tiled single-pass fallback (auto-detect / unsupported layouts).
            all_results = diagnoser.diagnose_all(signal, rhythm_strip=rhythm_strip)
        timing_breakdown["Diagnosis inference"] = _time.time() - diagnose_start
        results = [r for r in all_results if r.probability >= threshold]
        estimated_hr = diagnoser.last_estimated_hr_bpm
        intervals = diagnoser.last_interval_measurements

        # Rhythm classification + structured report (Phase C.3). Best-effort:
        # rhythm/report assembly must never break the core diagnosis path.
        try:
            rhythm = analyze_rhythm(signal, rhythm_strip=rhythm_strip)
        except Exception:  # noqa: BLE001
            logger.exception("Rhythm analysis failed")
            rhythm = None
        try:
            report = build_report(
                all_results,
                intervals,
                rhythm,
                threshold=threshold,
                estimated_hr_bpm=estimated_hr,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Report assembly failed")
            report = None

        # Step 3: Generate ECG paper visualization
        plot_start = _time.time()
        fig = plot_ecg_paper(signal)
        ecg_image = fig_to_pil(fig)
        timing_breakdown["Plot rendering"] = _time.time() - plot_start

        # Step 4: Collect timing from the wrapper
        wrapper_times = dict(getattr(digitiser._wrapper, "times", {}))
        wrapper_times.update(timing_breakdown)
        wrapper_times["Total (end-to-end)"] = _time.time() - total_start

        # Step 5: Format results
        critical_html = (
            _format_report_headline(report)
            + _format_intervals(intervals)
            + _format_critical(
                results=results,
                all_results=all_results,
                threshold=threshold,
                estimated_hr=estimated_hr,
            )
        )
        diagnoses_html = _format_diagnoses(all_results, threshold)
        debug_html = _format_debug_info(debug_info, wrapper_times, estimated_hr)

        response = (ecg_image, critical_html, diagnoses_html, debug_html)
        del signal, all_results, results, debug_info, wrapper_times, fig
        return response

    except RuntimeError as exc:
        error_msg = (
            "<div style='padding:16px; background:#FEF2F2; border-radius:8px; "
            f"border-left:4px solid #EF4444;'>"
            f"<b>Analysis Failed</b><br>{exc}</div>"
        )
        logger.exception("ECG analysis failed")
        response = (None, error_msg, "", "")
        return response
    except Exception:
        logger.exception("Unexpected exception during ECG analysis")
        response = (
            None,
            (
                "<div style='padding:16px; background:#FEF2F2; border-radius:8px; "
                "border-left:4px solid #EF4444;'>"
                "<b>Analysis Failed</b><br>Unexpected internal error. "
                "Check /tmp/corio-gradio.log for details.</div>"
            ),
            "",
            "",
        )
        return response
    finally:
        tmp_path.unlink(missing_ok=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()


def _format_report_headline(report: ECGReport | None) -> str:
    """Render the report headline: overall verdict + rhythm line + HR.

    This is the clinician-facing TLDR that sits above the detailed cards: is the
    ECG normal or not, what is the rhythm, and at what rate.
    """
    if report is None:
        return ""

    palette = {
        "Normal ECG": ("#166534", "#F0FDF4", "#22C55E"),
        "Abnormal ECG": ("#991B1B", "#FEF2F2", "#EF4444"),
        "Indeterminate ECG": ("#92400E", "#FFFBEB", "#F59E0B"),
    }
    fg, bg, accent = palette.get(report.overall_assessment, ("#1E3A8A", "#F8FAFC", "#3B82F6"))

    hr_text = f"{report.heart_rate_bpm:.0f} bpm" if report.heart_rate_bpm is not None else "n/a"
    rhythm_line = (
        f"<b>Rhythm:</b> {report.rhythm_classification} &nbsp;·&nbsp; "
        f"<b>HR:</b> {hr_text}"
    )
    if report.ectopy_present:
        rhythm_line += (
            f" &nbsp;·&nbsp; <span style='color:#B91C1C;'>"
            f"{report.pvc_count} PVC-like beat(s)</span>"
        )

    reasons_html = ""
    if report.abnormal_reasons:
        items = "".join(f"<li>{r}</li>" for r in report.abnormal_reasons)
        reasons_html = (
            "<ul style='margin:8px 0 0 0; padding-left:20px; font-size:13px; "
            f"color:{fg};'>{items}</ul>"
        )

    return (
        f"<div style='padding:14px 16px; margin-bottom:12px; background:{bg}; "
        f"border-radius:8px; border-left:4px solid {accent};'>"
        f"<div style='font-size:18px; font-weight:bold; color:{fg};'>"
        f"{report.overall_assessment}</div>"
        f"<div style='font-size:14px; color:#334155; margin-top:4px;'>{rhythm_line}</div>"
        f"{reasons_html}"
        "<div style='font-size:11px; color:#64748B; margin-top:8px;'>"
        "Decision support — final responsibility rests with the reviewing clinician."
        "</div></div>"
    )


def _format_intervals(measurements: IntervalMeasurements | None) -> str:
    """Render PR/QRS/QT/QTc as a measurement card with abnormality flags."""
    if measurements is None or measurements.quality == "unmeasurable":
        return (
            "<div style='padding:10px 14px; margin-bottom:12px; background:#F8FAFC; "
            "border-radius:8px; border-left:4px solid #94A3B8; color:#475569; "
            "font-size:13px;'><b>Intervals</b> — could not be measured on this signal."
            "</div>"
        )

    flags = interpret_intervals(measurements)
    abnormal_colors = {"normal": "#166534"}

    def cell(label: str, value: float | None, unit: str, flag_key: str) -> str:
        text = f"{value:.0f} {unit}" if value is not None else "n/a"
        flag = flags.get(flag_key, "")
        is_abnormal = bool(flag) and flag != "normal"
        color = "#B91C1C" if is_abnormal else abnormal_colors.get(flag, "#0F172A")
        note = (
            f"<div style='font-size:11px; color:{color}; margin-top:2px;'>{flag}</div>"
            if flag
            else ""
        )
        return (
            "<td style='padding:8px 12px; text-align:center; vertical-align:top;'>"
            f"<div style='font-size:12px; color:#64748B;'>{label}</div>"
            f"<div style='font-size:18px; font-weight:bold; color:{color}; "
            f"font-family:monospace;'>{text}</div>{note}</td>"
        )

    qtc = measurements.qtc_preferred_ms
    qtc_label = f"QTc<br><span style='font-size:10px;'>({measurements.qtc_formula})</span>"
    cells = (
        cell("PR", measurements.pr_ms, "ms", "pr")
        + cell("QRS", measurements.qrs_ms, "ms", "qrs")
        + cell("QT", measurements.qt_ms, "ms", "")
        + cell(qtc_label, qtc, "ms", "qtc")
    )

    qual_color = "#22C55E" if measurements.quality == "good" else "#F59E0B"
    footer = (
        f"Lead {measurements.measured_lead or '?'} · {measurements.n_beats} beats · "
        f"<span style='color:{qual_color};'>{measurements.quality} confidence</span> · "
        "research use — verify against the tracing"
    )

    return (
        "<div style='padding:12px 16px; margin-bottom:12px; background:#F8FAFC; "
        "border-radius:8px; border-left:4px solid #3B82F6;'>"
        "<b style='color:#1E3A8A;'>Interval Measurements</b>"
        "<table style='width:100%; border-collapse:collapse; margin-top:6px;'>"
        f"<tr>{cells}</tr></table>"
        f"<div style='font-size:11px; color:#64748B; margin-top:8px;'>{footer}</div>"
        "</div>"
    )


def _format_critical(
    results: list[DiagnosisResult],
    all_results: list[DiagnosisResult],
    threshold: float,
    estimated_hr: float | None,
) -> str:
    """Format critical findings as HTML alert boxes."""
    critical = [r for r in results if r.index in CRITICAL_DIAGNOSIS_INDICES]
    rhythm_alerts = _format_rhythm_considerations(all_results, threshold, estimated_hr)

    if not critical:
        return rhythm_alerts + (
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
    return rhythm_alerts + "\n".join(items)


def _format_rhythm_considerations(
    all_results: list[DiagnosisResult],
    threshold: float,
    estimated_hr: float | None,
) -> str:
    """Show rhythm-specific differential when HR is clearly abnormal."""
    if estimated_hr is None or estimated_hr < 110.0:
        return ""

    label_to_result = {result.label: result for result in all_results}
    if estimated_hr >= 150.0:
        labels = [
            "VENTRICULAR TACHYCARDIA",
            "WIDE QRS TACHYCARDIA",
            "SUPRAVENTRICULAR TACHYCARDIA",
            "SINUS TACHYCARDIA",
            "ATRIAL FLUTTER",
            "ATRIAL FIBRILLATION",
            "IDIOVENTRICULAR RHYTHM",
        ]
        title = "Tachyarrhythmia Considerations"
        accent = "#F59E0B"
        bg = "#FFF7ED"
        fg = "#9A3412"
        message = (
            f"Estimated heart rate is {estimated_hr:.0f} bpm. "
            "ECGFounder scores rhythm labels independently, so tachy subtypes can "
            "remain below the main display threshold on digitized signals."
        )
    else:
        labels = [
            "SINUS TACHYCARDIA",
            "SUPRAVENTRICULAR TACHYCARDIA",
            "ATRIAL FLUTTER",
            "ATRIAL FIBRILLATION",
        ]
        title = "Fast Rhythm Considerations"
        accent = "#3B82F6"
        bg = "#EFF6FF"
        fg = "#1D4ED8"
        message = (
            f"Estimated heart rate is {estimated_hr:.0f} bpm. "
            "Review rhythm-specific labels separately from the main thresholded list."
        )

    ranked = [
        label_to_result[label]
        for label in labels
        if label in label_to_result
    ]
    ranked.sort(key=lambda result: result.probability, reverse=True)
    ranked = ranked[:4]
    if not ranked:
        return ""

    rows = "".join(
        "<tr>"
        f"<td style='padding:4px 8px;'>{result.label}</td>"
        f"<td style='padding:4px 8px; text-align:right; font-family:monospace;'>{result.probability:.3f}</td>"
        f"<td style='padding:4px 8px; text-align:right;'>{'above threshold' if result.probability >= threshold else 'below threshold'}</td>"
        "</tr>"
        for result in ranked
    )

    return (
        f"<div style='padding:12px 16px; margin-bottom:12px; background:{bg}; "
        f"border-radius:8px; border-left:4px solid {accent}; color:{fg};'>"
        f"<b>{title}</b><br>{message}"
        f"<table style='width:100%; border-collapse:collapse; font-size:13px; margin-top:8px;'>"
        f"<thead><tr>"
        f"<th style='text-align:left; padding:4px 8px;'>Label</th>"
        f"<th style='text-align:right; padding:4px 8px;'>Probability</th>"
        f"<th style='text-align:right; padding:4px 8px;'>Threshold</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


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
    estimated_hr: float | None = None,
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
        total = timing.get("Total (end-to-end)")
        timing_rows = "".join(
            f"<tr><td style='padding:2px 8px;'>{name}</td>"
            f"<td style='padding:2px 8px; text-align:right; font-family:monospace;'>"
            f"{duration:.1f}s</td></tr>"
            for name, duration in timing.items()
            if name != "Total (end-to-end)"
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
        f"<div><b>Estimated HR:</b> "
        f"{f'{estimated_hr:.0f} bpm' if estimated_hr is not None else 'n/a'}</div>"
        f"<div><b>Einthoven:</b> {info.einthoven_score:.2f}"
        f"{'  ⚠️' if info.einthoven_score >= 0 and info.einthoven_score < 0.7 else ''}</div>"
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
                    value="3x4+1R (standard)",
                    label="ECG Layout",
                    info="Most standard paper ECGs are 3x4+1R; switch if your printout differs",
                )
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.7,
                    step=0.05,
                    label="Diagnosis Threshold",
                    info="For digitized photos, 0.65-0.75 usually gives cleaner results",
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
                    example_images.append([str(img), 0.7, "3x4+1R (standard)"])

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
