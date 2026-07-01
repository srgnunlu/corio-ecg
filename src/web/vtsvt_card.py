# HTML card for presenting VT/SVT wide-complex criteria audit in the web UI.

from __future__ import annotations

import html

from src.vtsvt.models import VTSVTAssessment


def format_vtsvt_card(assessment: VTSVTAssessment | None) -> str:
    """Render a compact audit card for in-scope wide-complex tachycardia criteria."""
    if assessment is None or not assessment.in_scope:
        return ""

    title = "Wide-complex tachycardia criteria"
    status = "VT supported" if assessment.supports_vt else "Indeterminate WCT"
    accent = "#DC2626" if assessment.supports_vt else "#F59E0B"
    fg = "#991B1B" if assessment.supports_vt else "#92400E"
    bg = "#FEF2F2" if assessment.supports_vt else "#FFFBEB"
    features = assessment.features
    hr = _fmt(features.heart_rate_bpm, "bpm", digits=0)
    qrs = _fmt(features.qrs_ms, "ms", digits=0)
    rs = _fmt(features.max_precordial_rs_interval_ms, "ms", digits=0)
    evidence = _items(assessment.evidence, fallback="No positive VT criteria")
    limitations = _items(assessment.limitations[:3], fallback="No listed limitations")

    return (
        f"<div style='padding:12px 16px; margin-bottom:12px; background:{bg}; "
        f"border-radius:8px; border-left:4px solid {accent}; color:{fg};'>"
        f"<div style='font-size:13px; font-weight:bold;'>{title}</div>"
        f"<div style='font-size:18px; font-weight:bold; margin-top:3px;'>{status}</div>"
        "<table style='width:100%; border-collapse:collapse; margin-top:8px; "
        "font-size:13px;'>"
        f"<tr><td><b>HR</b></td><td>{hr}</td><td><b>QRS</b></td><td>{qrs}</td></tr>"
        f"<tr><td><b>Max precordial RS</b></td><td>{rs}</td>"
        f"<td><b>Beats</b></td><td>{features.n_beats}</td></tr>"
        "</table>"
        f"<div style='font-size:12px; margin-top:8px;'><b>Evidence:</b> {evidence}</div>"
        f"<div style='font-size:11px; margin-top:6px; color:#64748B;'>"
        f"<b>Limitations:</b> {limitations}. This is not a standalone diagnosis."
        "</div></div>"
    )


def _fmt(value: float | None, unit: str, *, digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f} {unit}"


def _items(values: tuple[str, ...], *, fallback: str) -> str:
    if not values:
        return fallback
    return ", ".join(html.escape(value.replace("_", " ")) for value in values)
