# PDF flowables for VT/SVT wide-complex criteria audit.

from __future__ import annotations

from reportlab.lib import colors  # type: ignore[import-untyped]
from reportlab.lib.pagesizes import A4  # type: ignore[import-untyped]
from reportlab.lib.units import mm  # type: ignore[import-untyped]
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from src.report.structured_report import ECGReport
from src.vtsvt.models import VTSVTAssessment

_DOC_WIDTH: float = float(A4[0] - (18 * mm) * 2)


def vtsvt_pdf_section(report: ECGReport, styles: dict) -> list:
    """Build PDF flowables for an in-scope VT/SVT criteria audit."""
    assessment = report.vtsvt_assessment
    if assessment is None or not assessment.in_scope:
        return []

    return [
        Paragraph("Wide-Complex Tachycardia Criteria", styles["section"]),
        _audit_table(assessment),
        Spacer(1, 4),
        Paragraph(
            "This deterministic criteria audit is not a standalone diagnosis; "
            "review against the original tracing and clinical context.",
            styles["caption"],
        ),
    ]


def _audit_table(assessment: VTSVTAssessment) -> Table:
    features = assessment.features
    rows = [
        ["Status", "VT supported" if assessment.supports_vt else "Indeterminate WCT"],
        ["Evidence", _join(assessment.evidence, "No positive VT criteria")],
        ["Limitations", _join(assessment.limitations[:3], "No listed limitations")],
        ["Heart rate", _fmt(features.heart_rate_bpm, "bpm")],
        ["QRS duration", _fmt(features.qrs_ms, "ms")],
        ["Max precordial RS", _fmt(features.max_precordial_rs_interval_ms, "ms")],
    ]
    table = Table(rows, colWidths=[_DOC_WIDTH * 0.28, _DOC_WIDTH * 0.72])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FEF2F2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#991B1B")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _fmt(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f} {unit}"


def _join(values: tuple[str, ...], fallback: str) -> str:
    if not values:
        return fallback
    return ", ".join(value.replace("_", " ") for value in values)
