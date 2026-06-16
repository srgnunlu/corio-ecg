# Tests for the Phase D.2 PDF report generator.
# Verify a valid PDF is written for the normal/abnormal/missing-image paths.

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image

from src.report.pdf_report import build_pdf_report, doc_width
from src.report.structured_report import DiagnosisEntry, ECGReport


def _make_report(*, assessment: str = "Abnormal ECG") -> ECGReport:
    return ECGReport(
        heart_rate_bpm=78.0,
        rhythm_classification="Sinus rhythm",
        rhythm_basis="sinus",
        rate_category="normal",
        ectopy_present=True,
        pvc_count=2,
        pr_ms=160.0,
        qrs_ms=92.0,
        qt_ms=380.0,
        qtc_ms=410.0,
        qtc_formula="bazett",
        interval_flags={"pr": "normal", "qrs": "normal", "qtc": "prolonged"},
        top_diagnoses=[
            DiagnosisEntry(label="ATRIAL FIBRILLATION", probability=0.82, above_threshold=True),
            DiagnosisEntry(label="SINUS RHYTHM", probability=0.31, above_threshold=False),
        ],
        is_normal=False,
        overall_assessment=assessment,
        abnormal_reasons=["AI-flagged finding(s): ATRIAL FIBRILLATION", "QTc: prolonged"],
        normal_criteria={"sinus_rhythm": True, "normal_qtc": False},
    )


def _is_pdf(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1000 and path.read_bytes()[:4] == b"%PDF"


def test_build_pdf_with_images(tmp_path: Path) -> None:
    out = tmp_path / "report.pdf"
    original = Image.new("RGB", (400, 300), color=(220, 220, 220))
    signal = Image.new("RGB", (800, 600), color=(250, 246, 240))

    result = build_pdf_report(
        _make_report(),
        output_path=out,
        original_image=original,
        signal_image=signal,
        generated_at=datetime(2026, 6, 17, 14, 30),
    )

    assert result == out
    assert _is_pdf(out)


def test_build_pdf_without_images(tmp_path: Path) -> None:
    out = tmp_path / "report_no_img.pdf"

    build_pdf_report(_make_report(assessment="Normal ECG"), output_path=out)

    assert _is_pdf(out)


def test_build_pdf_indeterminate_no_diagnoses(tmp_path: Path) -> None:
    out = tmp_path / "report_indeterminate.pdf"
    report = _make_report(assessment="Indeterminate ECG")
    report.top_diagnoses = []
    report.heart_rate_bpm = None

    build_pdf_report(report, output_path=out)

    assert _is_pdf(out)


def test_doc_width_positive() -> None:
    assert doc_width() > 0
