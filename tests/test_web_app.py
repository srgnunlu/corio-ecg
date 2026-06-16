from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline.digitize import ECGDigitiser
from src.web.ecg_plot import fig_to_pil, plot_ecg_paper


class TestDigitizePayloadTrim:
    def test_trim_debug_payload_removes_large_debug_keys(self) -> None:
        result = {
            "input_image": object(),
            "aligned": {"image": object()},
            "signal": {"canonical_lines": object()},
            "layout_name": "3x4+1R",
        }

        ECGDigitiser._trim_debug_payload(result)

        assert "input_image" not in result
        assert "aligned" not in result
        assert result["layout_name"] == "3x4+1R"


class TestResolveSegmentLayout:
    def test_prefers_explicit_user_layout_choice(self) -> None:
        from src.web.app import _resolve_segment_layout

        assert _resolve_segment_layout("3x4+1R (standard)", "unknown") == "3x4"

    def test_falls_back_to_detected_layout_when_auto_detect(self) -> None:
        from src.web.app import _resolve_segment_layout

        assert _resolve_segment_layout("Auto-detect", "standard_6x2+1R") == "6x2"

    def test_returns_none_when_layout_is_unresolvable(self) -> None:
        from src.web.app import _resolve_segment_layout

        assert _resolve_segment_layout("Auto-detect", "unknown") is None


class TestEcgPlot:
    def test_fig_to_pil_returns_loaded_image(self) -> None:
        signal = np.zeros((12, 5000), dtype=np.float32)
        fig = plot_ecg_paper(signal)

        image = fig_to_pil(fig)

        assert isinstance(image, Image.Image)
        assert image.size[0] > 0
        assert image.size[1] > 0
        image.getbands()


class TestAnalyzeNoImage:
    def test_returns_empty_context_and_no_image_when_no_upload(self) -> None:
        from src.web.app import _EMPTY_CONTEXT, analyze_ecg

        result = analyze_ecg(None, 0.7, "Auto-detect")

        assert len(result) == 6
        ecg_image, _, _, _, context, original = result
        assert ecg_image is None
        assert original is None
        assert context is _EMPTY_CONTEXT


class TestGeneratePdf:
    def test_returns_none_without_a_report(self) -> None:
        from src.web.app import _EMPTY_CONTEXT, generate_pdf

        assert generate_pdf(_EMPTY_CONTEXT) is None
        assert generate_pdf(None) is None

    def test_writes_pdf_for_a_valid_context(self) -> None:
        from src.report.pdf_report import build_pdf_report
        from src.report.structured_report import DiagnosisEntry, ECGReport
        from src.web.app import AnalysisContext, generate_pdf

        report = ECGReport(
            heart_rate_bpm=72.0,
            rhythm_classification="Sinus rhythm",
            rhythm_basis="sinus",
            rate_category="normal",
            ectopy_present=False,
            pvc_count=0,
            pr_ms=150.0,
            qrs_ms=90.0,
            qt_ms=380.0,
            qtc_ms=400.0,
            qtc_formula="bazett",
            interval_flags={"pr": "normal", "qrs": "normal", "qtc": "normal"},
            top_diagnoses=[
                DiagnosisEntry(label="NORMAL ECG", probability=0.91, above_threshold=True),
            ],
            is_normal=True,
            overall_assessment="Normal ECG",
            abnormal_reasons=[],
            normal_criteria={"sinus_rhythm": True},
        )
        context = AnalysisContext(
            report=report,
            intervals=None,
            original_image=Image.new("RGB", (200, 150), color=(200, 200, 200)),
            signal_image=Image.new("RGB", (400, 300), color=(250, 246, 240)),
        )

        path = generate_pdf(context)

        assert path is not None
        pdf = Path(path)
        assert pdf.exists()
        assert pdf.read_bytes()[:4] == b"%PDF"
        # the helper builds straight to disk too
        assert build_pdf_report is not None
