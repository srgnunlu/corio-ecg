from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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

    def test_plot_uses_fixed_millivolt_grid_spacing(self) -> None:
        import matplotlib.pyplot as plt

        signal = np.zeros((12, 5000), dtype=np.float32)
        signal[1, 100:130] = 1.0
        fig = plot_ecg_paper(signal)
        ax = fig.axes[0]

        major_ticks = ax.yaxis.get_major_locator().tick_values(-1.0, 1.0)
        minor_ticks = ax.yaxis.get_minor_locator().tick_values(-1.0, 1.0)

        assert np.diff(major_ticks[:3]) == pytest.approx([0.5, 0.5])
        assert np.diff(minor_ticks[:3]) == pytest.approx([0.1, 0.1])
        plt.close(fig)

    def test_plot_uses_compact_borderless_reference_style(self) -> None:
        import matplotlib.pyplot as plt

        signal = np.zeros((12, 5000), dtype=np.float32)
        fig = plot_ecg_paper(signal)

        width, height = fig.get_size_inches()
        assert width / height == pytest.approx(2.0, rel=0.05)
        assert all(
            not spine.get_visible()
            for ax in fig.axes
            for spine in ax.spines.values()
        )
        assert any(
            line.get_label() == "1 mV calibration"
            for ax in fig.axes
            for line in ax.lines
        )
        plt.close(fig)


class TestAnalyzeNoImage:
    def test_returns_empty_context_and_no_image_when_no_upload(self) -> None:
        from src.web.app import _EMPTY_CONTEXT, analyze_ecg

        result = analyze_ecg(None, 0.7, "Auto-detect")

        assert len(result) == 6
        ecg_image, _, _, _, context, original = result
        assert ecg_image is None
        assert original is None
        assert context is _EMPTY_CONTEXT


class TestAnalyzeSignalScale:
    def test_uses_model_input_for_diagnosis_and_millivolts_for_plot(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from matplotlib.figure import Figure

        import src.web.app as web_app
        from src.pipeline.digitize import DigitizedSignals, DigitizeInfo
        from src.report.structured_report import ECGReport

        model_input = np.full((12, 5000), 7.0, dtype=np.float32)
        calibrated = np.full((12, 5000), 0.25, dtype=np.float32)
        captured: dict[str, np.ndarray] = {}

        class FakeDigitiser:
            def __init__(self) -> None:
                self._wrapper = SimpleNamespace(times={})
                self.last_info = DigitizeInfo()

            def digitize_with_calibrated(self, *_args, **_kwargs) -> DigitizedSignals:
                self.last_info = DigitizeInfo(
                    layout_name="standard_6x2+1R",
                    layout_cost=0.1,
                    raw_lines_count=7,
                    detected_leads_count=10,
                    avg_pixel_per_mm=9.0,
                    per_lead_energy=[1.0] * 12,
                    nonzero_leads_count=12,
                    einthoven_score=0.9,
                )
                return DigitizedSignals(
                    model_input=model_input,
                    calibrated_millivolts=calibrated,
                    rhythm_strip=None,
                )

        class FakeDiagnoser:
            last_estimated_hr_bpm = None
            last_interval_measurements = None

            def diagnose_all_segment_ensemble(self, signal: np.ndarray, **_kwargs) -> list:
                captured["diagnosis"] = signal
                return []

        def fake_report(*_args, **_kwargs) -> ECGReport:
            return ECGReport(
                heart_rate_bpm=None,
                rhythm_classification="Undetermined rhythm",
                rhythm_basis="undetermined",
                rate_category="unknown",
                ectopy_present=False,
                pvc_count=0,
                pr_ms=None,
                qrs_ms=None,
                qt_ms=None,
                qtc_ms=None,
                qtc_formula="bazett",
                interval_flags={},
                top_diagnoses=[],
                is_normal=False,
                overall_assessment="Indeterminate ECG",
                abnormal_reasons=[],
                normal_criteria={},
            )

        def fake_plot(signal: np.ndarray) -> Figure:
            captured["plot"] = signal
            return Figure()

        monkeypatch.setattr(web_app, "_get_digitiser", lambda: FakeDigitiser())
        monkeypatch.setattr(web_app, "_get_diagnoser", lambda: FakeDiagnoser())
        monkeypatch.setattr(web_app, "analyze_rhythm", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(web_app, "build_report", fake_report)
        monkeypatch.setattr(web_app, "narrative_available", lambda: False)
        monkeypatch.setattr(web_app, "plot_ecg_paper", fake_plot)
        monkeypatch.setattr(
            web_app,
            "fig_to_pil",
            lambda _fig: Image.new("RGB", (20, 20), color=(255, 255, 255)),
        )

        web_app.analyze_ecg(
            Image.new("RGB", (32, 32), color=(255, 255, 255)),
            0.7,
            "6x2+1R",
            progress=lambda *_args, **_kwargs: None,
        )

        assert captured["diagnosis"] is model_input
        assert captured["plot"] is calibrated

    def test_passes_vtsvt_assessment_to_report(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from matplotlib.figure import Figure

        import src.web.app as web_app
        from src.pipeline.digitize import DigitizedSignals, DigitizeInfo
        from src.report.structured_report import ECGReport

        signal = np.zeros((12, 5000), dtype=np.float32)
        captured: dict[str, object] = {}
        sentinel_assessment = object()

        class FakeDigitiser:
            def __init__(self) -> None:
                self._wrapper = SimpleNamespace(times={})
                self.last_info = DigitizeInfo(layout_name="3x4+1R")

            def digitize_with_calibrated(self, *_args, **_kwargs) -> DigitizedSignals:
                return DigitizedSignals(
                    model_input=signal,
                    calibrated_millivolts=signal,
                    rhythm_strip=None,
                )

        class FakeDiagnoser:
            last_estimated_hr_bpm = 130.0
            last_interval_measurements = None

            def diagnose_all_segment_ensemble(self, *_args, **_kwargs) -> list:
                return []

        def fake_report(*_args, **kwargs) -> ECGReport:
            captured["vtsvt"] = kwargs.get("vtsvt_assessment")
            return ECGReport(
                heart_rate_bpm=130.0,
                rhythm_classification="Undetermined rhythm (tachycardic)",
                rhythm_basis="undetermined",
                rate_category="tachycardia",
                ectopy_present=False,
                pvc_count=0,
                pr_ms=None,
                qrs_ms=None,
                qt_ms=None,
                qtc_ms=None,
                qtc_formula="n/a",
                interval_flags={},
                top_diagnoses=[],
                is_normal=False,
                overall_assessment="Abnormal ECG",
                abnormal_reasons=[],
                normal_criteria={},
            )

        monkeypatch.setattr(web_app, "_get_digitiser", lambda: FakeDigitiser())
        monkeypatch.setattr(web_app, "_get_diagnoser", lambda: FakeDiagnoser())
        monkeypatch.setattr(web_app, "analyze_rhythm", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(web_app, "assess_vtsvt", lambda *_args, **_kwargs: sentinel_assessment)
        monkeypatch.setattr(web_app, "build_report", fake_report)
        monkeypatch.setattr(web_app, "narrative_available", lambda: False)
        monkeypatch.setattr(web_app, "plot_ecg_paper", lambda _signal: Figure())
        monkeypatch.setattr(
            web_app,
            "fig_to_pil",
            lambda _fig: Image.new("RGB", (20, 20), color=(255, 255, 255)),
        )

        web_app.analyze_ecg(
            Image.new("RGB", (32, 32), color=(255, 255, 255)),
            0.7,
            "3x4+1R (standard)",
            progress=lambda *_args, **_kwargs: None,
        )

        assert captured["vtsvt"] is sentinel_assessment


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
