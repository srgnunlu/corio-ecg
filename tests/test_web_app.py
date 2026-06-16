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
