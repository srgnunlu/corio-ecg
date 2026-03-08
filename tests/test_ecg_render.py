# Tests for the synthetic ECG image renderer (ecg_render.py)
# Unit tests verify CLI argument construction; integration tests generate real images.

import os
import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.ecg_render import (
    DifficultyLevel,
    _build_cli_args,
    _find_generated_png,
    render_ecg_image,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RECORD = PROJECT_ROOT / "data" / "raw" / "ptb-xl" / "records500" / "00000" / "00001_hr"


def _ptbxl_available() -> bool:
    """Check if PTB-XL sample record exists on disk."""
    return SAMPLE_RECORD.with_suffix(".dat").exists() and SAMPLE_RECORD.with_suffix(".hea").exists()


# ---------------------------------------------------------------------------
# Unit tests — CLI argument construction
# ---------------------------------------------------------------------------

class TestBuildCliArgs:
    """Verify that each difficulty level produces the correct CLI flags."""

    def test_clean_args(self) -> None:
        args = _build_cli_args(
            dat_path="/tmp/rec.dat",
            hea_path="/tmp/rec.hea",
            output_dir="/tmp/out",
            difficulty=DifficultyLevel.CLEAN,
            seed=42,
        )
        assert "-r" in args
        assert args[args.index("-r") + 1] == "200"
        assert "--standard_grid_color" in args
        # CLEAN should NOT have distortion flags
        assert "--augment" not in args
        assert "--hw_text" not in args
        assert "--wrinkles" not in args

    def test_moderate_args(self) -> None:
        args = _build_cli_args(
            dat_path="/tmp/rec.dat",
            hea_path="/tmp/rec.hea",
            output_dir="/tmp/out",
            difficulty=DifficultyLevel.MODERATE,
            seed=99,
        )
        assert "--augment" in args
        assert "-noise" in args
        assert args[args.index("-noise") + 1] == "25"
        assert "-rot" in args
        assert args[args.index("-rot") + 1] == "2"
        assert "-t" in args
        assert args[args.index("-t") + 1] == "8000"
        # MODERATE should NOT have wrinkles or handwritten text
        assert "--hw_text" not in args
        assert "--wrinkles" not in args

    def test_hard_args(self) -> None:
        args = _build_cli_args(
            dat_path="/tmp/rec.dat",
            hea_path="/tmp/rec.hea",
            output_dir="/tmp/out",
            difficulty=DifficultyLevel.HARD,
            seed=7,
        )
        assert "--augment" in args
        assert "--hw_text" not in args
        assert "--wrinkles" not in args
        assert "--random_grid_color" in args
        assert "-noise" in args
        assert args[args.index("-noise") + 1] == "50"
        assert "-rot" in args
        assert args[args.index("-rot") + 1] == "8"
        # Find the crop -c flag (not the python -c flag which comes first)
        c_indices = [i for i, a in enumerate(args) if a == "-c"]
        assert len(c_indices) >= 2, "Expected at least 2 '-c' flags (python -c and crop -c)"
        crop_idx = c_indices[-1]  # last one is the crop parameter
        assert args[crop_idx + 1] == "0.03"
        assert "-t" in args
        assert args[args.index("-t") + 1] == "3000"

    def test_seed_is_passed(self) -> None:
        args = _build_cli_args(
            dat_path="/tmp/rec.dat",
            hea_path="/tmp/rec.hea",
            output_dir="/tmp/out",
            difficulty=DifficultyLevel.CLEAN,
            seed=123,
        )
        assert "-se" in args
        assert args[args.index("-se") + 1] == "123"

    def test_start_index_is_zero(self) -> None:
        args = _build_cli_args(
            dat_path="/tmp/rec.dat",
            hea_path="/tmp/rec.hea",
            output_dir="/tmp/out",
            difficulty=DifficultyLevel.CLEAN,
            seed=42,
        )
        assert "-st" in args
        assert args[args.index("-st") + 1] == "0"

    def test_all_levels_include_twelve_leads(self) -> None:
        for level in DifficultyLevel:
            args = _build_cli_args(
                dat_path="/tmp/rec.dat",
                hea_path="/tmp/rec.hea",
                output_dir="/tmp/out",
                difficulty=level,
                seed=42,
            )
            assert "--num_leads" in args
            assert args[args.index("--num_leads") + 1] == "twelve"


# ---------------------------------------------------------------------------
# Unit tests — PNG finder
# ---------------------------------------------------------------------------

class TestFindGeneratedPng:
    def test_finds_expected_filename(self, tmp_path: Path) -> None:
        png_file = tmp_path / "00001_hr-0.png"
        png_file.write_bytes(b"fake png")
        result = _find_generated_png(tmp_path, "00001_hr")
        assert result == png_file

    def test_fallback_glob(self, tmp_path: Path) -> None:
        # If the file has a different suffix pattern
        png_file = tmp_path / "00001_hr-1.png"
        png_file.write_bytes(b"fake png")
        result = _find_generated_png(tmp_path, "00001_hr")
        assert result == png_file

    def test_returns_none_when_no_match(self, tmp_path: Path) -> None:
        result = _find_generated_png(tmp_path, "nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Unit tests — error handling
# ---------------------------------------------------------------------------

class TestRenderErrors:
    def test_missing_dat_file(self, tmp_path: Path) -> None:
        hea = tmp_path / "test.hea"
        hea.write_text("fake header")
        with pytest.raises(FileNotFoundError, match="DAT file not found"):
            render_ecg_image(str(tmp_path / "test"), tmp_path / "out.png")

    def test_missing_hea_file(self, tmp_path: Path) -> None:
        dat = tmp_path / "test.dat"
        dat.write_bytes(b"fake data")
        with pytest.raises(FileNotFoundError, match="HEA file not found"):
            render_ecg_image(str(tmp_path / "test"), tmp_path / "out.png")


# ---------------------------------------------------------------------------
# Integration test — requires PTB-XL data on disk
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ptbxl_available(), reason="PTB-XL data not available")
class TestRenderIntegration:
    def test_generate_clean_image(self, tmp_path: Path) -> None:
        """Generate a real clean ECG image and verify it's a valid PNG."""
        output_file = tmp_path / "test_output.png"

        result = render_ecg_image(
            record_path=str(SAMPLE_RECORD),
            output_path=output_file,
            difficulty=DifficultyLevel.CLEAN,
            seed=42,
        )

        assert result.exists(), "Output PNG was not created"
        assert result.stat().st_size > 1000, "PNG file is suspiciously small"

        # Verify PNG magic bytes (first 8 bytes of any valid PNG)
        with open(result, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG", "File does not have PNG magic bytes"
