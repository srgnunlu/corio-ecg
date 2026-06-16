# Tests for audited synthetic-image digitization resume behavior.

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from scripts.digitize_synthetic_images import (
    DIGITIZATION_PIPELINE_VERSION,
    _audit_diagnostics,
    _is_current_saved_signal,
    _json_default,
    _metadata_path,
    collect_image_paths,
    digitize_batch,
)
from src.pipeline.digitize import DigitizeInfo


def _write_signal(path: Path, active_leads: int = 12) -> None:
    signal = np.zeros((12, 5000), dtype=np.float32)
    signal[:active_leads] = 1.0
    np.save(path, signal)


def _write_metadata(path: Path, version: int = DIGITIZATION_PIPELINE_VERSION) -> None:
    _metadata_path(path).write_text(json.dumps({"pipeline_version": version}))


def test_current_audited_signal_is_reusable(tmp_path: Path) -> None:
    signal_path = tmp_path / "1.npy"
    _write_signal(signal_path)
    _write_metadata(signal_path)

    assert _is_current_saved_signal(signal_path) is True


def test_signal_without_metadata_is_invalidated(tmp_path: Path) -> None:
    signal_path = tmp_path / "1.npy"
    _write_signal(signal_path)

    assert _is_current_saved_signal(signal_path) is False


def test_signal_from_old_pipeline_version_is_invalidated(tmp_path: Path) -> None:
    signal_path = tmp_path / "1.npy"
    _write_signal(signal_path)
    _write_metadata(signal_path, version=DIGITIZATION_PIPELINE_VERSION - 1)

    assert _is_current_saved_signal(signal_path) is False


def test_signal_with_too_few_active_leads_is_invalidated(tmp_path: Path) -> None:
    signal_path = tmp_path / "1.npy"
    _write_signal(signal_path, active_leads=4)
    _write_metadata(signal_path)

    assert _is_current_saved_signal(signal_path) is False


def test_json_default_converts_tensor_and_numpy_scalar() -> None:
    assert _json_default(torch.tensor(1.5)) == pytest.approx(1.5)
    assert _json_default(torch.tensor([1, 2])) == [1, 2]
    assert _json_default(np.float32(2.5)) == pytest.approx(2.5)


def test_audit_diagnostics_drops_bulky_arrays_and_serializes() -> None:
    # raw_lines / signal_probability are H×W debug arrays the audit JSON must
    # not contain — leaving them in made json.dumps raise, failing every
    # metadata write while the .npy signal saved fine.
    info = DigitizeInfo(nonzero_leads_count=11)
    info.raw_lines = np.zeros((4, 2200), dtype=np.float32)
    info.signal_probability = np.zeros((1800, 2400), dtype=np.float32)

    diagnostics = _audit_diagnostics(info)

    assert "raw_lines" not in diagnostics
    assert "signal_probability" not in diagnostics
    # Must round-trip through json without raising.
    payload = json.dumps({"diagnostics": diagnostics}, default=_json_default)
    assert json.loads(payload)["diagnostics"]["nonzero_leads_count"] == 11


def test_json_default_handles_multidim_ndarray() -> None:
    assert _json_default(np.zeros((2, 2), dtype=np.float32)) == [[0.0, 0.0], [0.0, 0.0]]


def test_collect_image_paths_sorts_numeric_ecg_ids_before_limiting(
    tmp_path: Path,
) -> None:
    level_dir = tmp_path / "clean"
    level_dir.mkdir()
    for stem in ("100", "9", "38"):
        (level_dir / f"{stem}.png").touch()

    paths = collect_image_paths(tmp_path, "clean", max_samples=2)

    assert [path.stem for path in paths] == ["9", "38"]


def test_digitize_batch_stops_after_max_new(tmp_path: Path) -> None:
    image_paths = [tmp_path / f"{index}.png" for index in range(3)]
    digitiser = MagicMock()
    digitiser.enable_dewarping_retry = False
    digitiser.enable_orientation_retry = False
    digitiser.last_info = DigitizeInfo(nonzero_leads_count=12)
    digitiser.digitize.return_value = np.ones((12, 5000), dtype=np.float32)

    summary = digitize_batch(
        digitiser,
        image_paths,
        tmp_path / "signals",
        level="clean",
        max_new=2,
    )

    assert summary["digitized"] == 2
    assert digitiser.digitize.call_count == 2


def test_failed_reprocessing_removes_stale_signal(tmp_path: Path) -> None:
    image_path = tmp_path / "1.png"
    output_path = tmp_path / "signals" / "clean" / "1.npy"
    output_path.parent.mkdir(parents=True)
    _write_signal(output_path, active_leads=4)
    digitiser = MagicMock()
    digitiser.digitize.side_effect = RuntimeError("failed")

    summary = digitize_batch(
        digitiser,
        [image_path],
        tmp_path / "signals",
        level="clean",
    )

    assert summary["failed"] == 1
    assert not output_path.exists()
