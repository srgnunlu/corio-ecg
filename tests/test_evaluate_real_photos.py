import csv
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_real_photos import (
    REAL_PHOTO_EVALUATION_VERSION,
    aggregate_records,
    collect_photo_paths,
    is_reusable_record,
    load_layout_hints,
    run_worker,
    write_reports,
)
from src.pipeline.digitize import DigitizedSignals, DigitizeInfo


def _success_record(
    *,
    mode: str = "default",
    nonzero_leads_count: int = 12,
    einthoven_score: float = 0.8,
    has_warnings: bool = False,
) -> dict[str, object]:
    return {
        "pipeline_version": REAL_PHOTO_EVALUATION_VERSION,
        "source_image": "data/real-phone/photos/case001__front.jpg",
        "source_sha256": "abc123",
        "mode": mode,
        "capture_variant": "front",
        "layout_hint": None,
        "status": "success",
        "elapsed_seconds": 12.0,
        "has_warnings": has_warnings,
        "diagnostics": {
            "layout_name": "standard_3x4+1R",
            "processing_mode": "default",
            "nonzero_leads_count": nonzero_leads_count,
            "einthoven_score": einthoven_score,
        },
    }


def test_collect_photo_paths_supports_common_image_extensions(tmp_path: Path) -> None:
    for name in ("case003.png", "case001.jpg", "case002.jpeg", "ignore.txt"):
        (tmp_path / name).touch()

    paths = collect_photo_paths(tmp_path)

    assert [path.name for path in paths] == [
        "case001.jpg",
        "case002.jpeg",
        "case003.png",
    ]


def test_load_layout_hints_ignores_unknown_layouts(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "case_id,variant,layout\n"
        "case001,front,standard_6x2\n"
        "case002,screen,unknown\n"
    )

    assert load_layout_hints(metadata_path) == {"case001": "standard_6x2"}


def test_aggregate_records_reports_acceptance_metrics() -> None:
    records = [
        _success_record(),
        _success_record(nonzero_leads_count=11, einthoven_score=0.6, has_warnings=True),
        {
            "mode": "default",
            "status": "timeout",
            "elapsed_seconds": 90.0,
            "diagnostics": {},
        },
    ]

    aggregate = aggregate_records(records)
    default = aggregate["modes"]["default"]

    assert default["total"] == 3
    assert default["successful"] == 2
    assert default["timed_out"] == 1
    assert default["success_rate"] == 2 / 3
    assert default["twelve_active_leads_rate"] == 0.5
    assert default["median_einthoven_score"] == 0.7
    assert default["acceptance"]["digitization_success_rate"] is False
    assert default["acceptance"]["twelve_active_leads_rate"] is False
    assert default["acceptance"]["median_einthoven_score"] is True
    assert aggregate["variants"]["front"]["modes"]["default"]["successful"] == 2


def test_is_reusable_record_requires_matching_version_mode_and_source_hash(
    tmp_path: Path,
) -> None:
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(_success_record()))

    assert is_reusable_record(
        record_path, mode="default", source_sha256="abc123", layout_hint=None
    ) is True
    assert is_reusable_record(
        record_path, mode="retry", source_sha256="abc123", layout_hint=None
    ) is False
    assert is_reusable_record(
        record_path, mode="default", source_sha256="changed", layout_hint=None
    ) is False
    assert is_reusable_record(
        record_path,
        mode="default",
        source_sha256="abc123",
        layout_hint="standard_6x2",
    ) is False


def test_failed_record_is_not_reusable(tmp_path: Path) -> None:
    record = _success_record()
    record["status"] = "failed"
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record))

    assert is_reusable_record(
        record_path, mode="default", source_sha256="abc123", layout_hint=None
    ) is False


def test_write_reports_creates_json_and_flat_csv(tmp_path: Path) -> None:
    record = _success_record()

    json_path, csv_path = write_reports([record], tmp_path)

    report = json.loads(json_path.read_text())
    assert report["reference_level"] == "photos_only"
    assert report["aggregate"]["modes"]["default"]["successful"] == 1

    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]["status"] == "success"
    assert rows[0]["capture_variant"] == "front"
    assert rows[0]["nonzero_leads_count"] == "12"


def test_run_worker_can_save_model_and_calibrated_signals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeDigitiser:
        def __init__(self, **_kwargs) -> None:
            self.last_info = DigitizeInfo()

        def digitize_with_calibrated(self, *_args, **_kwargs) -> DigitizedSignals:
            return DigitizedSignals(
                model_input=np.ones((12, 5000), dtype=np.float32),
                calibrated_millivolts=np.full((12, 5000), 0.5, dtype=np.float32),
            )

    image_path = tmp_path / "ecg.png"
    image_path.write_bytes(b"not-an-image")
    signal_path = tmp_path / "signal.npy"
    calibrated_path = tmp_path / "calibrated.npy"
    monkeypatch.setattr("scripts.evaluate_real_photos.ECGDigitiser", FakeDigitiser)

    record = run_worker(
        image_path,
        mode="default",
        layout_hint="3x4+1R",
        signal_path=signal_path,
        calibrated_signal_path=calibrated_path,
    )

    assert record["status"] == "success"
    assert np.load(signal_path).mean() == 1.0
    assert np.load(calibrated_path).mean() == 0.5
