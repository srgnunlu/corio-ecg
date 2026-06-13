import json

import numpy as np
import pandas as pd
import pytest

from scripts.download_pmcardio_reference_subset import (
    PHYSICAL_CATEGORIES,
    build_selection_manifest,
    reference_key_for_row,
    select_balanced_image_ids,
    select_reference_rows,
)
from scripts.evaluate_pmcardio_reference import aggregate_fidelity_records, write_report
from src.training.reference_fidelity import (
    evaluate_absolute_printed_segments,
    evaluate_printed_segments,
    shifted_correlation,
)


def _metadata_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category in PHYSICAL_CATEGORIES:
        rows.append(
            {
                "Image name": "img_4_page_0.jpeg",
                "ECG ID": "LPAE_20999_hr",
                "Image relative path": f"{category}/img_4_page_0.jpeg",
                "Image ID": 4,
                "Image page": 0,
                "ECG format": "3x4+1R",
            }
        )
    rows.append(
        {
            "Image name": "img_5_page_0.jpeg",
            "ECG ID": "LPAE_21515_hr",
            "Image relative path": "photos_iphone/img_5_page_0.jpeg",
            "Image ID": 5,
            "Image page": 0,
            "ECG format": "6x2",
        }
    )
    return pd.DataFrame(rows)


def test_select_reference_rows_returns_requested_physical_variants() -> None:
    selected = select_reference_rows(
        _metadata_rows(),
        image_ids=[4],
        categories=PHYSICAL_CATEGORIES,
        layout="3x4+1R",
    )

    assert len(selected) == len(PHYSICAL_CATEGORIES)
    assert set(selected["category"]) == set(PHYSICAL_CATEGORIES)


def test_select_reference_rows_rejects_missing_variants() -> None:
    metadata = _metadata_rows()
    metadata = metadata[
        metadata["Image relative path"] != "photos_iphone/img_4_page_0.jpeg"
    ]

    with pytest.raises(ValueError, match="Missing requested PMcardio variants"):
        select_reference_rows(
            metadata,
            image_ids=[4],
            categories=PHYSICAL_CATEGORIES,
            layout="3x4+1R",
        )


def test_select_balanced_image_ids_is_deterministic_and_preserves_required_ids() -> None:
    metadata = pd.concat(
        [
            _metadata_rows(),
            pd.DataFrame(
                [
                    {
                        "Image name": f"img_{image_id}_page_0.jpeg",
                        "ECG ID": f"ECG_{image_id}",
                        "Image relative path": f"{category}/img_{image_id}_page_0.jpeg",
                        "Image ID": image_id,
                        "Image page": 0,
                        "ECG format": "3x4+1R",
                    }
                    for image_id in range(20, 30)
                    for category in PHYSICAL_CATEGORIES
                ]
            ),
        ],
        ignore_index=True,
    )

    first = select_balanced_image_ids(
        metadata,
        categories=PHYSICAL_CATEGORIES,
        layout="3x4+1R",
        count=5,
        seed=42,
        required_ids=[4],
    )
    second = select_balanced_image_ids(
        metadata,
        categories=PHYSICAL_CATEGORIES,
        layout="3x4+1R",
        count=5,
        seed=42,
        required_ids=[4],
    )

    assert first == second
    assert len(first) == 5
    assert 4 in first


def test_select_balanced_image_ids_rejects_insufficient_common_ids() -> None:
    with pytest.raises(ValueError, match="common image IDs"):
        select_balanced_image_ids(
            _metadata_rows(),
            categories=PHYSICAL_CATEGORIES,
            layout="3x4+1R",
            count=10,
            seed=42,
            required_ids=[],
        )


def test_build_selection_manifest_has_stable_identity() -> None:
    selected = select_reference_rows(
        _metadata_rows(),
        image_ids=[4],
        categories=PHYSICAL_CATEGORIES,
        layout="3x4+1R",
    )

    first = build_selection_manifest(selected, selection_seed=42)
    second = build_selection_manifest(selected.sample(frac=1), selection_seed=42)

    assert first["manifest_version"] == 1
    assert first["manifest_id"] == second["manifest_id"]
    assert first["selection"]["images"] == len(PHYSICAL_CATEGORIES)
    assert first["selection"]["images_per_category"] == {
        category: 1 for category in PHYSICAL_CATEGORIES
    }


def test_reference_key_uses_base_ecg_for_physical_photo() -> None:
    row = _metadata_rows().iloc[0]

    assert reference_key_for_row(row) == "LPAE_20999_hr"


def test_reference_key_uses_noise_prefix_for_digital_noise_variant() -> None:
    row = pd.Series(
        {
            "ECG ID": "LPAE_20999_hr",
            "Image relative path": "digital_data_high_freq_noise_low/img_4_page_0.jpeg",
        }
    )

    assert reference_key_for_row(row) == "high_freq_noise_low_LPAE_20999_hr"


def test_shifted_correlation_recovers_small_horizontal_shift() -> None:
    reference = np.sin(np.linspace(0, 8 * np.pi, 500))
    predicted = np.roll(reference, 17)

    result = shifted_correlation(reference, predicted, max_shift=25)

    assert result.correlation > 0.999
    assert abs(result.shift_samples) == 17


def test_evaluate_printed_segments_scores_matching_leads() -> None:
    time = np.linspace(0, 10 * np.pi, 1250)
    reference = np.stack(
        [np.sin(time + lead_index / 4) for lead_index in range(12)],
        axis=1,
    )
    digitized = np.stack(
        [np.tile(reference[:, lead_index], 4) for lead_index in range(12)],
        axis=0,
    )

    result = evaluate_printed_segments(reference, digitized, max_shift_samples=50)

    assert result["median_correlation"] > 0.999
    assert result["mean_correlation"] > 0.999
    assert result["median_normalized_rmse"] < 1e-6
    assert len(result["per_lead"]) == 12


def test_evaluate_printed_segments_validates_shapes() -> None:
    with pytest.raises(ValueError, match="reference"):
        evaluate_printed_segments(np.ones((12, 100)), np.ones((12, 5000)))

    with pytest.raises(ValueError, match="digitized"):
        evaluate_printed_segments(np.ones((100, 12)), np.ones((11, 5000)))


def test_evaluate_absolute_printed_segments_reports_mv_error_and_snr() -> None:
    time = np.linspace(0, 10 * np.pi, 1250)
    reference = np.stack(
        [np.sin(time + lead_index / 4) for lead_index in range(12)],
        axis=1,
    )
    calibrated = np.stack(
        [np.tile(reference[:, lead_index] * 0.5, 4) for lead_index in range(12)],
        axis=0,
    )

    result = evaluate_absolute_printed_segments(
        reference,
        calibrated,
        max_shift_samples=50,
    )

    assert result["median_gain_ratio"] == pytest.approx(0.5, abs=1e-3)
    assert result["median_rmse_mv"] == pytest.approx(np.sqrt(0.125), abs=1e-3)
    assert result["median_snr_db"] == pytest.approx(6.0206, abs=1e-3)


def test_aggregate_fidelity_records_groups_categories() -> None:
    records = [
        {
            "category": "photos_iphone",
            "status": "success",
            "fidelity": {
                "median_correlation": 0.8,
                "absolute": {
                    "median_rmse_mv": 0.1,
                    "median_snr_db": 5.0,
                    "median_gain_ratio": 0.9,
                },
            },
        },
        {
            "category": "photos_iphone",
            "status": "failed",
            "fidelity": None,
        },
        {
            "category": "photos_scans",
            "status": "success",
            "fidelity": {
                "median_correlation": 0.9,
                "absolute": {
                    "median_rmse_mv": 0.05,
                    "median_snr_db": 10.0,
                    "median_gain_ratio": 1.0,
                },
            },
        },
    ]

    aggregate = aggregate_fidelity_records(records)

    assert aggregate["total"] == 3
    assert aggregate["successful"] == 2
    assert aggregate["success_rate"] == pytest.approx(2 / 3)
    assert aggregate["median_correlation"] == pytest.approx(0.85)
    assert aggregate["median_correlation_ci95"]["low"] <= 0.85
    assert aggregate["median_correlation_ci95"]["high"] >= 0.85
    assert aggregate["absolute_amplitude"]["available"] == 2
    assert aggregate["absolute_amplitude"]["median_rmse_mv"] == pytest.approx(0.075)
    assert aggregate["absolute_amplitude"]["median_gain_ratio"] == pytest.approx(0.95)
    assert aggregate["categories"]["photos_iphone"]["success_rate"] == 0.5
    assert aggregate["categories"]["photos_scans"]["median_correlation"] == 0.9


def test_write_report_embeds_selection_manifest(tmp_path) -> None:
    manifest = {"manifest_version": 1, "manifest_id": "abc123"}

    json_path, _ = write_report([], tmp_path, selection_manifest=manifest)
    report = json.loads(json_path.read_text())

    assert report["selection_manifest"] == manifest
