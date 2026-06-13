import numpy as np
import pytest

from scripts.evaluate_pmcardio_diagnosis_drift import (
    aggregate_drift_records,
    compare_probability_vectors,
    prepare_reference_model_input,
)


def test_prepare_reference_model_input_tiles_and_normalizes_segments() -> None:
    reference = np.stack(
        [np.linspace(lead_index, lead_index + 1, 1250) for lead_index in range(12)],
        axis=1,
    )

    model_input = prepare_reference_model_input(reference)

    assert model_input.shape == (12, 5000)
    assert model_input.dtype == np.float32
    assert np.mean(model_input) == pytest.approx(0.0, abs=1e-6)
    assert np.std(model_input) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(model_input[:, :1250], model_input[:, 1250:2500])


def test_compare_probability_vectors_reports_consistency_metrics() -> None:
    reference = np.array([0.1, 0.8, 0.6, 0.2])
    digitized = np.array([0.2, 0.7, 0.4, 0.3])

    result = compare_probability_vectors(reference, digitized, threshold=0.5)

    assert result["cosine_similarity"] > 0.9
    assert result["mean_abs_probability_difference"] == pytest.approx(0.125)
    assert result["agreement_rate"] == pytest.approx(0.75)


def test_aggregate_drift_records_groups_methods_and_categories() -> None:
    records = [
        {
            "category": "photos_scans",
            "status": "success",
            "methods": {
                "tiled": {
                    "cosine_similarity": 0.8,
                    "mean_abs_probability_difference": 0.1,
                    "agreement_rate": 0.9,
                },
                "segment_ensemble": {
                    "cosine_similarity": 0.9,
                    "mean_abs_probability_difference": 0.05,
                    "agreement_rate": 0.95,
                },
            },
        },
        {
            "category": "photos_bents",
            "status": "success",
            "methods": {
                "tiled": {
                    "cosine_similarity": 0.4,
                    "mean_abs_probability_difference": 0.3,
                    "agreement_rate": 0.7,
                },
                "segment_ensemble": {
                    "cosine_similarity": 0.6,
                    "mean_abs_probability_difference": 0.2,
                    "agreement_rate": 0.8,
                },
            },
        },
    ]

    aggregate = aggregate_drift_records(records)

    assert aggregate["total"] == 2
    assert aggregate["successful"] == 2
    assert aggregate["methods"]["tiled"]["mean_cosine_similarity"] == pytest.approx(0.6)
    assert aggregate["methods"]["segment_ensemble"]["mean_agreement_rate"] == pytest.approx(
        0.875
    )
    assert (
        aggregate["categories"]["photos_scans"]["methods"]["tiled"][
            "mean_cosine_similarity"
        ]
        == 0.8
    )
