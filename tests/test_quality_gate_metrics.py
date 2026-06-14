"""Tests for statistical quality-gate evaluation metrics."""

from __future__ import annotations

from src.evaluation.quality_gate_metrics import evaluate_quality_gate_rows


def _row(
    target: str,
    prediction: str,
    *,
    ecg_id: str,
    image_id: int,
) -> dict[str, object]:
    return {
        "target": target,
        "prediction": prediction,
        "ecg_id": ecg_id,
        "image_id": image_id,
        "category": "test",
    }


def test_evaluate_quality_gate_rows_reports_confusion_matrix_and_coverage() -> None:
    rows = [
        _row("accept", "accept", ecg_id="a", image_id=1),
        _row("warn", "warn", ecg_id="b", image_id=2),
        _row("reject", "warn", ecg_id="c", image_id=3),
        _row("reject", "reject", ecg_id="d", image_id=4),
    ]

    metrics = evaluate_quality_gate_rows(rows, bootstrap_resamples=100)

    assert metrics["confusion_matrix"]["reject"] == {
        "accept": 0,
        "warn": 1,
        "reject": 1,
    }
    assert metrics["non_reject_coverage"] == 0.75
    assert metrics["reject_recall"] == 0.5
    assert metrics["independent_groups"] == 4
    assert metrics["missed_reject_records"] == [
        {"ecg_id": "c", "image_id": 3, "category": "test", "prediction": "warn"}
    ]


def test_evaluate_quality_gate_rows_bootstrap_intervals_are_deterministic() -> None:
    rows = [
        _row("reject", "reject", ecg_id=f"r-{index}", image_id=index)
        for index in range(5)
    ] + [
        _row("accept", "accept", ecg_id=f"a-{index}", image_id=index + 10)
        for index in range(5)
    ]

    first = evaluate_quality_gate_rows(rows, bootstrap_resamples=100, bootstrap_seed=7)
    second = evaluate_quality_gate_rows(rows, bootstrap_resamples=100, bootstrap_seed=7)

    assert first["confidence_intervals"] == second["confidence_intervals"]
    assert first["confidence_intervals"]["reject_recall"]["low"] == 1.0
    assert first["confidence_intervals"]["reject_recall"]["high"] == 1.0
    assert first["confidence_intervals"]["reject_recall"]["resampling_unit"] == "ecg_id"


def test_evaluate_quality_gate_rows_handles_empty_input() -> None:
    metrics = evaluate_quality_gate_rows([], bootstrap_resamples=100)

    assert metrics["total"] == 0
    assert metrics["non_reject_coverage"] == 0.0
    assert metrics["confidence_intervals"]["reject_recall"] is None
    assert metrics["statistical_warning"] is not None


def test_evaluate_quality_gate_rows_warns_for_small_samples() -> None:
    metrics = evaluate_quality_gate_rows(
        [_row("accept", "accept", ecg_id="a", image_id=1)],
        bootstrap_resamples=100,
    )

    assert "1 ECG groups" in metrics["statistical_warning"]


def test_evaluate_quality_gate_rows_counts_variants_as_one_independent_group() -> None:
    metrics = evaluate_quality_gate_rows(
        [
            _row("accept", "accept", ecg_id="same-ecg", image_id=1),
            _row("reject", "warn", ecg_id="same-ecg", image_id=2),
        ],
        bootstrap_resamples=100,
    )

    assert metrics["total"] == 2
    assert metrics["independent_groups"] == 1
