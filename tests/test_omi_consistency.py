# Tests for paired clean/digitised consistency training.

import numpy as np
import pytest
import torch
from torch import nn

from src.omi.consistency import (
    ConsistencyConfig,
    PairedSignalDataset,
    consistency_penalty,
    predict_logits,
    predict_scores,
    train_consistency,
)


class _StubClassifier(nn.Module):
    """Minimal stand-in for OmiClassifier: one logit from the per-lead means."""

    def __init__(self, leads: int = 12) -> None:
        super().__init__()
        self.head = nn.Linear(leads, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.mean(dim=-1)).squeeze(-1)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())


def _seeded_stub(seed: int = 0) -> _StubClassifier:
    """A stub whose initial weights do not depend on test execution order."""
    torch.manual_seed(seed)
    return _StubClassifier()


def _separable_pairs(n: int = 48, leads: int = 12, samples: int = 64, seed: int = 3):
    """Positives sit higher than negatives; the digitised view is noisier."""
    rng = np.random.default_rng(seed)
    labels = np.array([1, 0] * (n // 2))
    offsets = np.where(labels == 1, 1.0, -1.0)[:, None, None]
    clean = (offsets + rng.normal(0, 0.1, (n, leads, samples))).astype(np.float16)
    digitised = (offsets * 0.6 + rng.normal(0, 0.3, (n, leads, samples))).astype(np.float16)
    return clean, digitised, labels


class TestPairedSignalDataset:
    """Tests for the paired dataset."""

    def test_serves_float32_triples(self) -> None:
        clean, digitised, labels = _separable_pairs()
        dataset = PairedSignalDataset(clean, digitised, labels)

        clean_item, digitised_item, label, teacher = dataset[0]

        assert clean_item.dtype == torch.float32
        assert digitised_item.dtype == torch.float32
        assert clean_item.shape == (12, 64)
        assert label.item() == float(labels[0])
        assert torch.isnan(teacher), "no teacher supplied — must be flagged as absent"

    def test_carries_a_teacher_logit_when_given(self) -> None:
        clean, digitised, labels = _separable_pairs()
        teacher = np.linspace(-3.0, 3.0, len(labels))

        _, _, _, served = PairedSignalDataset(clean, digitised, labels, teacher)[2]

        assert served.item() == pytest.approx(teacher[2])

    def test_length_mismatch_is_rejected(self) -> None:
        clean, digitised, labels = _separable_pairs()

        with pytest.raises(ValueError, match="same length"):
            PairedSignalDataset(clean, digitised[:-1], labels)

    def test_teacher_length_mismatch_is_rejected(self) -> None:
        clean, digitised, labels = _separable_pairs()

        with pytest.raises(ValueError, match="teacher logits"):
            PairedSignalDataset(clean, digitised, labels, np.zeros(len(labels) - 1))


class TestConsistencyPenalty:
    """Tests for the penalty that pulls the two views together."""

    def test_identical_views_cost_nothing(self) -> None:
        logits = torch.tensor([1.5, -2.0, 0.3])

        assert consistency_penalty(logits, logits).item() == pytest.approx(0.0)

    def test_disagreement_costs_more_the_further_apart(self) -> None:
        clean = torch.tensor([2.0, 2.0])
        near = torch.tensor([1.5, 1.5])
        far = torch.tensor([-2.0, -2.0])

        assert consistency_penalty(clean, far) > consistency_penalty(clean, near) > 0


class TestTrainConsistency:
    """Tests for the training loop."""

    def test_learns_a_separable_problem(self) -> None:
        clean, digitised, labels = _separable_pairs()
        teacher = np.where(labels == 1, 3.0, -3.0)
        model = _seeded_stub()
        device = torch.device("cpu")

        history, best = train_consistency(
            model,
            PairedSignalDataset(clean, digitised, labels, teacher),
            digitised,
            labels,
            device,
            ConsistencyConfig(epochs=6, batch_size=8),
        )

        assert len(history) >= 1
        assert best["auprc"] > 0.9
        assert best["epoch"] >= 1

    def test_a_dominant_consistency_term_does_not_collapse_the_model(self) -> None:
        """The reason the target is fixed: equal logits must not be a way out.

        With a symmetric penalty the cheapest way to agree is to predict the
        same thing everywhere, which scores perfectly on consistency and zero
        on the task. A teacher target makes agreement mean 'be right'.
        """
        clean, digitised, labels = _separable_pairs()
        teacher = np.where(labels == 1, 3.0, -3.0)
        model = _seeded_stub()

        _, best = train_consistency(
            model,
            PairedSignalDataset(clean, digitised, labels, teacher),
            digitised,
            labels,
            torch.device("cpu"),
            ConsistencyConfig(epochs=8, batch_size=8, consistency_weight=20.0),
        )

        assert best["auprc"] > 0.9

    def test_zero_weight_is_plain_digitised_training(self) -> None:
        """The ablation arm must run and report a zero consistency term."""
        clean, digitised, labels = _separable_pairs()
        model = _seeded_stub()

        history, _ = train_consistency(
            model,
            PairedSignalDataset(clean, digitised, labels),
            digitised,
            labels,
            torch.device("cpu"),
            ConsistencyConfig(epochs=2, batch_size=8, consistency_weight=0.0),
        )

        # The penalty is still measured for the report, just not optimised.
        assert all(record.consistency_loss >= 0.0 for record in history)
        assert len(history) == 2

    def test_model_keeps_the_best_epoch_not_the_last(self) -> None:
        clean, digitised, labels = _separable_pairs()
        model = _seeded_stub()
        device = torch.device("cpu")

        history, best = train_consistency(
            model,
            PairedSignalDataset(clean, digitised, labels),
            digitised,
            labels,
            device,
            ConsistencyConfig(epochs=8, batch_size=8),
        )

        from sklearn.metrics import average_precision_score

        final = average_precision_score(labels, predict_scores(model, digitised, device))
        assert final == pytest.approx(best["auprc"], abs=1e-9)
        assert best["auprc"] >= max(record.validation_auprc for record in history) - 1e-9


class TestPredictScores:
    """Tests for batched scoring."""

    def test_returns_one_probability_per_record(self) -> None:
        _, digitised, _ = _separable_pairs(n=20)
        scores = predict_scores(_seeded_stub(), digitised, torch.device("cpu"), batch_size=7)

        assert scores.shape == (20,)
        assert ((scores >= 0.0) & (scores <= 1.0)).all()

    def test_logits_are_the_unsquashed_form_of_the_scores(self) -> None:
        """Teacher targets are taken in logit space, so the two must agree."""
        _, digitised, _ = _separable_pairs(n=12)
        model = _seeded_stub()
        device = torch.device("cpu")

        logits = predict_logits(model, digitised, device)
        scores = predict_scores(model, digitised, device)

        assert logits.shape == scores.shape
        assert np.allclose(1.0 / (1.0 + np.exp(-logits)), scores, atol=1e-6)
