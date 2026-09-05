# Tests for paired clean/digitised consistency training.

import numpy as np
import pytest
import torch
from torch import nn

from src.omi.consistency import (
    ConsistencyConfig,
    PairedSignalDataset,
    consistency_penalty,
    consistency_target,
    predict_logits,
    predict_scores,
    train_consistency,
)


class _StubClassifier(nn.Module):
    """Minimal stand-in for OmiClassifier: logits from the per-lead means.

    One output mimics the OMI head; several mimic the 150-label general head.
    """

    def __init__(self, leads: int = 12, outputs: int = 1) -> None:
        super().__init__()
        self.head = nn.Linear(leads, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(x.mean(dim=-1))
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits

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


class TestConsistencyTarget:
    """Tests for the per-row penalty target."""

    def test_teacher_where_present_clean_logit_elsewhere(self) -> None:
        teacher = torch.tensor([2.0, float("nan"), -1.0])
        clean = torch.tensor([0.5, 0.7, 0.9], requires_grad=True)

        target = consistency_target(teacher, clean)

        assert target.tolist() == pytest.approx([2.0, 0.7, -1.0])
        assert not target.requires_grad, "the clean fallback must be detached"

    def test_a_scalar_nan_teacher_falls_back_across_a_vector_head(self) -> None:
        """An absent teacher is served per row; a 150-output head needs it per output."""
        teacher = torch.tensor([float("nan"), float("nan")])
        clean = torch.tensor([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])

        target = consistency_target(teacher, clean)

        assert target.shape == clean.shape
        assert torch.equal(target, clean)


class TestMultiLabelHead:
    """Tests for the general (many-output) head sharing the OMI training loop."""

    @staticmethod
    def _pairs(n: int = 48):
        """Output 0 is separable, output 1 has no ground truth at all."""
        clean, digitised, labels = _separable_pairs(n=n)
        missing = np.full(n, np.nan, dtype=np.float32)
        targets = np.stack([labels.astype(np.float32), missing], axis=1)
        return clean, digitised, targets

    def test_nan_outputs_are_skipped_and_the_rest_is_learned(self) -> None:
        clean, digitised, targets = self._pairs()

        def scorer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
            from sklearn.metrics import average_precision_score, roc_auc_score

            return (
                float(average_precision_score(labels[:, 0], scores[:, 0])),
                float(roc_auc_score(labels[:, 0], scores[:, 0])),
            )

        torch.manual_seed(0)
        history, best = train_consistency(
            _StubClassifier(outputs=2),
            PairedSignalDataset(clean, digitised, targets, np.zeros_like(targets)),
            digitised,
            targets,
            torch.device("cpu"),
            ConsistencyConfig(epochs=6, batch_size=8),
            positive_weight=torch.tensor([1.0, 1.0]),
            validation_scorer=scorer,
        )

        assert all(np.isfinite(record.total_loss) for record in history)
        assert history[0].digitised_loss > 0.0
        assert best["auprc"] > 0.9

    def test_per_output_positive_weights_are_accepted(self) -> None:
        clean, digitised, targets = self._pairs(n=16)
        torch.manual_seed(0)

        history, _ = train_consistency(
            _StubClassifier(outputs=2),
            PairedSignalDataset(clean, digitised, targets),
            digitised,
            targets,
            torch.device("cpu"),
            ConsistencyConfig(epochs=1, batch_size=8),
            positive_weight=torch.tensor([3.0, 1.0]),
            validation_scorer=lambda labels, scores: (0.5, 0.5),
        )

        assert len(history) == 1
        assert np.isfinite(history[0].total_loss)


def _unlabelled_pairs(n: int = 64, leads: int = 12, samples: int = 64, seed: int = 9):
    """Pairs with no ground truth: NaN labels, one teacher logit per row."""
    rng = np.random.default_rng(seed)
    clean = rng.normal(0, 1, (n, leads, samples)).astype(np.float16)
    digitised = (clean + rng.normal(0, 0.3, clean.shape)).astype(np.float16)
    labels = np.full(n, np.nan)
    teacher = rng.normal(0, 2, n)
    return clean, digitised, labels, teacher


class TestUnlabelledPairs:
    """Tests for scaling the consistency term with pairs that carry no label."""

    def test_dataset_serves_nan_for_a_missing_label(self) -> None:
        clean, digitised, labels, teacher = _unlabelled_pairs(n=4)

        _, _, label, served = PairedSignalDataset(clean, digitised, labels, teacher)[1]

        assert torch.isnan(label)
        assert served.item() == pytest.approx(teacher[1])

    def test_unlabelled_rows_train_only_the_consistency_term(self) -> None:
        clean, digitised, labels, teacher = _unlabelled_pairs()
        _, validation, validation_labels = _separable_pairs(n=16)

        history, _ = train_consistency(
            _seeded_stub(),
            PairedSignalDataset(clean, digitised, labels, teacher),
            validation,
            validation_labels,
            torch.device("cpu"),
            ConsistencyConfig(epochs=1, batch_size=8),
        )

        assert history[0].digitised_loss == 0.0
        assert history[0].clean_loss == 0.0
        assert history[0].consistency_loss > 0.0

    def test_unlabelled_corpus_is_mixed_in_without_drowning_supervision(self) -> None:
        """The reason for a fixed ratio: a large unlabelled set must not empty the BCE.

        Pooled into one loader, 200 unlabelled rows against 48 labelled would
        leave most batches with no label at all. Appended at a fixed size per
        batch, every step still carries its supervised terms.
        """
        clean, digitised, labels = _separable_pairs()
        teacher = np.where(labels == 1, 3.0, -3.0)
        model = _seeded_stub()
        device = torch.device("cpu")
        extra_clean, extra_digitised, extra_labels, _ = _unlabelled_pairs(n=200)
        # As in the real script, the unlabelled teacher is what the untouched
        # model said about the clean view.
        extra_teacher = predict_logits(model, extra_clean, device)
        extra = PairedSignalDataset(extra_clean, extra_digitised, extra_labels, extra_teacher)

        history, best = train_consistency(
            model,
            PairedSignalDataset(clean, digitised, labels, teacher),
            digitised,
            labels,
            device,
            ConsistencyConfig(epochs=4, batch_size=8, unlabelled_batch_size=8),
            unlabelled=extra,
        )

        assert all(record.digitised_loss > 0.0 for record in history)
        assert all(np.isfinite(record.total_loss) for record in history)
        assert best["auprc"] > 0.9

    def test_an_empty_unlabelled_set_is_ignored(self) -> None:
        clean, digitised, labels = _separable_pairs()
        empty = PairedSignalDataset(clean[:0], digitised[:0], labels[:0])

        history, _ = train_consistency(
            _seeded_stub(),
            PairedSignalDataset(clean, digitised, labels),
            digitised,
            labels,
            torch.device("cpu"),
            ConsistencyConfig(epochs=1, batch_size=8),
            unlabelled=empty,
        )

        assert len(history) == 1
