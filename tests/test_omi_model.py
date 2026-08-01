# Tests for the OMI classifier head, staged unfreezing, and the training helpers.

import numpy as np
import pytest
import torch

from src.models.net1d import Net1D
from src.omi.model import FEATURE_DIM, OmiClassifier
from src.omi.training import SignalDataset, TrainingConfig, positive_weight

# A deliberately small backbone: these tests cover wiring, not accuracy, and the
# real 76M-parameter checkpoint would make them slow.
SMALL_CONFIG = {
    "in_channels": 12,
    "base_filters": 8,
    "ratio": 1,
    "filter_list": [8, 16, FEATURE_DIM],
    "m_blocks_list": [1, 1, 1],
    "kernel_size": 16,
    "stride": 2,
    "groups_width": 8,
    "n_classes": 150,
    "use_bn": False,
    "use_do": False,
}


def _classifier() -> OmiClassifier:
    return OmiClassifier(Net1D(**SMALL_CONFIG))


class TestNet1DFeatures:
    """Tests for the representation exposed for transfer."""

    def test_forward_features_returns_pooled_vector(self) -> None:
        model = Net1D(**SMALL_CONFIG)
        features = model.forward_features(torch.zeros(2, 12, 5000))
        assert features.shape == (2, FEATURE_DIM)

    def test_forward_still_returns_class_logits(self) -> None:
        """Refactoring forward() must not change the inference path's output."""
        model = Net1D(**SMALL_CONFIG)
        assert model(torch.zeros(2, 12, 5000)).shape == (2, 150)


class TestOmiClassifier:
    """Tests for the binary head and freezing behaviour."""

    def test_forward_returns_one_logit_per_record(self) -> None:
        assert _classifier()(torch.zeros(3, 12, 5000)).shape == (3,)

    def test_freeze_backbone_leaves_only_the_head_trainable(self) -> None:
        model = _classifier()
        model.freeze_backbone()
        trainable = model.trainable_parameter_count()
        head_only = sum(p.numel() for p in model.head.parameters())
        assert trainable == head_only

    def test_unfreeze_reopens_the_last_stages(self) -> None:
        model = _classifier()
        model.freeze_backbone()
        frozen_count = model.trainable_parameter_count()

        unfrozen = model.unfreeze_last_stages(2)
        assert len(unfrozen) == 2
        assert unfrozen == ["stage_1", "stage_2"]
        assert model.trainable_parameter_count() > frozen_count

    def test_unfreeze_rejects_out_of_range_counts(self) -> None:
        model = _classifier()
        with pytest.raises(ValueError, match="count must be in"):
            model.unfreeze_last_stages(0)
        with pytest.raises(ValueError, match="count must be in"):
            model.unfreeze_last_stages(99)

    def test_early_stages_stay_frozen_after_partial_unfreeze(self) -> None:
        """Generic waveform features must survive a 14k-record fine-tune."""
        model = _classifier()
        model.freeze_backbone()
        model.unfreeze_last_stages(1)
        first_stage = list(model.backbone.stage_list)[0]
        assert all(not p.requires_grad for p in first_stage.parameters())


class TestPositiveWeight:
    """Tests for the class-imbalance weight."""

    def test_matches_negative_to_positive_ratio(self) -> None:
        labels = np.array([1] * 10 + [0] * 90)
        assert positive_weight(labels) == pytest.approx(9.0)

    def test_balanced_labels_give_weight_one(self) -> None:
        assert positive_weight(np.array([1, 0])) == pytest.approx(1.0)

    def test_no_positives_falls_back_to_one(self) -> None:
        assert positive_weight(np.zeros(10)) == 1.0


class TestSignalDataset:
    """Tests for the cached-signal dataset."""

    def test_serves_requested_rows_only(self) -> None:
        signals = np.arange(4 * 12 * 5000, dtype=np.float16).reshape(4, 12, 5000)
        labels = np.array([0, 1, 0, 1])
        dataset = SignalDataset(signals, labels, np.array([1, 3]))

        assert len(dataset) == 2
        _, first_label = dataset[0]
        assert float(first_label) == 1.0

    def test_converts_float16_cache_to_float32(self) -> None:
        signals = np.ones((2, 12, 5000), dtype=np.float16)
        dataset = SignalDataset(signals, np.array([0, 1]), np.array([0, 1]))
        signal, _ = dataset[0]
        assert signal.dtype == torch.float32


class TestTrainingConfig:
    """Tests for the training configuration defaults."""

    def test_defaults_train_head_before_unfreezing(self) -> None:
        config = TrainingConfig()
        assert config.head_epochs > 0
        assert config.backbone_learning_rate < config.head_learning_rate
