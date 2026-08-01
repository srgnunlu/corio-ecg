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
        _, first_label, _, _ = dataset[0]
        assert float(first_label) == 1.0

    def test_converts_float16_cache_to_float32(self) -> None:
        signals = np.ones((2, 12, 5000), dtype=np.float16)
        dataset = SignalDataset(signals, np.array([0, 1]), np.array([0, 1]))
        signal, _, _, _ = dataset[0]
        assert signal.dtype == torch.float32

    def test_auxiliary_is_empty_when_not_supplied(self) -> None:
        """Single-task runs must keep working through the same interface."""
        dataset = SignalDataset(
            np.ones((2, 12, 5000), dtype=np.float16), np.array([0, 1]), np.array([0, 1])
        )
        _, _, auxiliary, weight = dataset[0]
        assert auxiliary.numel() == 0
        assert float(weight) == 1.0

    def test_serves_auxiliary_targets_and_weights(self) -> None:
        auxiliary = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        weights = np.array([1.0, 3.0], dtype=np.float32)
        dataset = SignalDataset(
            np.ones((2, 12, 5000), dtype=np.float16),
            np.array([0, 1]),
            np.array([0, 1]),
            auxiliary,
            weights,
        )
        _, _, aux_row, weight = dataset[1]
        assert aux_row.tolist() == [0.0, 1.0]
        assert float(weight) == 3.0


class TestTrainingConfig:
    """Tests for the training configuration defaults."""

    def test_defaults_train_head_before_unfreezing(self) -> None:
        config = TrainingConfig()
        assert config.head_epochs > 0
        assert config.backbone_learning_rate < config.head_learning_rate


class TestMultiTaskHeads:
    """Tests for the MLP head and auxiliary branches."""

    def test_mlp_head_still_returns_one_logit(self) -> None:
        model = OmiClassifier(Net1D(**SMALL_CONFIG), hidden_dim=32)
        assert model(torch.zeros(2, 12, 5000)).shape == (2,)

    def test_mlp_head_has_more_capacity_than_linear(self) -> None:
        linear = OmiClassifier(Net1D(**SMALL_CONFIG))
        mlp = OmiClassifier(Net1D(**SMALL_CONFIG), hidden_dim=32)
        linear_params = sum(p.numel() for p in linear.head.parameters())
        mlp_params = sum(p.numel() for p in mlp.head.parameters())
        assert mlp_params > linear_params

    def test_auxiliary_head_returns_one_logit_per_target(self) -> None:
        model = OmiClassifier(Net1D(**SMALL_CONFIG), auxiliary_targets=7)
        omi, auxiliary = model.forward_with_auxiliary(torch.zeros(2, 12, 5000))
        assert omi.shape == (2,)
        assert auxiliary.shape == (2, 7)

    def test_auxiliary_is_none_when_disabled(self) -> None:
        """Single-task runs must not grow an unused branch."""
        _, auxiliary = _classifier().forward_with_auxiliary(torch.zeros(2, 12, 5000))
        assert auxiliary is None

    def test_forward_matches_forward_with_auxiliary(self) -> None:
        model = OmiClassifier(Net1D(**SMALL_CONFIG), auxiliary_targets=3)
        model.eval()
        signals = torch.zeros(2, 12, 5000)
        with torch.no_grad():
            assert torch.allclose(model(signals), model.forward_with_auxiliary(signals)[0])

    def test_head_parameters_include_the_auxiliary_branch(self) -> None:
        """Stage 2's optimiser group must not silently drop the aux head."""
        model = OmiClassifier(Net1D(**SMALL_CONFIG), hidden_dim=32, auxiliary_targets=3)
        counted = sum(p.numel() for p in model.head_parameters())
        expected = sum(p.numel() for p in model.head.parameters()) + sum(
            p.numel() for p in model.auxiliary_head.parameters()
        )
        assert counted == expected


class TestAuxiliaryMatrix:
    """Tests for assembling auxiliary targets from the label table."""

    def _table(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "STEMI": [1, 0], "NSTEMI": [0, 1], "AMI": [1, 1], "CTO": [0, 0],
                "PLAD": [1, 0], "MLAD": [0, 0], "DLAD": [0, 0], "DB": [0, 0],
                "PLCX": [0, 0], "MLCX": [0, 0], "DLCX": [0, 0], "OM": [0, 0],
                "PRCA": [0, 1], "MRCA": [0, 0], "DRCA": [0, 0],
            }
        )

    def test_collapses_segments_into_territories(self) -> None:
        from src.omi.model import build_auxiliary_matrix

        matrix, names = build_auxiliary_matrix(self._table())
        assert names == ["STEMI", "NSTEMI", "AMI", "CTO",
                         "culprit_lad", "culprit_lcx", "culprit_rca"]
        assert matrix.shape == (2, 7)
        # Record 0 has a proximal LAD culprit, record 1 a proximal RCA one.
        assert matrix[0, names.index("culprit_lad")] == 1.0
        assert matrix[1, names.index("culprit_rca")] == 1.0
        assert matrix[0, names.index("culprit_rca")] == 0.0


class TestNstemiSampleWeights:
    """Tests for NSTEMI-OMI up-weighting."""

    def test_only_nstemi_omi_records_are_weighted(self) -> None:
        from src.omi.training import nstemi_sample_weights

        omi = np.array([1, 1, 0, 0])
        nstemi = np.array([1, 0, 1, 0])
        weights = nstemi_sample_weights(omi, nstemi, 3.0)
        assert weights.tolist() == [3.0, 1.0, 1.0, 1.0]

    def test_weight_one_is_a_no_op(self) -> None:
        from src.omi.training import nstemi_sample_weights

        weights = nstemi_sample_weights(np.array([1, 0]), np.array([1, 1]), 1.0)
        assert weights.tolist() == [1.0, 1.0]
