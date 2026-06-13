# Regression tests for behavior required by the official ECGFounder Net1D.

from pathlib import Path

import pytest
import torch

from src.models.net1d import BasicBlock, Net1D
from src.pipeline.diagnose import MODEL_CONFIG

CHECKPOINT_PATH = Path("models/ecgfounder/base/12_lead_ECGFounder.pth")


def _zero_main_path(block: BasicBlock) -> None:
    for layer in (block.conv1.conv, block.conv2.conv, block.conv3.conv):
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)


def test_official_inference_disables_batch_norm_execution() -> None:
    assert MODEL_CONFIG["use_bn"] is False


def test_residual_channel_padding_is_centered() -> None:
    block = BasicBlock(
        in_channels=2,
        out_channels=4,
        ratio=1,
        kernel_size=1,
        stride=1,
        groups=1,
        downsample=False,
        use_bn=False,
        use_do=False,
    )
    _zero_main_path(block)
    input_tensor = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])

    output = block(input_tensor)

    expected = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]]
    )
    assert torch.equal(output, expected)


def test_first_block_skips_first_preactivation() -> None:
    first_block = BasicBlock(
        in_channels=2,
        out_channels=2,
        ratio=1,
        kernel_size=1,
        stride=1,
        groups=1,
        downsample=False,
        is_first_block=True,
        use_bn=False,
        use_do=False,
    )
    later_block = BasicBlock(
        in_channels=2,
        out_channels=2,
        ratio=1,
        kernel_size=1,
        stride=1,
        groups=1,
        downsample=False,
        is_first_block=False,
        use_bn=False,
        use_do=False,
    )
    later_block.load_state_dict(first_block.state_dict())
    input_tensor = -torch.ones((1, 2, 3))

    assert not torch.equal(first_block(input_tensor), later_block(input_tensor))


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="ECGFounder checkpoint not installed")
def test_checkpoint_golden_logits_match_official_execution() -> None:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = Net1D(**MODEL_CONFIG).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    input_tensor = torch.linspace(-1.0, 1.0, 12 * 5000).reshape(1, 12, 5000)
    expected = torch.tensor(
        [
            8.3751173,
            -4.1527028,
            -6.7856750,
            -2.8647900,
            4.4135675,
            -2.2547250,
            -5.3009405,
            -6.1225495,
            -1.0395477,
            -1.8894272,
        ]
    )

    with torch.no_grad():
        logits = model(input_tensor)

    assert torch.allclose(logits[0, :10], expected, atol=1e-5, rtol=1e-5)
