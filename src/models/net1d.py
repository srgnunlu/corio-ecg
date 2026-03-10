# Net1D architecture from ECGFounder (PKUDigitalHealth)
# Source: https://github.com/PKUDigitalHealth/ECGFounder
# License: MIT
# Reconstructed from checkpoint weight shapes to match the original architecture

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    """Conv1d with SAME padding regardless of kernel size and stride."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.conv(x)


class MyMaxPool1dPadSame(nn.Module):
    """MaxPool1d with SAME padding."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.pool(x)


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class BasicBlock(nn.Module):
    """Pre-activation residual block with squeeze-and-excitation.

    Architecture (from checkpoint analysis):
        bn1(in) -> act -> conv1(in->out, 1x1) ->
        bn2(out) -> act -> conv2(out->out, kxk, grouped) ->
        SE attention -> residual add
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, groups_width: int,
                 downsample: bool, is_first_block: bool = False,
                 use_bn: bool = True, use_do: bool = True) -> None:
        super().__init__()
        self.downsample = downsample
        self.is_first_block = is_first_block
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Pre-activation + 1x1 channel projection (in -> out)
        self.bn1 = nn.BatchNorm1d(in_channels) if use_bn else nn.Identity()
        self.act1 = Swish()
        self.do1 = nn.Dropout(p=0.5) if use_do else nn.Identity()
        self.conv1 = MyConv1dPadSame(in_channels, out_channels, 1, 1)

        # Pre-activation + grouped kxk convolution (spatial)
        n_groups = out_channels // groups_width
        self.bn2 = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.act2 = Swish()
        self.do2 = nn.Dropout(p=0.5) if use_do else nn.Identity()
        self.conv2 = MyConv1dPadSame(out_channels, out_channels, kernel_size,
                                     stride, groups=n_groups)

        # Pre-activation + 1x1 mixing convolution (out -> out)
        self.bn3 = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.act3 = Swish()
        self.do3 = nn.Dropout(p=0.5) if use_do else nn.Identity()
        self.conv3 = MyConv1dPadSame(out_channels, out_channels, 1, 1)

        # Squeeze-and-excitation (reduction ratio 2)
        self.se_fc1 = nn.Linear(out_channels, out_channels // 2)
        self.se_fc2 = nn.Linear(out_channels // 2, out_channels)
        self.se_act = Swish()

        # Downsample for residual path
        if downsample:
            self.max_pool = MyMaxPool1dPadSame(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 1x1 channel projection
        out = self.do1(self.act1(self.bn1(x)))
        out = self.conv1(out)

        # Grouped kxk spatial convolution
        out = self.do2(self.act2(self.bn2(out)))
        out = self.conv2(out)

        # 1x1 mixing convolution
        out = self.do3(self.act3(self.bn3(out)))
        out = self.conv3(out)

        # Squeeze-and-excitation attention
        se = out.mean(dim=-1)  # global avg pool -> (B, C)
        se = self.se_act(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        out = torch.einsum('abc,ab->abc', out, se)

        # Residual connection
        if self.downsample:
            identity = self.max_pool(identity)
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch_diff = self.out_channels - self.in_channels
            identity = F.pad(identity, (0, ch_diff), "constant", 0)
            identity = identity.transpose(-1, -2)

        return out + identity


class BasicStage(nn.Module):
    """A stage of multiple BasicBlocks."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, groups_width: int,
                 n_blocks: int, is_first_stage: bool = False,
                 use_bn: bool = True, use_do: bool = True) -> None:
        super().__init__()
        layers = []
        for i in range(n_blocks):
            is_first = (i == 0)
            layers.append(BasicBlock(
                in_channels=in_channels if is_first else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride if is_first else 1,
                groups_width=groups_width,
                downsample=is_first,
                is_first_block=(is_first and is_first_stage),
                use_bn=use_bn,
                use_do=use_do,
            ))
        self.block_list = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block_list(x)


class Net1D(nn.Module):
    """Net1D classifier from ECGFounder — 1D CNN for 12-lead ECG signals."""

    def __init__(self, in_channels: int, base_filters: int, ratio: int,
                 filter_list: list[int], m_blocks_list: list[int],
                 kernel_size: int, stride: int, groups_width: int,
                 n_classes: int, use_bn: bool = True,
                 use_do: bool = True) -> None:
        super().__init__()
        # Initial convolution
        self.first_conv = MyConv1dPadSame(in_channels, base_filters,
                                          kernel_size, stride=2)
        self.first_bn = nn.BatchNorm1d(base_filters) if use_bn else nn.Identity()
        self.first_act = Swish()

        # Build stages
        self.stage_list = nn.ModuleList()
        in_ch = base_filters
        for i, (out_ch, n_blocks) in enumerate(zip(filter_list, m_blocks_list)):
            self.stage_list.append(BasicStage(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                groups_width=groups_width,
                n_blocks=n_blocks,
                is_first_stage=(i == 0),
                use_bn=use_bn,
                use_do=use_do,
            ))
            in_ch = out_ch

        # Classification head
        self.dense = nn.Linear(in_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.first_act(self.first_bn(self.first_conv(x)))
        for stage in self.stage_list:
            out = stage(out)
        out = out.mean(dim=-1)  # global average pooling
        return self.dense(out)
