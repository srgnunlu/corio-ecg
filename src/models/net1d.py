# Net1D architecture from ECGFounder (PKUDigitalHealth)
# Source: https://github.com/PKUDigitalHealth/ECGFounder
# License: MIT
# Adapted for Corio ECG pipeline — removed MyDataset, added type hints

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
    """Bottleneck block with squeeze-and-excitation attention."""

    def __init__(self, in_channels: int, out_channels: int, ratio: int,
                 kernel_size: int, stride: int, groups: int,
                 downsample: bool, is_first_block: bool = False,
                 use_bn: bool = True, use_do: bool = True) -> None:
        super().__init__()
        self.downsample = downsample
        self.is_first_block = is_first_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.use_do = use_do

        mid_channels = out_channels // ratio
        # 1x1 down
        self.conv1 = MyConv1dPadSame(in_channels, mid_channels, 1, 1)
        self.bn1 = nn.BatchNorm1d(mid_channels) if use_bn else nn.Identity()
        self.act1 = Swish()
        self.do1 = nn.Dropout(p=0.5) if use_do else nn.Identity()
        # kxk conv
        self.convk = MyConv1dPadSame(mid_channels, mid_channels, kernel_size,
                                     stride, groups=mid_channels // groups)
        self.bnk = nn.BatchNorm1d(mid_channels) if use_bn else nn.Identity()
        self.actk = Swish()
        self.dok = nn.Dropout(p=0.5) if use_do else nn.Identity()
        # 1x1 up
        self.conv2 = MyConv1dPadSame(mid_channels, out_channels, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.act2 = Swish()
        self.do2 = nn.Dropout(p=0.5) if use_do else nn.Identity()
        # SE attention (reduction ratio 2)
        self.se_fc1 = nn.Linear(out_channels, out_channels // 2)
        self.se_fc2 = nn.Linear(out_channels // 2, out_channels)
        self.se_act = Swish()
        # Downsample for residual path
        if downsample:
            self.max_pool = MyMaxPool1dPadSame(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        # Main path
        out = self.do1(self.act1(self.bn1(self.conv1(x))))
        out = self.dok(self.actk(self.bnk(self.convk(out))))
        out = self.do2(self.act2(self.bn2(self.conv2(out))))
        # SE attention
        se = out.mean(dim=-1)  # global avg pool -> (B, C)
        se = self.se_act(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        out = torch.einsum('abc,ab->abc', out, se)
        # Residual connection
        if self.downsample:
            identity = self.max_pool(identity)
        if self.out_channels != self.in_channels:
            # Pad channels: (B, C, T) -> transpose -> pad -> transpose back
            identity = identity.transpose(-1, -2)
            ch_diff = self.out_channels - self.in_channels
            identity = F.pad(identity, (0, ch_diff), "constant", 0)
            identity = identity.transpose(-1, -2)
        # Skip first block residual when channels mismatch at very start
        if self.is_first_block:
            out = out + identity
        else:
            out = out + identity
        return out


class BasicStage(nn.Module):
    """A stage of multiple BasicBlocks."""

    def __init__(self, in_channels: int, out_channels: int, ratio: int,
                 kernel_size: int, stride: int, groups: int,
                 n_blocks: int, is_first_stage: bool = False,
                 use_bn: bool = True, use_do: bool = True) -> None:
        super().__init__()
        layers = []
        for i in range(n_blocks):
            is_first = (i == 0)
            layers.append(BasicBlock(
                in_channels=in_channels if is_first else out_channels,
                out_channels=out_channels,
                ratio=ratio,
                kernel_size=kernel_size,
                stride=stride if is_first else 1,
                groups=groups,
                downsample=is_first,
                is_first_block=(is_first and is_first_stage),
                use_bn=use_bn,
                use_do=use_do,
            ))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


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
        self.stages = nn.ModuleList()
        in_ch = base_filters
        for i, (out_ch, n_blocks) in enumerate(zip(filter_list, m_blocks_list)):
            self.stages.append(BasicStage(
                in_channels=in_ch,
                out_channels=out_ch,
                ratio=ratio,
                kernel_size=kernel_size,
                stride=stride,
                groups=groups_width,
                n_blocks=n_blocks,
                is_first_stage=(i == 0),
                use_bn=use_bn,
                use_do=use_do,
            ))
            in_ch = out_ch
        # Classification head
        self.dense = nn.Linear(in_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, in_channels, seq_len)
        out = self.first_act(self.first_bn(self.first_conv(x)))
        for stage in self.stages:
            out = stage(out)
        # Global average pooling over time dimension
        out = out.mean(dim=-1)
        return self.dense(out)
