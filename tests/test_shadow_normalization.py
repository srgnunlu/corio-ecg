"""Tests for conservative broad-shadow normalization."""

from __future__ import annotations

import numpy as np
import torch

from src.pipeline.shadow_normalization import normalize_broad_shadow


def test_normalize_broad_shadow_reduces_large_luminance_gradient() -> None:
    width = 1000
    gradient = np.linspace(100, 240, width, dtype=np.uint8)
    image = np.tile(gradient, (700, 1))
    image[::20, :] = image[::20, :] // 2
    tensor = torch.from_numpy(np.stack([image, image, image])).to(torch.uint8)

    result = normalize_broad_shadow(tensor)
    output = result.image.float().mean(dim=0)

    assert result.applied is True
    assert result.shadow_score >= 0.20
    assert abs(output[:, :100].mean() - output[:, -100:].mean()) < 90


def test_normalize_broad_shadow_leaves_uniform_page_unchanged() -> None:
    image = torch.full((3, 700, 1000), 220, dtype=torch.uint8)
    image[:, ::20, :] = 150
    image[:, :, ::20] = 150

    result = normalize_broad_shadow(image)

    assert result.applied is False
    assert result.image.data_ptr() == image.data_ptr()


def test_normalize_broad_shadow_leaves_dark_non_page_image_unchanged() -> None:
    image = torch.full((3, 700, 1000), 40, dtype=torch.uint8)
    image[:, :, 500:] = 90

    result = normalize_broad_shadow(image)

    assert result.applied is False
