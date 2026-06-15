"""Tests for conservative ECG page perspective correction."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from src.pipeline.perspective_correction import correct_perspective


def _trapezoid_page() -> torch.Tensor:
    image = np.full((700, 1000, 3), 25, dtype=np.uint8)
    page = np.full((500, 800, 3), 235, dtype=np.uint8)
    page[::20, :] = 150
    page[:, ::20] = 150
    source = np.float32([[0, 0], [799, 0], [799, 499], [0, 499]])
    target = np.float32([[140, 100], [900, 40], [850, 620], [90, 580]])
    transform = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(page, transform, (1000, 700))
    mask = cv2.warpPerspective(np.full((500, 800), 255, dtype=np.uint8), transform, (1000, 700))
    image[mask > 0] = warped[mask > 0]
    return torch.from_numpy(image).permute(2, 0, 1)


def test_correct_perspective_rectifies_high_confidence_page() -> None:
    result = correct_perspective(_trapezoid_page())

    assert result.applied is True
    assert result.confidence >= 0.80
    assert result.image.shape[2] > result.image.shape[1]
    assert result.image.shape[1] < 700


def test_correct_perspective_leaves_full_frame_page_unchanged() -> None:
    image = torch.full((3, 700, 1000), 235, dtype=torch.uint8)
    image[:, ::20, :] = 150
    image[:, :, ::20] = 150

    result = correct_perspective(image)

    assert result.applied is False
    assert result.image.data_ptr() == image.data_ptr()


def test_correct_perspective_leaves_ambiguous_image_unchanged() -> None:
    image = torch.zeros((3, 700, 1000), dtype=torch.uint8)
    image[:, 100:300, 100:350] = 230
    image[:, 400:600, 650:900] = 230

    result = correct_perspective(image)

    assert result.applied is False
