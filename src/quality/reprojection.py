"""Reference-free re-projection fidelity for digitized paper ECGs.

Internal-consistency probes (stability, perturbation, Goldberger redundancy) all
failed because a digitization can be self-consistent yet wrong. The one honest
reference-free fidelity signal is agreement with the *page*: re-render the
reconstructed traces and measure how well they overlay the ink the segmentation
model actually detected.

The segmentation probability map is upstream of the extracted polylines, so the
overlay is not circular — a hallucinated, mistracked, or off-grid reconstruction
sits off the ink (low precision), and a reconstruction that misses leads leaves
ink unexplained (low recall).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

# Probability above which a segmentation pixel counts as ink. The signal
# probability map is conservative — its values rarely exceed ~0.55 even on clean
# traces — so a 0.5 cutoff leaves an artificially sparse, intermittent mask that
# makes faithful traces look off-ink. 0.25 captures the trace body without
# pulling in the (separately-channelled) grid.
DEFAULT_PROBABILITY_THRESHOLD: float = 0.25
# Vertical tolerance (pixels) for a reconstructed pixel to "sit on" ink. Tight,
# because the extracted polyline should track the centre of a thin trace.
DEFAULT_PRECISION_TOLERANCE: int = 3
# Vertical tolerance (pixels) for ink to be "explained" by a reconstructed
# pixel. Looser, to absorb the finite thickness of a printed trace.
DEFAULT_RECALL_TOLERANCE: int = 8


@dataclass(frozen=True)
class ReprojectionFidelity:
    """Image-space agreement between reconstructed traces and detected ink.

    ``precision`` is the fraction of reconstructed pixels landing on ink (low =
    off-page / hallucinated signal). ``recall`` is the fraction of ink explained
    by the reconstruction (low = missed or under-segmented leads). ``f1`` is
    their harmonic mean and ``residual = 1 - f1`` (higher = worse) so it aligns
    with the other quality features.
    """

    precision: float
    recall: float
    f1: float
    residual: float
    ink_pixels: int
    reconstructed_pixels: int
    trace_count: int


def _render_traces(raw_lines: np.ndarray, crop_x0: int, height: int, width: int) -> np.ndarray:
    """Rasterize extracted pixel-Y polylines into the segmentation frame."""
    rendered = np.zeros((height, width), dtype=bool)
    if raw_lines.ndim != 2 or raw_lines.shape[0] == 0:
        return rendered
    columns = np.arange(raw_lines.shape[1])
    for trace in raw_lines:
        finite = np.isfinite(trace)
        if not finite.any():
            continue
        x = columns[finite] + crop_x0
        y = np.rint(trace[finite]).astype(np.int64)
        in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        rendered[y[in_bounds], x[in_bounds]] = True
    return rendered


def reprojection_fidelity(
    signal_probability: np.ndarray,
    raw_lines: np.ndarray,
    extraction_crop_x0: int = 0,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
    precision_tolerance: int = DEFAULT_PRECISION_TOLERANCE,
    recall_tolerance: int = DEFAULT_RECALL_TOLERANCE,
) -> ReprojectionFidelity:
    """Score how well reconstructed traces overlay the detected ink.

    Args:
        signal_probability: ``(H, W)`` segmentation probability map (the frame
            the polylines were extracted from).
        raw_lines: ``(n_traces, W_raw)`` extracted pixel-Y positions, NaN where a
            trace is absent. Columns map to ink columns via ``extraction_crop_x0``.
        extraction_crop_x0: Leading-column crop offset (``raw_lines`` column ``c``
            corresponds to ink column ``c + extraction_crop_x0``).
        probability_threshold: Ink cutoff on the probability map.
        precision_tolerance: Vertical px tolerance for a reconstructed pixel to
            count as sitting on ink.
        recall_tolerance: Vertical px tolerance for ink to count as explained.

    Returns:
        A :class:`ReprojectionFidelity`. Degenerate inputs (no ink or no traces)
        yield zero precision/recall and ``residual = 1.0``.
    """
    if signal_probability.ndim != 2:
        raise ValueError(f"signal_probability must be 2-D, got {signal_probability.shape}")

    height, width = signal_probability.shape
    ink = signal_probability >= probability_threshold
    rendered = _render_traces(raw_lines, int(extraction_crop_x0), height, width)
    trace_count = int(raw_lines.shape[0]) if raw_lines.ndim == 2 else 0

    ink_count = int(ink.sum())
    rendered_count = int(rendered.sum())
    if ink_count == 0 or rendered_count == 0:
        return ReprojectionFidelity(0.0, 0.0, 0.0, 1.0, ink_count, rendered_count, trace_count)

    ink_dilated = maximum_filter1d(ink.astype(np.uint8), size=2 * precision_tolerance + 1, axis=0)
    rendered_dilated = maximum_filter1d(
        rendered.astype(np.uint8), size=2 * recall_tolerance + 1, axis=0
    )

    precision = float(ink_dilated[rendered].astype(bool).mean())
    recall = float(rendered_dilated[ink].astype(bool).mean())
    denominator = precision + recall
    f1 = float(2.0 * precision * recall / denominator) if denominator > 0 else 0.0
    return ReprojectionFidelity(
        precision=precision,
        recall=recall,
        f1=f1,
        residual=float(1.0 - f1),
        ink_pixels=ink_count,
        reconstructed_pixels=rendered_count,
        trace_count=trace_count,
    )
