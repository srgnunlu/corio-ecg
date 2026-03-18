# Position-based lead assignment for known ECG paper layouts.
# When the digitizer's Lead Name U-Net cannot read text labels reliably,
# this module assigns leads based on their physical position on the paper
# using the known layout geometry selected by the user.

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Standard 12-lead indices in ECGFounder order:
# I=0, II=1, III=2, aVR=3, aVL=4, aVF=5, V1=6, V2=7, V3=8, V4=9, V5=10, V6=11

# Layout maps: each row lists lead indices left-to-right.
# Row order is top-to-bottom as they appear on paper.
LAYOUT_3X4: list[list[int]] = [
    [0, 3, 6, 9],    # Row 0: I, aVR, V1, V4
    [1, 4, 7, 10],   # Row 1: II, aVL, V2, V5
    [2, 5, 8, 11],   # Row 2: III, aVF, V3, V6
]

LAYOUT_6X2: list[list[int]] = [
    [0, 6],    # Row 0: I, V1
    [1, 7],    # Row 1: II, V2
    [2, 8],    # Row 2: III, V3
    [3, 9],    # Row 3: aVR, V4
    [4, 10],   # Row 4: aVL, V5
    [5, 11],   # Row 5: aVF, V6
]

RHYTHM_LEAD_INDEX: int = 1  # Lead II is the standard rhythm strip


def _select_layout(layout_hint: str) -> tuple[list[list[int]], int, int] | None:
    """Return (lead_map, n_rows, n_cols) for a known layout, or None."""
    if "3x4" in layout_hint:
        return LAYOUT_3X4, 3, 4
    if "6x2" in layout_hint:
        return LAYOUT_6X2, 6, 2
    return None


def override_lead_assignment(
    canonical: torch.Tensor,
    raw_lines: torch.Tensor | None,
    layout_hint: str,
) -> torch.Tensor:
    """Build canonical 12-lead signal from raw traces using known layout.

    The digitizer's raw_lines are signal traces sorted by vertical position
    (top to bottom). Each trace spans the full recording width. When the
    layout is known, we split each trace into column segments and assign
    directly to the correct lead index — no U-Net text detection needed.

    Args:
        canonical: (12, N) original canonical from the digitizer (fallback).
        raw_lines: (n_traces, N) raw signal traces sorted top-to-bottom.
        layout_hint: Layout type ("3x4+1R", "6x2+1R", etc.).

    Returns:
        (12, N) tensor with leads assigned by physical position.
    """
    if raw_lines is None:
        logger.warning("No raw_lines available — keeping digitizer assignment")
        return canonical

    layout_info = _select_layout(layout_hint)
    if layout_info is None:
        logger.debug("Unknown layout '%s' — keeping digitizer assignment", layout_hint)
        return canonical

    lead_map, n_rows, n_cols = layout_info
    n_traces = raw_lines.shape[0]
    n_samples = raw_lines.shape[1]

    # Verify we have enough traces for this layout
    if n_traces < n_rows:
        logger.warning(
            "Expected >= %d traces for %s, got %d — keeping digitizer assignment",
            n_rows, layout_hint, n_traces,
        )
        return canonical

    # Build new canonical: split each row trace into column segments
    new_canonical = torch.full(
        (12, n_samples), float("nan"), dtype=raw_lines.dtype,
    )
    col_width = n_samples // n_cols

    for row_idx in range(n_rows):
        trace = raw_lines[row_idx]
        for col_idx in range(n_cols):
            lead_idx = lead_map[row_idx][col_idx]
            start = col_idx * col_width
            end = start + col_width
            new_canonical[lead_idx, start:end] = trace[start:end]

    # Rhythm strip (last trace after row traces) → Lead II full-length
    if n_traces > n_rows:
        rhythm_trace = raw_lines[n_rows]
        if torch.isfinite(rhythm_trace).any():
            new_canonical[RHYTHM_LEAD_INDEX] = rhythm_trace

    logger.info(
        "Position-based lead assignment applied (%s layout, %d traces)",
        layout_hint, n_traces,
    )
    return new_canonical
