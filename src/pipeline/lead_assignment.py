# Lead-position maps for the standard paper ECG layouts.
# The segment-ensemble diagnosis path and the round-trip evaluator use these to
# know which lead sits in which column, so each 2.5 s column can be scored on
# its own instead of being tiled into a fake 10 s recording.

from __future__ import annotations

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
