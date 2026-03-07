# src/pipeline/digitize.py
# Paper ECG image to digital signal conversion
# Uses ECG-Digitiser (nnU-Net segmentation + Hough Transform)
# Source: https://github.com/felixkrones/ECG-Digitiser

from pathlib import Path

import numpy as np


class ECGDigitiser:
    """Converts paper ECG photographs to digital signals.

    Phase 1 placeholder. Full implementation requires:
    1. Clone ECG-Digitiser repo
    2. Install nnU-Net dependencies
    3. Download pre-trained segmentation model (~475 MB)

    Setup instructions:
        git clone https://github.com/felixkrones/ECG-Digitiser.git external/ecg-digitiser
        cd external/ecg-digitiser && pip install -e .
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else None
        self._model_loaded = False

    def digitize(self, image_path: str | Path) -> np.ndarray:
        """Convert a paper ECG image to a 12-lead digital signal.

        Args:
            image_path: Path to ECG image (PNG/JPG).

        Returns:
            signal: numpy array of shape (12, 5000), 500 Hz.

        Raises:
            NotImplementedError: Until ECG-Digitiser is fully integrated.
        """
        raise NotImplementedError(
            "ECG-Digitiser integration not yet complete. "
            "Use --signal flag with WFDB records for now. "
            "See docs/plans/2026-03-07-phase1-pipeline-setup-design.md"
        )
