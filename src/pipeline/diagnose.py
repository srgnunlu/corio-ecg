# ECGFounder inference module — loads Net1D model and runs 150-class diagnosis
# Input: z-score normalized 12-lead ECG signal (12, 5000)
# Output: list of DiagnosisResult sorted by probability descending

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.models.net1d import Net1D
from src.utils.ecg_labels import DEFAULT_THRESHOLD, ECG_FOUNDER_LABELS, NUM_CLASSES
from src.utils.rhythm import estimate_heart_rate_bpm

logger = logging.getLogger(__name__)

_LABEL_TO_INDEX: dict[str, int] = {
    label: index for index, label in enumerate(ECG_FOUNDER_LABELS)
}
_BRADYCARDIA_LABELS: tuple[str, ...] = (
    "SINUS BRADYCARDIA",
    "MARKED SINUS BRADYCARDIA",
    "JUNCTIONAL BRADYCARDIA",
)
_TACHYCARDIA_LABELS: tuple[str, ...] = (
    "SINUS TACHYCARDIA",
    "SUPRAVENTRICULAR TACHYCARDIA",
    "WIDE QRS TACHYCARDIA",
    "VENTRICULAR TACHYCARDIA",
    "MULTIFOCAL ATRIAL TACHYCARDIA",
)
_NORMALISH_LABELS: tuple[str, ...] = (
    "NORMAL SINUS RHYTHM",
    "SINUS RHYTHM",
    "NORMAL ECG",
    "OTHERWISE NORMAL ECG",
)

MODEL_CONFIG: dict = {
    "in_channels": 12,
    "base_filters": 64,
    "ratio": 1,
    "filter_list": [64, 160, 160, 400, 400, 1024, 1024],
    "m_blocks_list": [2, 2, 2, 3, 3, 4, 4],
    "kernel_size": 16,
    "stride": 2,
    "groups_width": 16,
    "n_classes": 150,
    "use_bn": True,
    "use_do": False,
}


@dataclass
class DiagnosisResult:
    """Single diagnosis prediction from ECGFounder."""

    label: str
    index: int
    probability: float


def get_device() -> torch.device:
    """Return the best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ECGDiagnoser:
    """Runs ECGFounder inference on a 12-lead ECG signal."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.device = device or get_device()
        self.threshold = threshold
        self.last_estimated_hr_bpm: float | None = None
        self.model = self._load_model(Path(checkpoint_path))
        logger.info(
            "ECGDiagnoser ready — device=%s, threshold=%.2f",
            self.device,
            self.threshold,
        )

    def _load_model(self, checkpoint_path: Path) -> Net1D:
        """Load Net1D from checkpoint, handling multiple saved formats."""
        model = Net1D(**MODEL_CONFIG)

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Extract state_dict from various checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Assume the dict itself is a state_dict
                state_dict = checkpoint
        else:
            # Raw model object — extract its state_dict
            state_dict = checkpoint.state_dict()

        # Remove "module." prefix added by DataParallel wrapping
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.removeprefix("module.")
            cleaned_state_dict[clean_key] = value

        model.load_state_dict(cleaned_state_dict, strict=True)
        model.to(self.device)
        # Set model to inference mode (no dropout, frozen batch norm)
        model.train(False)

        logger.info("Model loaded from %s", checkpoint_path)
        return model

    def diagnose(
        self,
        signal: np.ndarray,
        threshold: float | None = None,
    ) -> list[DiagnosisResult]:
        """Run multi-label diagnosis on a 12-lead ECG signal.

        Args:
            signal: Z-score normalized array with shape (12, 5000).
            threshold: Sigmoid probability cutoff. Uses instance default if None.

        Returns:
            Diagnosis results above threshold, sorted by probability descending.
        """
        effective_threshold = threshold if threshold is not None else self.threshold

        # Prepare input tensor: (12, 5000) -> (1, 12, 5000)
        tensor = torch.tensor(signal, dtype=torch.float32, device=self.device)
        tensor = tensor.unsqueeze(0)

        # Inference — multi-label so we use sigmoid, not softmax
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        probabilities = self._apply_rate_consistency_adjustments(probabilities, signal)

        # Collect results above threshold
        results: list[DiagnosisResult] = []
        for index in range(NUM_CLASSES):
            probability = float(probabilities[index])
            if probability >= effective_threshold:
                results.append(
                    DiagnosisResult(
                        label=ECG_FOUNDER_LABELS[index],
                        index=index,
                        probability=probability,
                    )
                )

        # Sort by probability descending
        results.sort(key=lambda r: r.probability, reverse=True)
        return results

    def diagnose_all(self, signal: np.ndarray) -> list[DiagnosisResult]:
        """Return all 150 diagnoses sorted by probability (no threshold filtering).

        Args:
            signal: Z-score normalized array with shape (12, 5000).

        Returns:
            All 150 diagnosis results sorted by probability descending.
        """
        return self.diagnose(signal, threshold=0.0)

    def _apply_rate_consistency_adjustments(
        self,
        probabilities: np.ndarray,
        signal: np.ndarray,
    ) -> np.ndarray:
        """Down-weight diagnoses that contradict an obvious heart-rate regime."""
        adjusted = probabilities.copy()
        estimated_hr = estimate_heart_rate_bpm(signal)
        self.last_estimated_hr_bpm = estimated_hr
        if estimated_hr is None:
            return adjusted

        if estimated_hr >= 110.0:
            self._scale_labels(adjusted, _BRADYCARDIA_LABELS, factor=0.05)
            self._scale_labels(adjusted, ("NORMAL ECG", "OTHERWISE NORMAL ECG"), factor=0.1)
            if estimated_hr >= 130.0:
                self._scale_labels(adjusted, ("NORMAL SINUS RHYTHM",), factor=0.1)
                self._scale_labels(adjusted, ("SINUS RHYTHM",), factor=0.2)

        if estimated_hr <= 55.0:
            self._scale_labels(adjusted, _TACHYCARDIA_LABELS, factor=0.05)
            self._scale_labels(adjusted, ("NORMAL ECG", "OTHERWISE NORMAL ECG"), factor=0.1)
            if estimated_hr <= 45.0:
                self._scale_labels(adjusted, ("NORMAL SINUS RHYTHM",), factor=0.2)
                self._scale_labels(adjusted, ("SINUS RHYTHM",), factor=0.3)

        if estimated_hr < 50.0 or estimated_hr > 100.0:
            self._scale_labels(adjusted, _NORMALISH_LABELS, factor=0.25)

        return adjusted

    @staticmethod
    def _scale_labels(
        probabilities: np.ndarray,
        labels: tuple[str, ...],
        factor: float,
    ) -> None:
        """Scale selected diagnosis probabilities in-place."""
        for label in labels:
            label_index = _LABEL_TO_INDEX.get(label)
            if label_index is not None:
                probabilities[label_index] *= factor
