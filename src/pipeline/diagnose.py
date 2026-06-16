# ECGFounder inference module — loads Net1D model and runs 150-class diagnosis
# Input: z-score normalized 12-lead ECG signal (12, 5000)
# Output: list of DiagnosisResult sorted by probability descending

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from src.models.net1d import Net1D
from src.pipeline.lead_assignment import LAYOUT_3X4, LAYOUT_6X2
from src.utils.ecg_labels import DEFAULT_THRESHOLD, ECG_FOUNDER_LABELS, NUM_CLASSES
from src.utils.rhythm import estimate_heart_rate_bpm, estimate_rhythm_hr

logger = logging.getLogger(__name__)

# Paper-column lead groupings for segment-ensemble inference. Each entry maps
# a layout substring to the rows-of-lead-indices that share a printed column.
# Leads in the same column were recorded in the same time window, so grouping
# them keeps cross-lead features (axis, bundle blocks) temporally coherent —
# unlike feeding all 12 tiled-from-different-windows leads to the model at once.
PAPER_COLUMN_LAYOUTS: dict[str, list[list[int]]] = {
    "3x4": LAYOUT_3X4,
    "6x2": LAYOUT_6X2,
}
DEFAULT_SEGMENT_LAYOUT: str = "3x4"
DEFAULT_SEGMENT_AGGREGATION: str = "mean"

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
    # Official ECGFounder PTB-XL evaluation constructs BatchNorm parameters
    # from the checkpoint but disables BatchNorm execution during inference.
    "use_bn": False,
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
        *,
        apply_rate_adjustments: bool = True,
        rhythm_strip: np.ndarray | None = None,
    ) -> list[DiagnosisResult]:
        """Run multi-label diagnosis on a 12-lead ECG signal.

        Args:
            signal: Z-score normalized array with shape (12, 5000).
            threshold: Sigmoid probability cutoff. Uses instance default if None.
            apply_rate_adjustments: Apply heart-rate consistency heuristics. Keep
                enabled for the UI; disable for raw model benchmarking.
            rhythm_strip: Optional full-duration rhythm strip (uncropped) used
                for heart-rate estimation. When provided, HR comes from the true
                10 s RR sequence instead of the tiled diagnosis signal.

        Returns:
            Diagnosis results above threshold, sorted by probability descending.
        """
        effective_threshold = threshold if threshold is not None else self.threshold

        probabilities = self._forward_probabilities(signal)
        if apply_rate_adjustments:
            probabilities = self._apply_rate_consistency_adjustments(
                probabilities, signal, rhythm_strip=rhythm_strip
            )
        else:
            self.last_estimated_hr_bpm = None

        return self._collect_results(probabilities, effective_threshold)

    def _forward_probabilities(self, signal: np.ndarray) -> np.ndarray:
        """Run a single ECGFounder forward pass and return 150-class sigmoid probs.

        Aligns each lead's active data to start at sample 0 (paper layouts place
        leads at different time offsets), then runs the multi-label model.
        """
        # Align leads so all active data starts at sample 0 for cross-lead features.
        model_signal = _align_leads_for_model(signal)

        # Prepare input tensor: (12, 5000) -> (1, 12, 5000)
        tensor = torch.tensor(model_signal, dtype=torch.float32, device=self.device)
        tensor = tensor.unsqueeze(0)

        # Inference — multi-label so we use sigmoid, not softmax
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return probabilities

    @staticmethod
    def _collect_results(
        probabilities: np.ndarray,
        threshold: float,
    ) -> list[DiagnosisResult]:
        """Build sorted DiagnosisResult list for probabilities above threshold."""
        results: list[DiagnosisResult] = []
        for index in range(NUM_CLASSES):
            probability = float(probabilities[index])
            if probability >= threshold:
                results.append(
                    DiagnosisResult(
                        label=ECG_FOUNDER_LABELS[index],
                        index=index,
                        probability=probability,
                    )
                )
        results.sort(key=lambda r: r.probability, reverse=True)
        return results

    def diagnose_segment_ensemble(
        self,
        signal: np.ndarray,
        threshold: float | None = None,
        *,
        layout: str = DEFAULT_SEGMENT_LAYOUT,
        aggregation: str = DEFAULT_SEGMENT_AGGREGATION,
        apply_rate_adjustments: bool = True,
        rhythm_strip: np.ndarray | None = None,
    ) -> list[DiagnosisResult]:
        """Diagnose each printed paper column independently and aggregate.

        Feeding all 12 leads at once corrupts cross-lead features because paper
        leads are recorded in different time windows and then tiled. Running each
        column (whose leads ARE simultaneous) through the model separately and
        averaging the probability vectors recovers most of that lost accuracy
        (PTB-XL round-trip: macro AUROC ~0.75 tiled -> ~0.87 segment-ensemble).

        Args:
            signal: Z-score normalized array with shape (12, 5000).
            threshold: Sigmoid probability cutoff. Uses instance default if None.
            layout: Paper layout substring ("3x4" or "6x2").
            aggregation: How to combine per-column probabilities ("mean" or "max").
            apply_rate_adjustments: Apply heart-rate consistency heuristics on the
                aggregated vector.
            rhythm_strip: Optional full-duration rhythm strip (uncropped) used
                for heart-rate estimation. When provided, HR comes from the true
                10 s RR sequence instead of the tiled diagnosis signal.

        Returns:
            Diagnosis results above threshold, sorted by probability descending.
        """
        effective_threshold = threshold if threshold is not None else self.threshold

        column_signals = build_paper_column_signals(signal, layout)
        column_probabilities = [
            self._forward_probabilities(column_signal)
            for column_signal in column_signals
        ]
        probabilities = aggregate_probability_vectors(column_probabilities, aggregation)

        if apply_rate_adjustments:
            probabilities = self._apply_rate_consistency_adjustments(
                probabilities, signal, rhythm_strip=rhythm_strip
            )
        else:
            self.last_estimated_hr_bpm = None

        return self._collect_results(probabilities, effective_threshold)

    def diagnose_all_segment_ensemble(
        self,
        signal: np.ndarray,
        *,
        layout: str = DEFAULT_SEGMENT_LAYOUT,
        aggregation: str = DEFAULT_SEGMENT_AGGREGATION,
        apply_rate_adjustments: bool = True,
        rhythm_strip: np.ndarray | None = None,
    ) -> list[DiagnosisResult]:
        """Segment-ensemble variant of diagnose_all (no threshold filtering)."""
        return self.diagnose_segment_ensemble(
            signal,
            threshold=0.0,
            layout=layout,
            aggregation=aggregation,
            apply_rate_adjustments=apply_rate_adjustments,
            rhythm_strip=rhythm_strip,
        )

    def diagnose_all(
        self,
        signal: np.ndarray,
        *,
        apply_rate_adjustments: bool = True,
        rhythm_strip: np.ndarray | None = None,
    ) -> list[DiagnosisResult]:
        """Return all 150 diagnoses sorted by probability (no threshold filtering).

        Args:
            signal: Z-score normalized array with shape (12, 5000).
            rhythm_strip: Optional full-duration rhythm strip for HR estimation.

        Returns:
            All 150 diagnosis results sorted by probability descending.
        """
        return self.diagnose(
            signal,
            threshold=0.0,
            apply_rate_adjustments=apply_rate_adjustments,
            rhythm_strip=rhythm_strip,
        )

    def _apply_rate_consistency_adjustments(
        self,
        probabilities: np.ndarray,
        signal: np.ndarray,
        rhythm_strip: np.ndarray | None = None,
    ) -> np.ndarray:
        """Down-weight diagnoses that contradict an obvious heart-rate regime.

        Heart rate is read from the full-duration rhythm strip when available
        (the genuine 10 s RR sequence); otherwise it falls back to the tiled
        diagnosis signal, whose repeated ~2.5 s segment is far less reliable.
        """
        adjusted = probabilities.copy()
        estimated_hr: float | None = None
        if rhythm_strip is not None:
            estimated_hr = estimate_rhythm_hr(rhythm_strip)
        if estimated_hr is None:
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


def _match_paper_layout(layout_name: str) -> list[list[int]] | None:
    """Return the column lead-map whose key is a substring of layout_name."""
    for name, lead_rows in PAPER_COLUMN_LAYOUTS.items():
        if name in layout_name:
            return lead_rows
    return None


def build_paper_column_signals(
    signal: np.ndarray,
    layout_name: str,
) -> list[np.ndarray]:
    """Build one sparse 12-lead model input for each printed paper column.

    Each returned array keeps only the leads belonging to a single column
    (the rest zeroed), so the model sees a temporally coherent lead group.
    """
    matched_layout = _match_paper_layout(layout_name)
    if matched_layout is None:
        raise ValueError(f"Unsupported paper layout: {layout_name}")
    if signal.ndim != 2 or signal.shape[0] != 12:
        raise ValueError("signal must have shape (12, samples)")

    column_signals: list[np.ndarray] = []
    for column_index in range(len(matched_layout[0])):
        column_signal = np.zeros_like(signal)
        lead_indices = [row[column_index] for row in matched_layout]
        column_signal[lead_indices] = signal[lead_indices]
        column_signals.append(column_signal)
    return column_signals


def aggregate_probability_vectors(
    probability_vectors: list[np.ndarray],
    method: str,
) -> np.ndarray:
    """Aggregate independent paper-column diagnosis probabilities."""
    if not probability_vectors:
        raise ValueError("At least one probability vector is required")

    stacked = np.stack(probability_vectors)
    if method == "mean":
        return cast(np.ndarray, np.mean(stacked, axis=0))
    if method == "max":
        return cast(np.ndarray, np.max(stacked, axis=0))
    raise ValueError(f"Unsupported probability aggregation: {method}")


def _align_leads_for_model(signal: np.ndarray) -> np.ndarray:
    """Shift each lead's active data to start at sample 0 (zero-padded).

    Paper ECG layouts place leads at different time offsets in the
    canonical array (e.g. Lead I at [0:1250], V1 at [2500:3750] in
    3x4+1R). ECGFounder needs all leads aligned to compute cross-lead
    features like axis deviation and bundle branch blocks.

    Only shifts — no tiling/repeating. The zero-padded tail is
    preferable to tiling because tile boundaries create artificial
    discontinuities that ECGFounder misinterprets as arrhythmia.
    """
    total_len = signal.shape[1]
    aligned = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        lead = signal[i]
        abs_lead = np.abs(lead)
        peak_val = np.max(abs_lead)
        if peak_val < 1e-8:
            continue
        threshold = peak_val * 0.01
        active_indices = np.where(abs_lead > threshold)[0]
        if len(active_indices) == 0:
            aligned[i] = lead
            continue
        start = active_indices[0]
        end = active_indices[-1] + 1
        segment = lead[start:end]
        aligned[i, :min(len(segment), total_len)] = segment[:total_len]
    return aligned
