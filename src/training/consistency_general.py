# Consistency training for ECGFounder's own 150-label head on clean/digitised
# PTB-XL pairs. The OMI experiments showed the recipe works for one target; the
# general head is what the web app serves, and PTB-XL ships labels for it, so the
# same pairs can repair "diagnosis degrades after digitisation" where it is felt.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from src.calibration.ptbxl_data import load_official_labels
from src.models.net1d import Net1D
from src.omi.consistency import predict_scores
from src.omi.model import load_pretrained_backbone
from src.pipeline.diagnose import aggregate_probability_vectors, build_paper_column_signals
from src.utils.ecg_labels import ECG_FOUNDER_LABELS
from src.utils.wfdb_helpers import read_ecg_signal

# Labels a clinician would look at first; reported one by one next to the macro.
HEADLINE_LABELS: tuple[str, ...] = (
    "NORMAL ECG",
    "ABNORMAL ECG",
    "ATRIAL FIBRILLATION",
    "SINUS TACHYCARDIA",
    "SINUS BRADYCARDIA",
    "PREMATURE VENTRICULAR COMPLEXES",
    "LEFT BUNDLE BRANCH BLOCK",
    "RIGHT BUNDLE BRANCH BLOCK",
    "LEFT VENTRICULAR HYPERTROPHY",
    "INFERIOR INFARCT",
    "ANTERIOR INFARCT",
    "NONSPECIFIC T WAVE ABNORMALITY",
)
# A class with a handful of positives cannot be learned or scored honestly.
MAX_POSITIVE_WEIGHT = 50.0


class GeneralClassifier(nn.Module):
    """ECGFounder with its pretrained 150-class projection as the trainable head.

    Unlike OmiClassifier nothing is discarded: the head starts from the
    weights that already score 0.87 macro AUROC on clean PTB-XL, and training
    only has to teach it that a digitised recording is the same recording.
    """

    def __init__(self, backbone: Net1D) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 150 logits."""
        return self.backbone(x)

    def head_parameters(self) -> list[nn.Parameter]:
        """The final projection only."""
        return list(self.backbone.dense.parameters())

    def freeze_backbone(self) -> None:
        """Everything before the projection stays as ECGFounder shipped it."""
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = name.startswith("dense.")


def build_general_classifier(checkpoint_path: Path, device: torch.device) -> GeneralClassifier:
    """Load ECGFounder and freeze all but its projection."""
    classifier = GeneralClassifier(load_pretrained_backbone(checkpoint_path, device)).to(device)
    classifier.freeze_backbone()
    return classifier


@dataclass
class PtbxlPairs:
    """Clean and digitised views of PTB-XL recordings with the official targets."""

    ecg_ids: np.ndarray
    clean: np.ndarray
    digitised: np.ndarray
    targets: np.ndarray
    strat_fold: np.ndarray
    layout_names: np.ndarray

    def __len__(self) -> int:
        return len(self.ecg_ids)

    def subset(self, mask: np.ndarray) -> PtbxlPairs:
        """Rows where `mask` is true."""
        return PtbxlPairs(
            self.ecg_ids[mask],
            self.clean[mask],
            self.digitised[mask],
            self.targets[mask],
            self.strat_fold[mask],
            self.layout_names[mask],
        )


def load_ptbxl_pairs(
    manifest_path: Path, data_dir: Path, labels_path: Path, root: Path
) -> PtbxlPairs:
    """Load every row of a PTB-XL consistency corpus with its 150-label target.

    Args:
        manifest_path: From build_ptbxl_consistency_corpus.py.
        data_dir: PTB-XL root holding ptbxl_database.csv and records500/.
        labels_path: ECGFounder's official ptbxl label CSV.
        root: What relative manifest paths are relative to (the project root).
    """
    manifest = pd.read_csv(manifest_path)
    official = load_official_labels(labels_path)
    database = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")

    # ECGFounder's authors did not keep every PTB-XL record in their label
    # file; a corpus row without a target is unusable here, and how many were
    # dropped is worth printing so the training set size is not a surprise.
    keys = database.loc[manifest.ecg_id, "filename_hr"].astype(str).to_numpy()
    labelled = np.array([key in official for key in keys])
    if not labelled.all():
        print(f"Skipping {int((~labelled).sum())} corpus records absent from {labels_path.name}")
    manifest = manifest[labelled].reset_index(drop=True)
    keys = keys[labelled]

    clean: list[np.ndarray] = []
    digitised: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for row, key in zip(manifest.itertuples(index=False), keys, strict=True):
        record_path = Path(row.record_path)
        signal_path = Path(row.signal_path)
        if not record_path.is_absolute():
            record_path = root / record_path
        if not signal_path.is_absolute():
            signal_path = root / signal_path
        targets.append(official[key].astype(np.float32))
        clean.append(read_ecg_signal(str(record_path))[0].astype(np.float16))
        digitised.append(np.load(signal_path).astype(np.float16))

    return PtbxlPairs(
        ecg_ids=manifest.ecg_id.to_numpy(),
        clean=np.stack(clean),
        digitised=np.stack(digitised),
        targets=np.stack(targets),
        strat_fold=database.loc[manifest.ecg_id, "strat_fold"].to_numpy(),
        layout_names=manifest.layout_name.to_numpy(),
    )


def supported_labels(targets: np.ndarray, min_positives: int) -> np.ndarray:
    """Boolean mask over the 150 outputs with at least `min_positives` positives."""
    return targets.sum(axis=0) >= min_positives


def mask_unsupported(targets: np.ndarray, support: np.ndarray) -> np.ndarray:
    """NaN out the outputs that cannot be trained, so the loss skips them."""
    masked = targets.astype(np.float32).copy()
    masked[:, ~support] = np.nan
    return masked


def positive_weights(targets: np.ndarray, support: np.ndarray) -> torch.Tensor:
    """Per-output negative/positive ratio, capped; 1.0 where unsupported."""
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    weights = np.ones(targets.shape[1], dtype=np.float32)
    weights[support] = np.minimum(negatives[support] / positives[support], MAX_POSITIVE_WEIGHT)
    return torch.from_numpy(weights)


def macro_scorer(support: np.ndarray) -> Callable[[np.ndarray, np.ndarray], tuple[float, float]]:
    """(macro AUPRC, macro AUROC) over supported outputs that have both classes."""

    def score(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
        auprcs: list[float] = []
        aurocs: list[float] = []
        for column in np.flatnonzero(support):
            truth = labels[:, column]
            if truth.sum() == 0 or truth.sum() == len(truth):
                continue
            auprcs.append(float(average_precision_score(truth, scores[:, column])))
            aurocs.append(float(roc_auc_score(truth, scores[:, column])))
        return float(np.mean(auprcs)), float(np.mean(aurocs))

    return score


def per_label_auroc(
    labels: np.ndarray, scores: np.ndarray, support: np.ndarray
) -> dict[str, float]:
    """AUROC for every supported output, keyed by ECGFounder label name."""
    out: dict[str, float] = {}
    for column in np.flatnonzero(support):
        truth = labels[:, column]
        if 0 < truth.sum() < len(truth):
            out[ECG_FOUNDER_LABELS[column]] = float(roc_auc_score(truth, scores[:, column]))
    return out


def macro_separation(labels: np.ndarray, scores: np.ndarray, support: np.ndarray) -> float:
    """Median positive score minus median negative score, averaged over supported outputs."""
    gaps = []
    for column in np.flatnonzero(support):
        truth = labels[:, column].astype(bool)
        if 0 < truth.sum() < len(truth):
            gaps.append(np.median(scores[truth, column]) - np.median(scores[~truth, column]))
    return float(np.mean(gaps))


@torch.no_grad()
def segment_ensemble_scores(
    model: nn.Module,
    signals: np.ndarray,
    layout_names: np.ndarray,
    device: torch.device,
    aggregation: str = "mean",
) -> np.ndarray:
    """The web app's path: score each paper column alone and average.

    Mirrors ECGDiagnoser.diagnose_segment_ensemble minus calibration and the
    rate heuristics, which sit after the model and are not what is trained.
    """
    out = []
    for signal, layout_name in zip(signals, layout_names, strict=True):
        columns = np.stack(
            build_paper_column_signals(np.asarray(signal, dtype=np.float32), str(layout_name))
        )
        column_scores = list(predict_scores(model, columns, device))
        out.append(aggregate_probability_vectors(column_scores, aggregation))
    return np.stack(out)
