# Consistency training: force the same OMI probability on a clean recording and
# on the signal recovered from its photograph.

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.omi.model import OmiClassifier

# Phase B measured what goes wrong: the score distributions compress rather than
# shift, so the positives and negatives move toward each other. A cutoff cannot
# undo that; only teaching the model that both views of a recording carry the
# same answer can.
DEFAULT_CONSISTENCY_WEIGHT = 1.0
DEFAULT_CLEAN_WEIGHT = 0.5


@dataclass
class ConsistencyConfig:
    """Hyperparameters for paired clean/digitised training."""

    epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    seed: int = 20260802
    patience: int = 4
    # How hard to pull the two views together, in logit space.
    consistency_weight: float = DEFAULT_CONSISTENCY_WEIGHT
    # The clean view keeps its own supervision so the model does not drift into
    # a digitised-only specialist that forgets the signal it started good at.
    clean_weight: float = DEFAULT_CLEAN_WEIGHT
    # Rows drawn from an unlabelled corpus and appended to every labelled batch.
    # A fixed ratio keeps the supervised terms present in each step; pooling
    # both corpora in one loader would let 20K unlabelled rows swamp 460
    # labelled ones and leave most batches with no supervision at all.
    unlabelled_batch_size: int = 16


@dataclass
class ConsistencyEpoch:
    """One epoch's losses and validation score."""

    epoch: int
    total_loss: float
    digitised_loss: float
    clean_loss: float
    consistency_loss: float
    validation_auprc: float
    validation_auroc: float


class PairedSignalDataset(Dataset):
    """Serves (clean, digitised, label, teacher_logit) for one recording.

    The teacher logit is what the untouched clean-signal model said about this
    recording. Without it the penalty has a degenerate solution — a model that
    outputs the same logit for everything satisfies consistency perfectly while
    discriminating nothing. A fixed target removes that escape route.

    A label may be NaN, marking a recording with no OMI ground truth. Such rows
    still carry a teacher logit and so still train photograph-invariance; only
    the supervised terms skip them.
    """

    def __init__(
        self,
        clean: np.ndarray,
        digitised: np.ndarray,
        labels: np.ndarray,
        teacher_logits: np.ndarray | None = None,
    ) -> None:
        if not len(clean) == len(digitised) == len(labels):
            raise ValueError("clean, digitised and labels must be the same length")
        if teacher_logits is not None and len(teacher_logits) != len(labels):
            raise ValueError("teacher logits must be the same length as labels")
        self.clean = clean
        self.digitised = digitised
        self.labels = labels
        self.teacher_logits = teacher_logits

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Both caches are float16 on disk; the model runs in float32. Labels and
        # teacher logits may be scalars (one OMI target) or vectors (the 150
        # ECGFounder targets); an absent teacher is a scalar NaN either way.
        teacher = (
            torch.as_tensor(np.asarray(self.teacher_logits[index], dtype=np.float32))
            if self.teacher_logits is not None
            else torch.tensor(float("nan"))
        )
        return (
            torch.from_numpy(np.asarray(self.clean[index], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.digitised[index], dtype=np.float32)),
            torch.as_tensor(np.asarray(self.labels[index], dtype=np.float32)),
            teacher,
        )


def consistency_penalty(
    target_logits: torch.Tensor, digitised_logits: torch.Tensor
) -> torch.Tensor:
    """Squared distance from the digitised view to its target.

    Logit space rather than probability space on purpose: near 0 and 1 the
    probability scale saturates, and those are exactly the confident calls whose
    collapse the pilot measured.
    """
    return torch.mean((target_logits - digitised_logits) ** 2)


def consistency_target(teacher: torch.Tensor, clean_logits: torch.Tensor) -> torch.Tensor:
    """Per-row target for the penalty: the teacher logit where one exists.

    Rows without a precomputed teacher fall back to the live clean logit,
    detached so the penalty cannot be paid by dragging the clean side down to
    meet the digitised one. Row-wise rather than batch-wise so a mixed batch
    never loses its teachers because one row arrived without one.
    """
    if teacher.dim() < clean_logits.dim():
        # An absent teacher is served as a scalar NaN per row; broadcast it
        # across a vector-valued head so the fallback applies to every output.
        teacher = teacher.reshape(-1, *([1] * (clean_logits.dim() - 1))).expand_as(
            clean_logits
        )
    return torch.where(torch.isnan(teacher), clean_logits.detach(), teacher)


def _binary_scorer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """(AUPRC, AUROC) for a one-logit head."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    return (
        float(average_precision_score(labels, scores)),
        float(roc_auc_score(labels, scores)),
    )


def _endless_batches(dataset: Dataset, batch_size: int) -> Iterator[list[torch.Tensor]]:
    """Reshuffle and start over whenever the loader runs dry."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    while True:
        yield from loader


@torch.no_grad()
def predict_logits(
    model: OmiClassifier,
    signals: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Raw logits for every row — the teacher targets are taken from here."""
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(signals), batch_size):
        batch = torch.tensor(
            np.asarray(signals[start : start + batch_size], dtype=np.float32),
            device=device,
        )
        outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([])


@torch.no_grad()
def predict_scores(
    model: OmiClassifier,
    signals: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """OMI probability for every row of a signal array."""
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(signals), batch_size):
        batch = torch.tensor(
            np.asarray(signals[start : start + batch_size], dtype=np.float32),
            device=device,
        )
        outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([])


def train_consistency(
    model: OmiClassifier,
    dataset: PairedSignalDataset,
    validation_signals: np.ndarray,
    validation_labels: np.ndarray,
    device: torch.device,
    config: ConsistencyConfig,
    positive_weight: float | torch.Tensor = 1.0,
    unlabelled: PairedSignalDataset | None = None,
    validation_scorer: Callable[[np.ndarray, np.ndarray], tuple[float, float]] | None = None,
) -> tuple[list[ConsistencyEpoch], dict]:
    """Train the head on paired views, selecting on digitised validation AUPRC.

    The backbone stays frozen: the Gate 2 ablation showed unfreezing it buys
    +0.003 AUPRC, which is inside this pipeline's own run-to-run noise.

    Args:
        model: Classifier initialised from the clean-signal checkpoint.
        dataset: Paired clean/digitised training recordings.
        validation_signals: Digitised validation signals, held-out patients.
        validation_labels: Their OMI labels.
        device: Where to run.
        config: Hyperparameters.
        positive_weight: BCE positive-class weight for an unbalanced corpus —
            a scalar for a one-logit head, or one weight per output for a
            multi-label head.
        unlabelled: Optional pairs with NaN labels. Each labelled batch gets
            `config.unlabelled_batch_size` of these appended; they train
            photograph-invariance only. An epoch is still one pass over the
            labelled set, and the unlabelled loader cycles underneath it.
        validation_scorer: (labels, scores) -> (auprc, auroc). Defaults to the
            binary sklearn pair; a multi-label head passes its own macro scorer.

    Returns:
        (per-epoch history, best-epoch summary). The model is left holding the
        best epoch's weights.
    """
    torch.manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    unlabelled_batches = (
        _endless_batches(unlabelled, config.unlabelled_batch_size)
        if unlabelled is not None and len(unlabelled) > 0
        else None
    )
    optimiser = torch.optim.AdamW(
        model.head_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    pos_weight = torch.as_tensor(positive_weight, dtype=torch.float32, device=device)

    def supervised_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # A NaN label marks a missing target: a whole unlabelled row, or one
        # output a multi-label corpus has no positives for. Masking element-
        # wise covers both, and for a fully labelled one-logit head this is
        # exactly BCEWithLogitsLoss(pos_weight=...) averaged over the batch.
        mask = ~torch.isnan(labels)
        if not mask.any():
            return torch.zeros((), device=device)
        elementwise = nn.functional.binary_cross_entropy_with_logits(
            logits, torch.nan_to_num(labels), pos_weight=pos_weight, reduction="none"
        )
        return elementwise[mask].mean()

    if validation_scorer is None:
        validation_scorer = _binary_scorer

    history: list[ConsistencyEpoch] = []
    best = {"auprc": -1.0, "epoch": 0, "auroc": 0.0}
    best_state: dict | None = None
    epochs_without_gain = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"total": 0.0, "digitised": 0.0, "clean": 0.0, "consistency": 0.0}
        batches = 0

        for clean, digitised, labels, teacher in loader:
            if unlabelled_batches is not None:
                extra = next(unlabelled_batches)
                clean = torch.cat([clean, extra[0]])
                digitised = torch.cat([digitised, extra[1]])
                labels = torch.cat([labels, extra[2]])
                teacher = torch.cat([teacher, extra[3]])
            clean = clean.to(device)
            digitised = digitised.to(device)
            labels = labels.to(device)
            teacher = teacher.to(device)

            clean_logits = model(clean)
            digitised_logits = model(digitised)

            # A NaN label marks an unlabelled pair. Those rows carry only the
            # consistency term — the pairing itself is the supervision — which
            # is what lets an open dataset with no OMI labels scale this.
            digitised_loss = supervised_loss(digitised_logits, labels)
            clean_loss = supervised_loss(clean_logits, labels)
            penalty = consistency_penalty(
                consistency_target(teacher, clean_logits), digitised_logits
            )
            loss = (
                digitised_loss
                + config.clean_weight * clean_loss
                + config.consistency_weight * penalty
            )

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            totals["total"] += loss.item()
            totals["digitised"] += digitised_loss.item()
            totals["clean"] += clean_loss.item()
            totals["consistency"] += penalty.item()
            batches += 1

        scores = predict_scores(model, validation_signals, device)
        auprc, auroc = validation_scorer(validation_labels, scores)
        history.append(
            ConsistencyEpoch(
                epoch=epoch,
                total_loss=totals["total"] / max(batches, 1),
                digitised_loss=totals["digitised"] / max(batches, 1),
                clean_loss=totals["clean"] / max(batches, 1),
                consistency_loss=totals["consistency"] / max(batches, 1),
                validation_auprc=auprc,
                validation_auroc=auroc,
            )
        )

        if auprc > best["auprc"]:
            best = {"auprc": auprc, "epoch": epoch, "auroc": auroc}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best
