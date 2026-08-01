# Two-stage fine-tuning loop for the OMI classifier: head first, then late stages.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.omi.model import OmiClassifier


@dataclass
class TrainingConfig:
    """Hyperparameters for the two-stage schedule."""

    head_epochs: int = 8
    finetune_epochs: int = 6
    unfrozen_stages: int = 2
    head_learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    batch_size: int = 32
    seed: int = 20260801
    # Stop when validation AUPRC has not improved for this many epochs.
    patience: int = 4


@dataclass
class EpochRecord:
    """One epoch's training loss and validation score."""

    stage: str
    epoch: int
    train_loss: float
    validation_auprc: float


@dataclass
class TrainingHistory:
    """Everything the run produced, for the experiment write-up."""

    epochs: list[EpochRecord] = field(default_factory=list)
    best_auprc: float = 0.0
    best_stage: str = ""
    best_epoch: int = 0


class SignalDataset(Dataset):
    """Serves cached signals and OMI labels by row index."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray, rows: np.ndarray) -> None:
        self.signals = signals
        self.labels = labels
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        # The cache is float16 on disk; the model runs in float32.
        signal = torch.from_numpy(np.asarray(self.signals[row], dtype=np.float32))
        return signal, torch.tensor(float(self.labels[row]))


def positive_weight(labels: np.ndarray) -> float:
    """Weight for BCE that offsets the ~6% positive rate.

    Without it the loss is minimised by predicting "no OMI" everywhere.
    """
    positives = float(labels.sum())
    if positives == 0:
        return 1.0
    return float((len(labels) - positives) / positives)


@torch.no_grad()
def predict(
    model: OmiClassifier, loader: DataLoader, device: torch.device
) -> np.ndarray:
    """Return OMI probabilities for every row the loader serves, in order."""
    model.eval()
    scores: list[np.ndarray] = []
    for signals, _ in loader:
        logits = model(signals.to(device))
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(scores)


def _run_epoch(
    model: OmiClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    description: str,
) -> float:
    """Train for one epoch and return the mean loss."""
    model.train()
    total_loss = 0.0
    batches = 0
    for signals, labels in tqdm(loader, desc=description, leave=False):
        optimizer.zero_grad()
        logits = model(signals.to(device))
        loss = criterion(logits, labels.to(device))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)


def train_omi_classifier(
    model: OmiClassifier,
    signals: np.ndarray,
    labels: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    device: torch.device,
    config: TrainingConfig,
) -> tuple[dict, TrainingHistory]:
    """Run the two-stage schedule and return the best weights and history.

    Stage 1 trains the head on a frozen backbone. Stage 2 reopens the last few
    stages at a much lower rate. Selection is on validation AUPRC, which unlike
    AUROC does not flatter a model on this 6% prevalence.

    Returns:
        (best_state_dict, history)
    """
    torch.manual_seed(config.seed)

    train_loader = DataLoader(
        SignalDataset(signals, labels, train_rows),
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        SignalDataset(signals, labels, validation_rows),
        batch_size=config.batch_size,
        shuffle=False,
    )
    validation_labels = labels[validation_rows]

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight(labels[train_rows]), device=device)
    )
    history = TrainingHistory()
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    epochs_without_improvement = 0

    def evaluate_and_record(stage: str, epoch: int, train_loss: float) -> bool:
        """Score the epoch, keep the best weights, and report whether to continue."""
        nonlocal best_state, epochs_without_improvement
        scores = predict(model, validation_loader, device)
        auprc = float(average_precision_score(validation_labels, scores))
        history.epochs.append(EpochRecord(stage, epoch, train_loss, auprc))
        print(f"  [{stage}] epoch {epoch}: loss {train_loss:.4f}  val AUPRC {auprc:.4f}")

        if auprc > history.best_auprc:
            history.best_auprc = auprc
            history.best_stage = stage
            history.best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            return True
        epochs_without_improvement += 1
        return epochs_without_improvement < config.patience

    print(f"\nStage 1 — head only ({model.trainable_parameter_count():,} trainable)")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.head_learning_rate,
        weight_decay=config.weight_decay,
    )
    for epoch in range(1, config.head_epochs + 1):
        loss = _run_epoch(model, train_loader, optimizer, criterion, device, f"head e{epoch}")
        if not evaluate_and_record("head", epoch, loss):
            print("  early stop")
            break

    unfrozen = model.unfreeze_last_stages(config.unfrozen_stages)
    print(
        f"\nStage 2 — unfroze {', '.join(unfrozen)} "
        f"({model.trainable_parameter_count():,} trainable)"
    )
    # The head keeps its higher rate; reopened backbone stages move slowly so
    # 14k records cannot wash out what 10M ECGs taught the encoder.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": config.head_learning_rate},
            {
                "params": [
                    p for stage in list(model.backbone.stage_list)[-config.unfrozen_stages :]
                    for p in stage.parameters()
                ],
                "lr": config.backbone_learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )
    epochs_without_improvement = 0
    for epoch in range(1, config.finetune_epochs + 1):
        loss = _run_epoch(model, train_loader, optimizer, criterion, device, f"ft e{epoch}")
        if not evaluate_and_record("finetune", epoch, loss):
            print("  early stop")
            break

    print(
        f"\nBest: {history.best_stage} epoch {history.best_epoch} "
        f"— val AUPRC {history.best_auprc:.4f}"
    )
    return best_state, history
