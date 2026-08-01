# ECGFounder backbone with a binary OMI head, and staged unfreezing.

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.net1d import Net1D
from src.pipeline.diagnose import MODEL_CONFIG

FEATURE_DIM = 1024  # filter_list[-1] of the ECGFounder configuration


# Auxiliary targets trained alongside OMI. They are clinically adjacent labels
# the dataset already ships, so they cost nothing to add and give the shared
# representation more to organise around.
AUXILIARY_LABELS: list[str] = ["STEMI", "NSTEMI", "AMI", "CTO"]

# Culprit territories, collapsed from the 12 per-segment columns. LM is left out
# — 5 cases in the whole training set is not a learnable target.
CULPRIT_TERRITORIES: dict[str, list[str]] = {
    "lad": ["PLAD", "MLAD", "DLAD", "DB"],
    "lcx": ["PLCX", "MLCX", "DLCX", "OM"],
    "rca": ["PRCA", "MRCA", "DRCA"],
}


class OmiClassifier(nn.Module):
    """ECGFounder backbone plus an occlusion head, optionally multi-task.

    The 150-class projection is discarded; everything before it is reused.
    """

    def __init__(
        self,
        backbone: Net1D,
        dropout: float = 0.2,
        hidden_dim: int | None = None,
        auxiliary_targets: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head: nn.Module = (
            nn.Sequential(
                nn.Linear(FEATURE_DIM, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            if hidden_dim
            else nn.Linear(FEATURE_DIM, 1)
        )
        # Auxiliary predictions branch off the shared representation directly,
        # so they shape the encoder without competing for the OMI head.
        self.auxiliary_head = (
            nn.Linear(FEATURE_DIM, auxiliary_targets) if auxiliary_targets else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one OMI logit per recording."""
        return self.forward_with_auxiliary(x)[0]

    def forward_with_auxiliary(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (omi_logit, auxiliary_logits); the second is None when off."""
        features = self.dropout(self.backbone.forward_features(x))
        omi_logit = self.head(features).squeeze(-1)
        auxiliary = self.auxiliary_head(features) if self.auxiliary_head else None
        return omi_logit, auxiliary

    def head_parameters(self) -> list[nn.Parameter]:
        """Every parameter outside the backbone."""
        parameters = list(self.head.parameters())
        if self.auxiliary_head is not None:
            parameters += list(self.auxiliary_head.parameters())
        return parameters

    def freeze_backbone(self) -> None:
        """Train the head alone — the cheap first stage."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_last_stages(self, count: int) -> list[str]:
        """Reopen the final `count` backbone stages for low-rate fine-tuning.

        Later stages carry the task-specific features; early ones encode generic
        waveform structure that 10 million ECGs taught better than 14k can.

        Args:
            count: How many trailing stages to unfreeze.

        Returns:
            Names of the unfrozen stages.
        """
        stages = list(self.backbone.stage_list)
        if not 0 < count <= len(stages):
            raise ValueError(f"count must be in 1..{len(stages)}, got {count}")

        unfrozen = []
        for offset, stage in enumerate(stages[-count:]):
            for parameter in stage.parameters():
                parameter.requires_grad = True
            unfrozen.append(f"stage_{len(stages) - count + offset}")
        return unfrozen

    def trainable_parameter_count(self) -> int:
        """How many parameters currently receive gradients."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_pretrained_backbone(checkpoint_path: Path, device: torch.device) -> Net1D:
    """Load ECGFounder weights into a Net1D backbone.

    Reuses the production MODEL_CONFIG so the architecture cannot drift apart
    from the one the inference path validates against.
    """
    model = Net1D(**MODEL_CONFIG)
    # weights_only=True cannot read this file: the upstream ECGFounder release
    # ships a pickled model object, not a plain tensor dict. Same local
    # checkpoint the inference path already loads (scripts/download_models.py).
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state_dict = checkpoint.state_dict()

    cleaned = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    return model.to(device)


def build_classifier(
    checkpoint_path: Path,
    device: torch.device,
    dropout: float = 0.2,
    hidden_dim: int | None = None,
    auxiliary_targets: int = 0,
) -> OmiClassifier:
    """Build an OmiClassifier on pretrained ECGFounder weights, head frozen-ready."""
    backbone = load_pretrained_backbone(checkpoint_path, device)
    classifier = OmiClassifier(
        backbone,
        dropout=dropout,
        hidden_dim=hidden_dim,
        auxiliary_targets=auxiliary_targets,
    ).to(device)
    classifier.freeze_backbone()
    return classifier


def build_auxiliary_matrix(table) -> tuple[np.ndarray, list[str]]:
    """Assemble the auxiliary target matrix from the label table.

    Args:
        table: Label table carrying the dataset's diagnosis and segment columns.

    Returns:
        (matrix of shape (n_records, n_targets), target names in column order).
    """
    columns: list[np.ndarray] = []
    names: list[str] = []

    for label in AUXILIARY_LABELS:
        columns.append(table[label].to_numpy())
        names.append(label)

    for territory, segments in CULPRIT_TERRITORIES.items():
        present = [segment for segment in segments if segment in table.columns]
        columns.append((table[present].sum(axis=1) > 0).astype(int).to_numpy())
        names.append(f"culprit_{territory}")

    return np.stack(columns, axis=1).astype(np.float32), names
