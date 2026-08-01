# ECGFounder backbone with a binary OMI head, and staged unfreezing.

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.models.net1d import Net1D
from src.pipeline.diagnose import MODEL_CONFIG

FEATURE_DIM = 1024  # filter_list[-1] of the ECGFounder configuration


class OmiClassifier(nn.Module):
    """ECGFounder backbone plus a single-logit occlusion head.

    The 150-class projection is discarded; everything before it is reused.
    """

    def __init__(self, backbone: Net1D, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(FEATURE_DIM, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one OMI logit per recording."""
        features = self.backbone.forward_features(x)
        return self.head(self.dropout(features)).squeeze(-1)

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
    checkpoint_path: Path, device: torch.device, dropout: float = 0.2
) -> OmiClassifier:
    """Build an OmiClassifier on pretrained ECGFounder weights, head frozen-ready."""
    backbone = load_pretrained_backbone(checkpoint_path, device)
    classifier = OmiClassifier(backbone, dropout=dropout).to(device)
    classifier.freeze_backbone()
    return classifier
