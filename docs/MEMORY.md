# Corio ECG Project Memory

## Stable Technical Facts

- Corio ECG is a research prototype: ECG image digitization followed by
  ECGFounder inference. It is not clinically validated.
- Local ECGFounder logits match the official implementation exactly on the
  same checkpoint and input (`max_abs_diff = 0.0`).
- Ground-truth diagnostic performance, round-trip consistency, matched
  waveform fidelity, and operational photo success are separate questions and
  must not be presented as one accuracy metric.
- Successful digitization is not evidence of signal fidelity. Einthoven
  consistency alone is also insufficient as a quality gate.
- Standard paper ECG layouts contain sequential lead segments. Tiling them into
  full-length leads is convenient but does not reconstruct a simultaneous
  10-second 12-lead recording.
- Segment-aware mean aggregation improves PMcardio matched-reference diagnosis
  consistency, but remains experimental and outside the production UI path.

## Frozen Phase 2 Baselines

- Full PTB-XL official evaluation: macro AUROC `0.8679`, macro AP `0.4454`,
  micro F1 `0.5885`, macro F1 `0.3673`.
- Balanced PMcardio set: 70/70 success, median waveform correlation `0.823`,
  median RMSE `0.108 mV`, median SNR `3.83 dB`, median gain `0.904`.
- PMcardio diagnosis drift: tiled cosine `0.8939`; segment ensemble cosine
  `0.9238`.
- Metadata-informed anonymous real-photo batch: 10/10 extraction success, but
  no matched references and therefore no accuracy claim.

## Working Rules

- Keep aggregate benchmark reports in git; keep per-image operational records,
  source datasets, model weights, and anonymous source photos out of git.
- Preserve matched-reference benchmark results as frozen baselines when testing
  pipeline changes.
- Disable unrestricted dewarping retry on the current 24 GB macOS machine; it
  can be killed for memory use or exceed the timeout.
- Use the local machine for development and focused evaluation. Prepare a
  reproducible, resume-capable job before renting a remote GPU.
- Do not fine-tune the diagnosis model until unreliable digitizations can be
  detected and a matched real-photo validation set exists.

## Current Direction

Phase 3 starts with an interpretable quality gate and abstention policy,
followed by targeted bent/crumpled reconstruction experiments and Level B
matched real-photo collection. See
`docs/plans/2026-06-13-next-priorities.md`.
