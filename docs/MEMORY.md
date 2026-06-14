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

The canonical phase headings are in `docs/plans/master-roadmap.md`. The active
implementation plan is
`docs/plans/2026-06-14-phase3-digitization-v2-implementation-plan.md`.

Phase 3 is Digitization v2, but it starts by hardening the quality gate because
reconstruction changes cannot be evaluated safely without reliable rejection.
After the gate baseline is frozen, work proceeds through leakage-safe splits,
statistical reporting, a runtime abstention API, controlled bent/crumpled
experiments, and Level B matched real-photo validation.

## Active Phase 3 State

- Task 1.1 is complete: quality-gate thresholds are versioned in
  `configs/quality_gate_v1.yaml` and loaded through typed validated models.
- Benchmark reports now include the resolved quality-gate config and version.
- The versioned config preserves the existing 70-image aggregate exactly:
  `0` false accepts, `9` false rejects, `4` missed rejects, reject recall
  `81.8%`.
- These results are development-only because threshold development and
  evaluation used the same 70 PMcardio images.
- Task 1.2 is complete: the development baseline is frozen in
  `docs/baselines/phase3-quality-gate-v1.md`; generated reports include source
  SHA-256 and evaluation-stage metadata, validate an expected source hash, and
  refuse accidental overwrite.
- Tasks 2.1 and 2.2 are complete. The deterministic manifest at
  `results/quality-gate/quality_gate_split_v1.json` keeps all seven variants of
  each ECG together and assigns 10 ECG groups as train/tune/test = `6/2/2`.
- The split is explicitly `retroactive-development-only` because thresholds
  were developed before it existed. The evaluator refuses to present its test
  partition as locked holdout evidence.
- Threshold-tuning runs may access only the tune split. Locked internal
  evaluation requires a future pre-registered test split; locked external
  evaluation must not use the internal split manifest.
- Task 3.1 is complete. Quality-gate reports include a full confusion matrix,
  non-reject coverage, reviewable missed-reject identifiers, and deterministic
  bootstrap confidence intervals resampled by `ecg_id`.
- The development baseline contains 70 images but only 10 independent ECG
  groups. Reports automatically warn that its rates and intervals are unstable
  and are not external performance evidence.
- The four missed rejects are warnings rather than accepts: crumpled image 43,
  screen image 92, and iPhone/scan variants of image 14.
- The next task is Task 3.2: define and test the expanded inference-time quality
  feature contract before creating the runtime quality decision API.
- Do not wire quality-gate rejection into diagnosis or Gradio before an
  independent matched holdout review.
