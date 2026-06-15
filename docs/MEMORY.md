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
- Task 3.2 is complete. `configs/quality_feature_contract_v1.yaml` documents the
  six active inference-time features with type, range, unit, and reject-on-missing
  policy. The gate now consumes typed validated features.
- Blur, glare, shadow, occlusion, calibration-pulse confidence, reconstruction
  disagreement, and diagnosis disagreement are explicitly unavailable and are
  not consumed by the current gate.
- Task 4.1 is complete. Runtime decisions live in `src/quality/gate.py`, typed
  decision models and stable audit reason codes live in `src/quality/models.py`,
  and runtime threshold loading is isolated in `src/quality/thresholds.py`.
- Runtime quality modules do not import training or matched-reference fidelity
  target code. Benchmark reports include both stable reason codes and readable
  reason messages.
- Task 4.2, enforcing abstention in Gradio before diagnosis, remains gated on an
  independent matched holdout review and must not be wired yet.
- Task 5.1 is complete. `src/evaluation/digitization_regression.py` compares
  paired candidate artifacts against the frozen Phase 2 baseline and refuses
  mismatched record sets or source-image hashes.
- `configs/digitization_regression_v1.yaml` locks supported-category promotion
  tolerances for scans and Doogee/iPhone/Samsung phone photos. Bent and crumpled
  are target categories; screens are reported but do not determine promotion.
- Regression reports include correlation, RMSE, SNR, gain error, runtime,
  failure rate, and quality-gate outcome changes. Missing required supported
  categories or unavailable required metrics fail promotion.
- Run candidate review with
  `python scripts/evaluate_digitization_experiment.py --candidate <report.json>
  --experiment-name <name> --output-dir <dir>`. A failed promotion writes its
  audit artifacts and exits nonzero.
- Task 5.2 experiment 1 is complete and rejected. `conservative-perspective-v1`
  selected only three crumpled images and did not regress supported-category
  tolerances, but two corrected records regressed substantially. Crumpled
  median correlation changed only `+0.0011`; RMSE and SNR worsened.
- Do not enable `enable_perspective_correction` in the default pipeline or UI.
  The experimental mode remains available only for controlled evaluation.
- Task 5.2 experiment 2 is complete and rejected by the locked promotion gate.
  `conservative-shadow-v1` improved bent correlation by `+0.1749` and crumpled
  correlation by `+0.0960`, but Doogee median wall runtime increased `33.2%`
  against a locked `25%` tolerance.
- Do not enable `enable_shadow_normalization` in the default pipeline or UI.
  It remains experimental because target fidelity improved substantially.
- Task 5.2 experiment 3 is complete and rejected after a 14-image pilot.
  `layout-segments-v1` constrained `3x4+1R` pages to `3x4` so the rhythm strip
  could not overwrite Lead II, but produced no target aggregate benefit and
  worsened one crumpled Lead II correlation by `-0.169`.
- Do not use the dormant `src/pipeline/lead_assignment.py` raw-line override:
  `raw_lines` are pixel Y-coordinates while canonical lines are microvolts.
- Task 5.2 experiment 4 is rejected after an early pilot stop. The existing
  vendor dewarping retry hit the strict `60 second` timeout on crumpled image 4,
  the second pilot record. No full benchmark was run.
- Task 5.2 experiment 5 is complete. The versioned research-only contract at
  `configs/segment_aware_experiment_v1.yaml` passed on the existing 70-image
  diagnosis-drift artifact: bent mean cosine improved `+0.0336`, crumpled
  improved `+0.0145`, and supported categories had no failed checks.
- Segment-aware mean aggregation remains `experimental-only`. The result is
  diagnosis-consistency evidence, not clinical accuracy or digitization
  fidelity, and it must not replace the production diagnosis path.
- Task 5.2's controlled experiment series is complete.
- Task 6.1's validator implementation is complete. The locked contract is
  `configs/matched_photo_manifest_v1.yaml`; validation is run with
  `scripts/validate_matched_photo_dataset.py` and writes only aggregate,
  path-free evidence.
- `scripts/build_matched_photo_manifest.py` builds the local manifest from a
  strict `collection.csv`, computes SHA-256 values, preserves explicitly
  pre-registered splits, and refuses to write unless the complete dataset
  passes the locked validator.
- The validator enforces at least 20 anonymous matched ECG records, at least
  two capture sources, photo/reference SHA-256 verification, explicit PHI
  review, safe relative paths, and case-grouped pre-registered tune/test splits.
- Task 6.1 acceptance remains pending because the current 10-photo real-phone
  batch is Level A and contains no matched references. The next executable work
  is Level B data collection followed by local manifest validation.
- A separate pre-registered internal PMcardio holdout is locked in
  `results/pmcardio-holdout/pmcardio_holdout_v1.json`. It excludes the 10
  development ECGs and assigns all remaining 90 complete physical ECGs as
  `30` tune and `60` locked test groups across 630 images and six layouts.
- The PMcardio holdout is stronger engineering evidence than the 10-ECG
  development baseline, but it is not external or clinical evidence because it
  comes from the same public dataset.
- The 30-group PMcardio tune split has been evaluated; the separate 60-group
  locked test remains unopened. Tune extraction succeeded on `180/210` images
  with median correlation `0.5264`.
- The frozen v1 gate failed tune review with `7` false accepts (`5.22%`) and
  reject recall `67.91%`. It must not be used for production abstention or
  locked test evaluation.
- A tune-only layout-scope experiment limited support to `3x4+1R` and
  `3x4+3R`. It reduced false accepts to `2` and raised reject recall to
  `95.52%`, but false rejects reached `56.58%` and the false-accept upper 95%
  bound remained above `2%`. Do not promote this policy.
- The two remaining in-scope false accepts look healthy under the existing
  inference features. The next quality-gate work must add reconstruction or
  image-to-signal disagreement rather than only tightening current thresholds.
- A reference-free reconstruction disagreement metric now compares paired
  attempts using shifted morphology correlation and optional calibrated mV
  disagreement. A two-record tune pilot rejected conservative shadow
  normalization as the perturbation: both false accepts remained above
  `0.9994` median pair correlation, demonstrating stable-but-wrong extraction.
- Ten or twenty independent ECGs support pilots and pipeline validation, not
  reliable performance claims. Roughly 150 independent relevant cases with
  zero failures are needed to place a rule-of-three upper 95% bound near `2%`;
  clinical claims require larger external evidence.
- Do not wire quality-gate rejection into diagnosis or Gradio before an
  independent matched holdout review.
