# Phase 3: Digitization v2 Implementation Plan

**Date:** 2026-06-14
**Status:** Active
**Goal:** Improve ECG reconstruction fidelity without hiding failures or degrading
supported scan and phone-photo inputs.

## Entry Conditions

- Preserve the 70-image PMcardio report as the frozen Phase 2 baseline.
- Treat the current quality-gate result as development-only because thresholds and
  evaluation use the same 70 images.
- Do not connect the gate to diagnosis or the UI until an independent holdout exists.
- Do not claim clinical accuracy from digitization fidelity or diagnosis consistency.

## Phase Exit Gate

- Quality-gate thresholds are locked before evaluation on an independent matched holdout.
- Reject recall target is at least `95%`; false-accept target is below `2%`, with confidence
  intervals and category-level reporting.
- Bent and crumpled fidelity improves against the frozen baseline without regression in
  supported scan and phone-photo categories.
- Supported inputs preserve calibrated amplitude, morphology, timing, and lead identity.
- Unsupported conditions produce an explicit reject or abstention outcome.
- All benchmark artifacts record code, config, model, and dataset manifest versions.

## Workstream 1: Freeze and Version the Quality-Gate Baseline

### Task 1.1: Introduce a Versioned Gate Configuration [Completed 2026-06-14]

**Files**

- Create: `configs/quality_gate_v1.yaml`
- Modify: `src/training/quality_gate.py`
- Modify: `scripts/evaluate_quality_gate.py`
- Modify: `tests/test_quality_gate.py`

**Implementation**

- Define typed fidelity-target and inference-gate threshold models.
- Load the default gate from a versioned YAML file.
- Pass configuration explicitly into classification and evaluation functions.
- Write the config version and resolved thresholds into benchmark reports.
- Preserve current v2 benchmark behavior exactly.

**Tests**

- Default config reproduces current target and prediction outcomes.
- Boundary values behave deterministically.
- Missing or invalid config values fail with actionable errors.
- Benchmark report contains the resolved config and version.

**Acceptance**

- Existing 70-image aggregate metrics do not change.
- `pytest tests/test_quality_gate.py -q` passes.
- Ruff passes for changed Python files.

### Task 1.2: Freeze the Development Baseline [Completed 2026-06-14]

**Files**

- Create: `docs/baselines/phase3-quality-gate-v1.md`
- Modify: `scripts/evaluate_quality_gate.py`
- Modify: `tests/test_quality_gate.py`

**Implementation**

- Add source manifest and input hash validation.
- Record the frozen aggregate and category metrics.
- Mark reports as development, holdout, or external evaluation.
- Prevent accidental overwrite of a frozen report without an explicit flag.

**Acceptance**

- Baseline can be regenerated from one command.
- A source mismatch or accidental overwrite fails loudly.

## Workstream 2: Build Leakage-Safe Benchmark Splits

### Task 2.1: Add Grouped Split Manifests [Completed 2026-06-14]

**Files**

- Create: `src/evaluation/grouped_split.py`
- Create: `scripts/create_quality_gate_split.py`
- Create: `tests/test_grouped_split.py`
- Create: `configs/quality_gate_split_v1.yaml`

**Implementation**

- Group variants by `ecg_id`.
- Generate deterministic train, tune, and test manifests.
- Stratify capture categories where data permits.
- Validate that no ECG identity crosses split boundaries.

**Acceptance**

- Re-running with the same seed produces identical manifests.
- Leakage validation rejects malformed manifests.

### Task 2.2: Separate Tuning from Evaluation [Completed 2026-06-14]

**Files**

- Modify: `scripts/evaluate_quality_gate.py`
- Modify: `src/training/quality_gate.py`
- Modify: `tests/test_quality_gate.py`

**Implementation**

- Permit threshold tuning only on the tune split.
- Evaluate locked thresholds on test or external splits.
- Label reports clearly when sample size is insufficient for a meaningful holdout.

**Acceptance**

- The evaluator refuses to tune and score on the same records.

## Workstream 3: Improve Quality-Gate Evaluation

### Task 3.1: Add Statistical Reporting [Completed 2026-06-14]

**Files**

- Create: `src/evaluation/quality_gate_metrics.py`
- Create: `tests/test_quality_gate_metrics.py`
- Modify: `scripts/evaluate_quality_gate.py`

**Implementation**

- Report confusion matrix, reject recall, false-accept rate, false-reject rate, and coverage.
- Add bootstrap confidence intervals overall and by category.
- Report missed rejects as reviewable record identifiers.

**Acceptance**

- Metrics handle empty and small categories without division errors.
- Reports make unsupported statistical conclusions explicit.

### Task 3.2: Expand Inference-Time Feature Contract [Completed 2026-06-14]

Candidate features include capture defects, layout and lead-label confidence, calibration,
active-lead consistency, reconstruction disagreement, and diagnostic disagreement.

**Acceptance**

- Each feature has a documented definition, range, missing-value policy, and unit test.
- Features unavailable at inference are never used by the production gate.

## Workstream 4: Introduce a Production Quality Decision API

### Task 4.1: Move Runtime Decisions Out of Training Code [Completed 2026-06-14]

**Files**

- Create: `src/quality/models.py`
- Create: `src/quality/gate.py`
- Create: `tests/test_quality_gate_runtime.py`
- Modify: `src/training/quality_gate.py`

**Implementation**

- Define typed `accept`, `warn`, and `reject` decisions with reason codes.
- Keep matched-reference target generation under evaluation/training code.
- Expose a runtime API that consumes only inference-time diagnostics.

**Acceptance**

- Runtime modules do not import matched-reference metrics.
- Reason codes are stable and suitable for UI rendering and audit logs.

### Task 4.2: Enforce Abstention Before Diagnosis

**Files**

- Modify: `src/web/app.py`
- Create or modify: `tests/test_web_app.py`

**Implementation**

- Run the quality decision after digitization and before diagnosis.
- Never call ECGFounder for rejected inputs.
- Display retake reasons rather than diagnosis output.
- Show warnings without presenting them as successful high-confidence results.

**Acceptance**

- Integration tests prove rejected inputs never reach the diagnoser.
- UI wiring happens only after independent holdout review.

## Workstream 5: Controlled Digitization v2 Experiments

### Task 5.1: Create a Regression Harness [Completed 2026-06-15]

**Files**

- Create: `src/evaluation/digitization_regression.py`
- Create: `scripts/evaluate_digitization_experiment.py`
- Create: `tests/test_digitization_regression.py`

**Implementation**

- Compare candidate outputs with the frozen baseline by record and category.
- Report correlation, RMSE, SNR, gain, runtime, failure rate, and quality-gate outcome.
- Fail experiment promotion when supported categories regress beyond locked tolerances.

**Completed**

- Added paired record and category comparison with source-image identity validation.
- Locked v1 supported-category tolerances in
  `configs/digitization_regression_v1.yaml`.
- Added an audit-safe CLI that exits nonzero when promotion checks fail.
- Kept bent/crumpled as target categories and screen captures as reported but
  unclassified; promotion is currently determined by scans and supported phone photos.

### Task 5.2: Run Targeted Reconstruction Experiments [Completed 2026-06-15]

**Experiment order**

1. Conservative page-boundary and perspective correction. [Completed and rejected 2026-06-15]
2. Fold- and shadow-aware normalization. [Completed and rejected 2026-06-15]
3. Layout-specific trace extraction. [Completed and rejected after pilot 2026-06-15]
4. Calibrated dewarping with strict memory and timeout limits. [Rejected after pilot 2026-06-15]
5. Segment-aware reconstruction without fake 10-second tiling. [Research candidate passed 2026-06-15]

**Acceptance**

- Each experiment changes one controlled factor.
- Bent and crumpled results improve without supported-category regression.
- Memory, timeout, and failure behavior are reported.

**Experiment 1 result**

- `conservative-perspective-v1` selected only 3/70 images, all crumpled, and
  preserved supported-category regression tolerances.
- The candidate was rejected because two of three corrected records regressed
  substantially; aggregate crumpled RMSE and SNR worsened.
- Detailed result: `docs/experiments/phase3-conservative-perspective-v1.md`.

**Experiment 2 result**

- `conservative-shadow-v1` substantially improved bent and crumpled correlation
  while preserving supported-category fidelity tolerances.
- The candidate was rejected because Doogee median wall runtime increased
  `33.2%`, above the locked `25%` category tolerance.
- Detailed result: `docs/experiments/phase3-conservative-shadow-v1.md`.

**Experiment 3 result**

- `layout-segments-v1` constrained `3x4+1R` pages to short `3x4` segments so
  the rhythm strip could not overwrite Lead II.
- The 14-image pilot showed no target-category aggregate benefit and worsened
  one crumpled Lead II correlation by `-0.169`; the full benchmark was not run.
- Detailed result: `docs/experiments/phase3-layout-segments-v1.md`.

**Experiment 4 result**

- Existing vendor dewarping retry was evaluated in isolated workers with a
  strict `60 second` timeout.
- The second pilot record, crumpled image 4, timed out; the pilot was stopped
  immediately and no full benchmark was run.
- Detailed result: `docs/experiments/phase3-dewarping-retry-v1.md`.

**Experiment 5 result**

- `segment-aware-experiment-v1` applied a locked research-only contract to the
  existing 70-image matched diagnosis-drift artifact without rerunning the model.
- Segment ensemble improved mean cosine by `+0.0336` on bent and `+0.0145` on
  crumpled images, with no failed checks in supported categories.
- The research candidate passed, but production remains `experimental-only`
  because this is diagnosis-consistency evidence, not clinical accuracy or
  digitization-fidelity evidence.
- Detailed result: `docs/experiments/phase3-segment-aware-v1.md`.

## Workstream 6: Level B Matched Real-Photo Validation

### Task 6.1: Add Dataset Manifest and Validation [Validator completed; collection pending]

**Files**

- Create: `src/evaluation/matched_photo_manifest.py`
- Create: `scripts/validate_matched_photo_dataset.py`
- Create: `tests/test_matched_photo_manifest.py`
- Create: `configs/matched_photo_manifest_v1.yaml`

**Acceptance**

- At least 20 distinct anonymous ECG records with matched references.
- Capture metadata and ECG-grouped splits validate successfully.
- No source images or patient-identifiable data enter git.

**Implementation status 2026-06-15**

- Added the locked `matched-photo-manifest-v1` contract and aggregate-only
  validation CLI.
- Added a strict CSV-to-manifest builder that computes local file hashes,
  preserves explicitly pre-registered splits, rejects unexpected inventory
  columns, and writes only after full validation passes.
- Validation rejects non-anonymous IDs, declared or direct PHI fields, unsafe
  paths, missing or mismatched file hashes, inconsistent matched references,
  insufficient capture-source diversity, and ECG-group split leakage.
- The existing 10-photo real-phone batch remains Level A and has no matched
  references. Task 6.1 acceptance remains pending until at least 20 Level B
  records are collected and the local manifest passes validation.

### Task 6.2: Run Independent Phase 3 Gate Review

**Acceptance**

- Locked quality gate is evaluated on unseen matched photos.
- Digitization v2 is compared with the frozen Phase 2 baseline.
- Results determine whether Phase 3 continues, narrows supported inputs, or returns to
  reconstruction development.

**Pre-registration status 2026-06-15**

- Locked an internal PMcardio holdout before evaluation: 90 previously unused
  ECG groups, 630 physical images, and layout-balanced `30` tune / `60` test
  assignments.
- This is internal holdout evidence, not external evidence. Level B real-photo
  collection remains required for the independent external review.

**Tune review status 2026-06-15**

- Evaluated only the 30-group tune split; the 60-group test split remains
  locked.
- Extraction succeeded on `180/210` images with median waveform correlation
  `0.5264`. The frozen v1 gate produced `7` false accepts (`5.22%`) and
  `67.91%` reject recall.
- A research-only `3x4+1R` / `3x4+3R` layout scope reduced false accepts to `2`
  and raised reject recall to `95.52%`, but false rejects rose to `56.58%`.
- Task 6.2 is blocked until a revised gate adds a new reconstruction or
  image-to-signal disagreement feature and is frozen using tune-only evidence.
- A reference-free paired-reconstruction metric is implemented. Conservative
  shadow normalization was rejected as its perturbation after both remaining
  in-scope false accepts stayed above `0.9994` median pair correlation.
- Detailed result: `docs/experiments/phase3-pmcardio-holdout-tune-v1.md`.
