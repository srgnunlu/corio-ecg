# Corio ECG Next Priorities

**Date:** 2026-06-13
**Current phase:** Phase 3 reliability and external validation

## Current Position

Phase 2 established a reproducible evaluation baseline:

- Local ECGFounder inference exactly matches the official implementation.
- The balanced 70-image PMcardio benchmark measures matched waveform fidelity,
  calibrated amplitude error, and diagnosis drift.
- Segment-aware diagnosis evaluation improves mean matched-reference
  consistency over tiled reconstruction.
- The anonymous real-photo batch verifies operational behavior, but it has no
  matched signal reference and cannot establish accuracy.
- Evaluation code, aggregate results, and methodology are committed and pushed
  to `feature/test-ui-gradio`.

The project can digitize ECG images and run a diagnostic foundation model. It
is not clinically validated, and successful extraction must not be interpreted
as a reliable diagnosis.

## Recommended Execution Order

### Priority 1: Quality Gate and Abstention

Build a quality score that can reject or strongly warn on unreliable
digitizations before diagnosis is shown.

Use PMcardio matched-reference fidelity as the target and combine independent
signals such as layout confidence, active-lead count, Einthoven consistency,
calibration confidence, and extraction warnings.

Acceptance criteria:

- Define explicit `accept`, `warn`, and `reject` outcomes.
- Evaluate the gate against matched-reference fidelity thresholds.
- Report false-accept rate, false-reject rate, and category-level behavior.
- The UI must not present rejected inputs as successful diagnoses.

### Priority 2: Bent and Crumpled Paper Reconstruction

Run controlled experiments targeting the weakest PMcardio categories instead
of changing the whole pipeline without measurement.

Candidate experiments:

- conservative page boundary detection and perspective correction;
- fold/shadow-aware contrast normalization;
- layout-specific trace extraction;
- calibrated dewarping with strict memory and timeout limits.

Acceptance criteria:

- Keep the current 70-image report as the frozen baseline.
- Improve bent and crumpled median waveform correlation without degrading scan
  and phone-photo categories.
- Measure absolute RMSE, SNR, gain, runtime, and failure rate.

### Priority 3: Level B Real-Photo Validation Set

Collect anonymous phone photographs with a matched PDF, flat scan, or digital
waveform reference. This is the most important step for measuring real-world
accuracy rather than operational success.

Acceptance criteria:

- At least 20 distinct ECG records from more than one device or printer.
- Front, angled, and difficult-light variants where possible.
- Known layout and anonymous capture metadata.
- Matched waveform-fidelity and diagnosis-drift reports.

### Priority 4: Reproducible GPU Workload

Prepare a versioned remote job before renting GPU capacity.

The local machine is sufficient for development, inference, focused
benchmarks, and small evaluations. Use Vast.ai or another GPU provider only
after the training/evaluation job is reproducible locally on a small subset.

Acceptance criteria:

- Pinned environment and model/data checksums.
- Single command for training or large evaluation.
- Resume-capable checkpoints and experiment metadata.
- Downloadable aggregate results without patient-identifiable data.

### Priority 5: Artifact-Aware Fine-Tuning

Fine-tune only after the quality gate and real matched-reference validation are
in place. Training a diagnostic model now risks teaching it digitization
artifacts while hiding reconstruction failures.

Candidate directions:

- fine-tune a digitizer or reconstruction component on matched distortions;
- train diagnosis robustness with paired clean/digitized signals;
- calibrate class thresholds on a sufficiently supported validation set.

Acceptance criteria:

- Compare against the frozen Phase 2 baseline.
- Use patient-level splits and class-support reporting.
- Report uncertainty and category-specific regressions.
- Make no clinical-performance claim without an appropriate external test set.

### Priority 6: Structured Reporting

Add structured report generation only after unreliable inputs can be rejected
and diagnostic outputs are calibrated. Reports must preserve uncertainty and
quality warnings.

## First Task For The Next Session

Design and implement the first quality-gate benchmark:

1. Define a matched-fidelity failure label from PMcardio correlation and
   absolute metrics.
2. Extract candidate quality features already emitted by the digitizer.
3. Fit or tune an interpretable baseline gate.
4. Report false accepts, false rejects, and results by capture category.
5. Wire the gate into the Gradio UI only after benchmark review.
