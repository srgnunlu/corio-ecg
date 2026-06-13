# Corio ECG Next Priorities

**Date:** 2026-06-13
**Current phase:** Phase 2 evaluation and digitization robustness

## Objective

Turn the current research prototype into a reproducible evaluation baseline that
can identify whether paper-layout reconstruction, digitization fidelity, or the
diagnostic model is responsible for each observed performance loss.

## Progress

- Priority 1 complete: segment-aware round-trip evaluation.
- Priority 2 complete: balanced 70-image PMcardio benchmark with manifest and
  bootstrap confidence intervals.
- Priority 3 complete: calibrated mV output with RMSE, SNR, and gain metrics.
- Priority 4 complete: matched-reference diagnosis drift for tiled and
  segment-ensemble inference.
- Priority 5 pending: split and publish verified work packages.

## Priority 1: Segment-Aware Diagnosis Evaluation

Standard 3x4 paper ECGs contain four sequential time columns, while ECGFounder
expects a simultaneous 10-second 12-lead tensor. The current digitizer expands
each printed lead segment across the full model input. This priority evaluates
an alternative without changing production diagnosis behavior:

1. Build one sparse 12-lead model input for each printed time column.
2. Run ECGFounder independently on each column input.
3. Aggregate the four probability vectors.
4. Report the result beside the current tiled-signal method.

Acceptance criteria:

- Unit tests verify exact 3x4 and 6x2 lead membership.
- Mean and maximum probability aggregation are deterministic and validated.
- Round-trip results retain the current metrics and add segment-aware metrics.
- A smoke benchmark records whether segment-aware inference improves matched
  ground-truth metrics or only changes baseline agreement.

## Priority 2: Larger Matched-Reference Benchmark

Expand PMcardio beyond the current 21-image smoke subset using a balanced,
versioned manifest. Prioritize bent, crumpled, and degraded paper categories.

Acceptance criteria:

- The subset manifest records category, source member, and reference identity.
- Results include confidence intervals and category-level failure counts.
- Benchmark outputs are reproducible without downloading the full archive.

## Priority 3: Absolute-Amplitude Reconstruction

Preserve a calibrated, pre-normalization digitizer output so absolute millivolt
error, SNR, and amplitude-dependent morphology can be evaluated.

Acceptance criteria:

- Normalized model input and calibrated signal are separate explicit outputs.
- Matched-reference evaluation reports mV RMSE and SNR where calibration exists.
- Existing diagnosis and UI behavior remain backward compatible.

## Priority 4: Matched-Reference Diagnosis Drift

Run ECGFounder on PMcardio reference signals and their matched digitized signals,
then compare probabilities and thresholded diagnoses.

Acceptance criteria:

- Comparisons use the same preprocessing and model settings.
- Results are split by capture category and digitization quality.
- Claims remain limited to model consistency, not clinical accuracy.

## Priority 5: Stabilize and Publish Work Packages

After each priority passes its acceptance criteria:

1. Review generated artifacts and exclude machine-specific or oversized files.
2. Split source, tests, documentation, and benchmark results into intentional
   commits.
3. Push the branch only after the full test suite and focused lint checks pass.

## Execution Order

The first implementation package is Priority 1 because it directly tests the
current primary blocker and determines whether larger synthetic evaluation is
scientifically useful. Priorities 2-4 follow based on that result.
