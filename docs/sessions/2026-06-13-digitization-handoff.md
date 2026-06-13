# Digitization Benchmark Handoff

## Session State

- Date: 2026-06-13
- Branch: `feature/test-ui-gradio`
- Branch status at handoff: substantial uncommitted project changes from the
  current evaluation and digitization work. Inspect `git status` before making
  new edits; do not revert unrelated changes.
- Do not revert unrelated user files:
  - `docs/superpowers/`
  - `memory_test.py`

## Completed In This Session

### ECGFounder Evaluation Foundation

- Corrected local Net1D behavior to exactly match the official ECGFounder
  implementation (`max_abs_diff = 0.0`).
- Added official ECGFounder PTB-XL target evaluation.
- Full PTB-XL test fold:
  - 2,203 / 2,203 records processed.
  - 2,198 matched official targets.
  - Official macro AUROC: 0.8679.
  - Official macro average precision: 0.4454.
  - Official micro F1: 0.5885.
  - Official macro F1: 0.3673.

### Digitization Pipeline

- Replaced independent per-lead threshold alignment with synchronized canonical
  segment expansion in `src/pipeline/digitize.py`.
- Preserved paper-column timing encoded by canonical NaN boundaries.
- Cropped full-width rhythm strips to the synchronized lead column before
  tiling in multi-column layouts.
- Fixed negative Einthoven scores being ignored by quality-warning logic.
- Removed the obsolete `_align_leads_to_origin` implementation.

### Audited Synthetic Benchmark

- Added digitization pipeline version 4 and paired JSON audit metadata.
- Existing signals are reused only when version, shape, finite values, and
  active-lead checks pass.
- Added isolated per-image subprocess execution and timeout handling.
- Disabled dewarping retry for controlled synthetic images because it can hang
  and provides no value for flat generated images.
- Fixed numeric ECG ID ordering so `--max-samples` selects the same records in
  generation, digitization, and evaluation.
- Regenerated and audited all 150 outputs: 50 clean, 50 moderate, 50 hard.

## Verified Round-Trip Results

| Scenario | Cosine similarity | Agreement | Pearson | Micro F1 delta |
|---|---:|---:|---:|---:|
| Clean | 0.5453 | 94.80% | 0.2175 | -0.2032 |
| Moderate | 0.5394 | 94.67% | 0.2061 | -0.1922 |
| Hard | 0.5492 | 94.56% | 0.1850 | -0.1922 |

| Scenario | Mean Einthoven | 12 active leads | Quality warnings |
|---|---:|---:|---:|
| Clean | 0.9893 | 50 / 50 | 0 |
| Moderate | 0.9473 | 49 / 50 | 1 |
| Hard | 0.8744 | 48 / 50 | 10 |

Interpretation: physical lead consistency now degrades as expected with image
difficulty, but diagnostic drift is similar across all difficulty levels. The
dominant remaining problem is paper-layout reconstruction: a standard 3x4 ECG
contains sequential 2.5-second lead segments, while ECGFounder expects a
simultaneous 10-second 12-lead tensor.

## Synthetic Benchmark State

- Current audited synthetic outputs:
  - clean: 50 / 50
  - moderate: 50 / 50
  - hard: 50 / 50
- The latest full-project verification is recorded under `Current Verification`
  below.

## Real Phone Photo Validation

- Received and evaluated 10 anonymous real ECG photographs:
  - 6 close photographs of printed sheets.
  - 4 photographs of ECGs displayed on monitors.
- Added `scripts/evaluate_real_photos.py` with isolated workers, hard timeout,
  audit records, JSON/CSV reports, and optional known-layout hints.
- Added a conservative portrait-screen crop before general auto-rotation.
- Added a weak-result opposite-orientation retry.
- Latest metadata-informed report:
  - Success: 10 / 10, improved from 8 / 10.
  - Twelve active leads: 9 / 10.
  - Overall median Einthoven: 0.657.
  - Printed-sheet group median Einthoven: 0.773.
  - Screen group median Einthoven: 0.123.
- Interpretation:
  - Printed-sheet group passes the initial operational criteria.
  - Monitor photographs now digitize but remain low-quality and must keep
    warnings. Successful extraction is not clinical reliability.
- Current report:
  `results/real-phone-layout-informed/real_phone_evaluation.json`

## Matched PMcardio Reference Validation

- Official source: <https://zenodo.org/records/13617673>
- Dataset: PMcardio ECG Image Database, GPL-3.0-or-later.
- Added a range-request downloader that extracts a small subset without
  downloading the full 33.9 GB ZIP:
  `scripts/download_pmcardio_reference_subset.py`.
- Added matched printed-segment fidelity metrics:
  `src/training/reference_fidelity.py`.
- Added evaluator:
  `scripts/evaluate_pmcardio_reference.py`.
- Latest balanced report:
  - Success: 70 / 70.
  - Overall median per-image waveform-shape correlation: 0.823.
  - Bootstrap 95% confidence interval for the median: 0.739 to 0.878.
  - Median absolute RMSE: 0.108 mV.
  - Median absolute SNR: 3.83 dB.
  - Median amplitude gain ratio: 0.904.
  - Scan category median correlation: 0.975.
  - Screen category median correlation: 0.773.
  - Bent and heavily degraded paper remain the weakest cases.
- This measures normalized waveform shape and calibrated reconstruction
  fidelity, not clinical diagnostic accuracy.
- Current report:
  `results/pmcardio-reference/pmcardio_reference_fidelity.json`

## Matched PMcardio Diagnosis Drift

- Added matched-reference ECGFounder consistency evaluation:
  `scripts/evaluate_pmcardio_diagnosis_drift.py`.
- Balanced 70-image result:
  - Tiled mean cosine: 0.8939.
  - Segment-ensemble mean cosine: 0.9238.
  - Tiled agreement: 96.68%.
  - Segment-ensemble agreement: 97.73%.
- Segment ensemble improves every category mean but is not uniformly better for
  every individual record.
- Current report:
  `results/pmcardio-reference/pmcardio_diagnosis_drift.json`

## Current Verification

- Full test suite: 129 passed, 1 skipped.
- Focused digitization/reference/drift tests: passed.
- Ruff on files touched by the latest reference/photo work: passed.
- `git diff --check`: passed.
- Repo-wide Ruff still reports unrelated pre-existing style issues.

## Next Session Plan

1. Improve monitor-photo quality after cropping:
   - reject or strongly warn on low Einthoven consistency;
   - investigate layout selection for screen images;
   - avoid treating extraction success as acceptable quality.
2. Increase the corrected synthetic benchmark beyond 50 records with both
   tiled and segment-ensemble evaluation.
3. Add class-support and uncertainty reporting before making diagnostic claims.
4. Keep all claims research-only; no clinical-validity claim is supported.

## Useful Commands

```bash
.venv/bin/pytest -q
.venv/bin/python scripts/download_pmcardio_reference_subset.py --balanced-count 10
.venv/bin/python scripts/evaluate_pmcardio_reference.py --timeout 180 --overwrite
.venv/bin/python scripts/evaluate_pmcardio_diagnosis_drift.py
.venv/bin/python scripts/evaluate_real_photos.py --modes default \
  --use-metadata-layouts --timeout 180 --overwrite \
  --signal-dir data/real-phone/signals-layout-informed \
  --output-dir results/real-phone-layout-informed
.venv/bin/ruff check src/pipeline/digitize.py src/web/ecg_plot.py \
  scripts/digitize_synthetic_images.py scripts/run_full_evaluation.py \
  tests/test_digitize_synthetic_images.py
.venv/bin/python scripts/run_full_evaluation.py \
  --skip-generation --max-samples 50 --digitizer-timeout 25
```
