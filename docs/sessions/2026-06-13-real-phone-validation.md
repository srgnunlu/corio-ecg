# Real Phone Photo Validation

## Input Batch

- 10 anonymous real ECG photographs were added locally under
  `data/real-phone/photos`.
- 6 are close phone photographs of printed ECG sheets (`front`).
- 4 are photographs of ECGs displayed on a monitor (`screen`).
- No matched PDF, flat scan, or digital waveform was provided.
- This is a Level A operational-quality evaluation only. It does not measure
  signal fidelity or diagnostic accuracy.

## Implementation

- Added `scripts/evaluate_real_photos.py`.
- Each photo attempt runs in an isolated subprocess with a hard timeout.
- Default and dewarping-retry modes are evaluated independently.
- Source SHA-256, pipeline version, mode, and layout hint protect against stale
  audit-record reuse.
- Per-photo diagnostics and aggregate JSON/CSV reports are written to
  `results/real-phone`.
- Added focused tests in `tests/test_evaluate_real_photos.py`.

## Results

| Capture type | Mode | Success | 12 active leads among successes | Median Einthoven |
|---|---|---:|---:|---:|
| Front | Default | 6 / 6 | 100% | 0.657 |
| Screen | Default | 2 / 4 | 100% | -0.443 |
| Front | Retry | 3 / 6 | 100% | 0.703 |
| Screen | Retry | 0 / 4 | N/A | N/A |

Overall default mode:

- Digitization success: 8 / 10 (80%).
- 12 active leads among successes: 8 / 8 (100%).
- Median Einthoven consistency: 0.563.
- No timeout.

Overall retry mode:

- Digitization success: 3 / 10 (30%).
- Six workers were killed with exit code `-9`.
- One worker reached the 180-second timeout.

The dewarping retry is unsafe on the current 24 GB macOS environment and should
remain disabled in the local Gradio service.

## Layout Finding

`case006__front.jpg` was auto-detected as `3x4+1R`, producing an Einthoven score
of `-0.941`. A focused rerun with the visually correct `standard_6x2` hint
produced `0.972`. Known printer layouts should therefore be supplied explicitly
and included in audit identity.

A separate metadata-informed default run was written to
`results/real-phone-layout-informed`:

| Capture type | Success | 12 active leads among successes | Median Einthoven |
|---|---:|---:|---:|
| Front | 6 / 6 | 100% | 0.773 |
| Screen | 4 / 4 | 75% | 0.123 |
| Overall | 10 / 10 | 90% | 0.657 |

The front-photo group passed all initial Level A acceptance criteria after
supplying the known layout. A conservative pre-crop now finds an already-upright
landscape ECG page inside portrait monitor photos before general auto-rotation.
This raised mixed-batch success from `8 / 10` to `10 / 10`.

The complete mixed batch still fails the active-lead and median Einthoven
criteria because monitor photographs remain lower quality. Successful
digitization must not be interpreted as clinical reliability.

## Commands

```bash
.venv/bin/python scripts/evaluate_real_photos.py \
  --modes default retry \
  --timeout 180

.venv/bin/python scripts/evaluate_real_photos.py \
  --modes default \
  --use-metadata-layouts \
  --signal-dir data/real-phone/signals-layout-informed \
  --output-dir results/real-phone-layout-informed

.venv/bin/pytest -q tests/test_evaluate_real_photos.py
```
