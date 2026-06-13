# Expanded PMcardio Matched-Reference Benchmark

## Objective

Replace the initial 21-image smoke benchmark with a larger, balanced, and
content-addressed matched-reference subset.

## Implementation

- Added deterministic selection from image IDs available in every requested
  category.
- Added `--balanced-count`, `--selection-seed`, and `--required-image-ids`.
- Preserved the original smoke-test IDs `4`, `10`, and `14`.
- Added `selection_manifest.json` with a stable SHA-256 manifest identity.
- Added deterministic 2,000-resample bootstrap confidence intervals for mean
  and median waveform-shape correlation.
- Embedded the selection manifest in the benchmark report.
- Added calibrated pre-normalization mV outputs and absolute RMSE, SNR, and gain
  metrics.

## Selection

- Layout: `3x4+1R`
- Categories: 7 physical capture categories
- Images per category: 10
- Total images: 70
- Selected image IDs: `4, 10, 14, 40, 42, 43, 80, 81, 82, 92`
- Manifest ID:
  `22b4fbf08c7358bd68ecfe5f978609b9c446996a116de87af1fa5c6ab376e65d`

## Results

- Digitization success: 70 / 70.
- Overall median correlation: 0.8234.
- Overall median bootstrap 95% CI: 0.7394 to 0.8784.
- Overall median RMSE: 0.1076 mV.
- Overall median SNR: 3.83 dB.
- Overall median gain ratio: 0.9038.

| Category | N | Median correlation | Median RMSE mV | Median SNR dB | Median gain |
|---|---:|---:|---:|---:|---:|
| Bent paper | 10 | 0.2801 | 0.1822 | -0.27 | 0.891 |
| Crumpled paper | 10 | 0.4233 | 0.1870 | -0.09 | 0.902 |
| Doogee photos | 10 | 0.9384 | 0.0823 | 7.74 | 0.898 |
| iPhone photos | 10 | 0.8805 | 0.0933 | 5.52 | 0.903 |
| Samsung photos | 10 | 0.9132 | 0.0722 | 6.51 | 0.909 |
| Scans | 10 | 0.9746 | 0.0716 | 8.88 | 0.925 |
| Screens | 10 | 0.7754 | 0.1163 | 3.19 | 0.872 |

## Interpretation

The expanded result is consistent with the initial 21-image smoke benchmark.
Bent and crumpled paper are the primary digitization-fidelity weaknesses. The
median gain ratio of 0.904 indicates that reconstructed amplitude is typically
about 10% lower than the matched reference.

High Einthoven consistency does not guarantee high matched waveform fidelity.
For example, some scan, phone, and screen records have low matched correlation
despite positive Einthoven scores. Quality gating therefore needs multiple
independent signals.

Absolute metrics use the calibrated output after high-pass filtering and
denoising but before global normalization. This benchmark does not measure
clinical diagnostic accuracy.

## Verification

- Focused PMcardio tests: 11 passed.
- PMcardio and real-photo integration tests: 17 passed.
- Focused Ruff checks: passed.
- Manifest integrity: 70 rows, ten per category, no missing images.
