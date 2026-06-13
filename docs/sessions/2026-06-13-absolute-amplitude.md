# Absolute-Amplitude Reconstruction

## Objective

Preserve the digitizer's calibrated millivolt signal before global z-score
normalization and quantify absolute reconstruction fidelity on the balanced
PMcardio benchmark.

## Implementation

- Added `DigitizedSignals` with:
  - `model_input`: existing globally normalized ECGFounder input.
  - `calibrated_millivolts`: filtered and denoised signal before z-score.
- Kept `ECGDigitiser.digitize()` backward compatible.
- Added `ECGDigitiser.digitize_with_calibrated()` for evaluation workflows.
- Extended isolated photo workers to optionally save calibrated outputs.
- Added bounded-alignment absolute metrics:
  - millivolt RMSE;
  - SNR in dB;
  - RMS amplitude gain ratio.

## Balanced 70-Image Results

- Calibrated outputs available: 70 / 70.
- Median RMSE: 0.1076 mV.
- Median RMSE bootstrap 95% CI: 0.0890 to 0.1332 mV.
- Median SNR: 3.83 dB.
- Median SNR bootstrap 95% CI: 2.66 to 5.62 dB.
- Median gain ratio: 0.9038.
- Median gain bootstrap 95% CI: 0.8889 to 0.9220.

The gain ratio indicates that reconstructed signal amplitude is typically about
10% lower than the reference. Scans have the strongest absolute fidelity
(`0.0716 mV`, `8.88 dB`), while bent and crumpled paper have negative median
SNR.

## Limitations

Absolute metrics use pixel-derived calibration after Corio's high-pass filter
and wavelet denoising. They quantify the current reconstruction pipeline, not
raw vendor digitizer output or clinical diagnostic accuracy.

The repeated 70-image run changed the normalized median correlation from
0.8188 to 0.8234, showing small run-to-run digitizer variation.
