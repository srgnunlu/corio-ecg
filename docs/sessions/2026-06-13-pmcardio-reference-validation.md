# PMcardio Matched Reference Validation

## Source

- Dataset: PMcardio ECG Image Database (PM-ECG-ID)
- Official record: <https://zenodo.org/records/13617673>
- DOI: `10.5281/zenodo.13617673`
- License: GPL-3.0-or-later
- Full archive size: 33.9 GB

The local downloader uses HTTP range requests to extract only the requested
images and matched reference arrays. The default subset contains 21 images:
three ECG IDs across bent, crumpled, Doogee, iPhone, Samsung, scan, and screen
categories.

## Implementation

- Added `scripts/download_pmcardio_reference_subset.py`.
- Added `scripts/evaluate_pmcardio_reference.py`.
- Added matched printed-segment metrics in
  `src/training/reference_fidelity.py`.
- Added focused tests in `tests/test_pmcardio_reference.py`.
- Added a weak-result orientation retry in `src/pipeline/digitize.py`.

## Results

- Digitization success: 21 / 21.
- Overall median per-image median correlation: `0.811`.
- Scan category median correlation improved from `0.210` to `0.971`.
- The scan improvement came from selecting the correct opposite orientation
  when the first digitization attempt had poor layout geometry.

Category median correlations:

| Category | Correlation |
|---|---:|
| Bent paper | 0.359 |
| Crumpled paper | 0.635 |
| Doogee photos | 0.943 |
| iPhone photos | 0.878 |
| Samsung photos | 0.936 |
| Scans | 0.971 |
| Screens | 0.811 |

The benchmark measures normalized waveform-shape fidelity after up to
`+/-100 ms` per-lead alignment. It does not yet measure absolute millivolt
amplitude or clinical diagnostic accuracy.

## Commands

```bash
.venv/bin/python scripts/download_pmcardio_reference_subset.py
.venv/bin/python scripts/evaluate_pmcardio_reference.py --timeout 180 --overwrite
.venv/bin/pytest -q tests/test_pmcardio_reference.py
```
