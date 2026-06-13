# Evaluation Methodology

## Two Separate Questions

Corio ECG evaluates two different properties. They must not be interpreted as
the same metric.

1. **Ground-truth diagnostic performance**
   - Primary benchmark compares model probabilities with ECGFounder's official
     150-output PTB-XL target vectors.
   - Uses raw ECGFounder probabilities without UI heart-rate heuristics.
   - Reports macro AUROC, macro average precision, micro F1, macro F1, and
     per-class metrics.

2. **Semantic SCP subset performance**
   - Maps only PTB-XL SCP codes with a direct semantic equivalent in
     ECGFounder's output vocabulary.
   - Provides an independently inspectable subset metric, but is not the
     official ECGFounder benchmark.

3. **Round-trip consistency**
   - Compares ECGFounder outputs before and after signal → image → signal
     conversion.
   - Reports cosine similarity, threshold agreement, probability difference,
     and signal correlation.
   - Measures digitization-induced output drift, not diagnostic correctness.

4. **Matched image-to-signal fidelity**
   - Compares digitized PMcardio ECG images with their matched digital signals.
   - Measures printed waveform-shape recovery using per-lead correlation after
     up to +/-100 ms alignment.
   - Preserves the digitizer's calibrated signal before global normalization
     and reports millivolt RMSE, SNR, and amplitude gain ratio.

5. **Matched-reference diagnosis drift**
   - Compares ECGFounder probabilities on matched PMcardio reference and
     digitized signals.
   - Reports tiled and experimental segment-ensemble consistency separately.
   - Measures model-output consistency, not diagnostic correctness.

## Label Mapping Policy

Official targets come from the upstream ECGFounder repository's
`csv/ptbxl_label.csv`, pinned by `scripts/download_ecgfounder_eval_labels.py`.
The semantic subset mapping is defined in `src/training/ptbxl_labels.py`.

- Codes are mapped by presence in `scp_codes`; PTB-XL confidence values may be
  zero even when a statement is present.
- Ambiguous mappings are excluded.
- Multiple SCP codes may map to one ECGFounder output.
- A class enters AUROC/AP calculation only when the evaluated subset contains
  both negative examples and the configured minimum positive support. The CLI
  default is five positive examples.

## Commands

```bash
python -m src.training.evaluate --max-samples 500 --threshold 0.5
python -m src.training.evaluate_roundtrip --max-samples 500 --threshold 0.5
python scripts/download_pmcardio_reference_subset.py --balanced-count 10
python scripts/evaluate_pmcardio_reference.py --timeout 180
python scripts/evaluate_pmcardio_diagnosis_drift.py
```

Round-trip results include official and semantic-subset metrics for each
digitized scenario, the matched clean-signal baseline, and metric deltas.

Synthetic digitization outputs are paired with JSON audit metadata. Resume
logic reuses an output only when its pipeline version, shape, finite values,
and active-lead count pass validation. Image selection uses numeric ECG IDs so
`--max-samples` selects the same PTB-XL records in generation, digitization,
and evaluation.

Standard paper layouts contain sequential lead segments rather than a
simultaneous full-duration 12-lead recording. Full-signal Pearson correlation
therefore remains a conservative diagnostic metric; matched printed-segment
fidelity is required for detailed reconstruction analysis.

Round-trip evaluation also reports an experimental segment ensemble. It builds
one sparse 12-lead input for each printed paper column, runs ECGFounder on each
column independently, and averages the probability vectors. This measures
whether respecting sequential paper timing reduces diagnosis drift. Sparse
column inputs remain outside ECGFounder's expected input distribution, so this
method is not used by the production or Gradio diagnosis path.

## Matched PMcardio Reference Benchmark

The current matched-reference batch uses 70 GPL-3.0-or-later PMcardio images:
ten ECG IDs across seven physical capture categories. All selected IDs are
available in every category. The deterministic selection preserves the original
three smoke-test IDs and records a content-addressed manifest. The downloader
reads only selected members from the 33.9 GB Zenodo ZIP using HTTP range
requests.

- Digitization success: 70 / 70.
- Median per-image median shifted correlation: 0.823.
- Bootstrap 95% confidence interval for the median: 0.739 to 0.878.
- Median absolute RMSE: 0.108 mV.
- Median absolute SNR: 3.83 dB.
- Median amplitude gain ratio: 0.904.
- Median scanned-image correlation: 0.975.
- Bent and crumpled paper remain the weakest categories, with median
  correlations of 0.280 and 0.423.

Einthoven consistency and waveform fidelity measure different failure modes.
Some low-fidelity records retain a high Einthoven score, so physical lead
consistency must not be used as the only acceptance criterion.

Absolute metrics use the digitizer's pixel-derived calibration after high-pass
filtering and denoising but before global z-score normalization. They measure
reconstruction fidelity, not clinical diagnostic accuracy.

## Matched PMcardio Diagnosis Drift

On the balanced 70-image subset, segment-aware mean aggregation improves
matched-reference consistency relative to the tiled representation:

| Method | Mean cosine | Mean agreement | Mean absolute probability difference |
|---|---:|---:|---:|
| Tiled | 0.8939 | 96.68% | 0.0467 |
| Segment ensemble | 0.9238 | 97.73% | 0.0357 |

Segment ensemble improves the category-level mean cosine in all seven physical
capture categories, but individual records can still perform worse than tiled
inference. It remains an experimental evaluation strategy and is not used by
the production or Gradio diagnosis path.
