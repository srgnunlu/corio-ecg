# PMcardio Matched-Reference Diagnosis Drift

## Objective

Measure ECGFounder output consistency between matched PMcardio reference
signals and their digitized physical-image variants. Compare the current tiled
representation with experimental segment-aware mean aggregation.

## Method

- Convert each matched 2.5-second per-lead reference into the same tiled,
  globally normalized model input used by the digitizer.
- Cache reference inference by ECG identity, layout, and method.
- Compare all 150 ECGFounder probabilities without UI heart-rate heuristics.
- Report cosine similarity, threshold agreement, and mean absolute probability
  difference.
- Treat the result as consistency, not clinical diagnostic accuracy.

## Balanced 70-Image Results

| Method | Mean cosine | Mean agreement | Mean abs probability difference |
|---|---:|---:|---:|
| Tiled | 0.8939 | 96.68% | 0.0467 |
| Segment ensemble | 0.9238 | 97.73% | 0.0357 |

Bootstrap 95% confidence intervals:

- Tiled mean cosine: 0.8758 to 0.9126.
- Segment mean cosine: 0.9053 to 0.9414.
- Tiled mean agreement: 96.29% to 97.06%.
- Segment mean agreement: 97.33% to 98.14%.

Segment ensemble improves category-level mean cosine in all seven physical
capture categories. The largest category-level gain is for screen images:
`0.8772` to `0.9209`.

## Limitations

Segment ensemble is not uniformly better for every individual record. Some
crumpled, phone, scan, and bent-paper records retain higher tiled consistency.
The strategy therefore remains experimental and is not enabled in production
or the Gradio UI.

The benchmark measures model-output consistency against matched reference
signals. It does not contain clinical diagnosis labels and cannot establish
clinical accuracy.

## Outputs

- `results/pmcardio-reference/pmcardio_diagnosis_drift.json`
- `results/pmcardio-reference/pmcardio_diagnosis_drift.csv`
