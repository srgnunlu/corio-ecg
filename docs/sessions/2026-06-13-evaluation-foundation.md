# Evaluation Foundation and ECGFounder Parity

## Objective

Establish a reproducible, ground-truth-based evaluation foundation and identify
the highest-priority technical blocker.

## Completed

- Added real multi-label AUROC, average precision, and F1 metrics.
- Added an explicit semantic PTB-XL SCP-code subset mapping.
- Added support for ECGFounder's official 150-output PTB-XL target vectors.
- Added a pinned, validated official-label download script.
- Separated raw model benchmarking from UI heart-rate heuristics.
- Added official and semantic matched-baseline deltas to round-trip evaluation.
- Pinned external digitizer/image-generation revisions and captured the local
  Open-ECG-Digitizer patch.
- Restricted pytest discovery to the project's own test suite.
- Corrected README and evaluation-methodology claims.

## Critical Model Fix

The local Net1D implementation was not equivalent to the official ECGFounder
implementation:

- BatchNorm executed locally but is disabled in official PTB-XL inference.
- The first block's pre-activation behavior differed.
- Residual channel padding was placed at the end instead of centered.

After correction, the local and official models produce exactly equal logits
for the same checkpoint and input (`max_abs_diff = 0.0`).

## Verified Results

Full PTB-XL test fold:

- 2,203 / 2,203 records processed.
- 2,198 records matched to official ECGFounder targets.
- Official macro AUROC: 0.8679.
- Official macro average precision: 0.4454.
- Official micro F1: 0.5885.
- Official macro F1: 0.3673.

Audited 50-record round-trip smoke benchmark:

- Regenerated all 150 outputs with synchronized canonical segment expansion and
  versioned audit metadata.
- Overall class agreement: 94.56% to 94.80%.
- Cosine similarity: 0.5394 to 0.5492.
- Mean Pearson signal correlation: 0.1850 to 0.2175.
- Official micro F1 delta: -0.1922 to -0.2032.
- Mean Einthoven consistency: clean 0.9893, moderate 0.9473, hard 0.8744.
- Only three classes meet minimum support, so this is not a robust accuracy
  estimate.

## Current Priority

The diagnostic model integration is now validated against upstream behavior.
Paper-layout reconstruction is the primary technical blocker: standard 3x4
paper ECGs contain sequential lead segments, while ECGFounder expects a
simultaneous 10-second 12-lead tensor. A larger corrected round-trip evaluation
and real-phone-photo validation are the next critical gates.
