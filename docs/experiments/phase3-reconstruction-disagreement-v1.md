# Phase 3 Experiment: Reconstruction Disagreement v1

**Date:** 2026-06-15
**Status:** Research metric implemented; shadow perturbation rejected after pilot
**Factor:** Stability between two non-destructive inference attempts

## Rationale

The PMcardio tune review found two low-fidelity `3x4+1R` records that appeared
healthy under all active inference features. Tightening the existing layout,
lead-count, Einthoven, or grid-density thresholds cannot reliably detect these
failures.

This experiment compares two reconstructions of the same image after a small,
clinically irrelevant preprocessing perturbation. A stable extraction should
preserve morphology and calibrated amplitude. The comparison does not use the
matched reference signal and could therefore become an inference-time feature.

## Implemented Metric

`src/evaluation/reconstruction_disagreement.py` reports:

- per-lead correlation after bounded horizontal alignment;
- median and minimum correlation;
- median normalized RMSE;
- count of leads below `0.8` correlation;
- calibrated mV RMSE and gain disagreement when both attempts provide
  calibrated signals.

## Guardrails

- Research-only; not part of the runtime quality feature contract or gate.
- Run and select perturbations only on the PMcardio tune split.
- Keep the 60-group PMcardio test split locked until the perturbation and
  candidate thresholds are frozen.
- Stability is not fidelity: two attempts can agree and still both be wrong.
- Reject perturbations that materially change good supported inputs or add
  unacceptable runtime.

## Shadow Perturbation Pilot

The existing conservative shadow normalization was tested as the second attempt
on the two remaining layout-scope false accepts.

| Record | Shadow applied | Median pair correlation | Minimum pair correlation | Median calibrated RMSE |
|---|---:|---:|---:|---:|
| Doogee image 74 | no | `0.99941` | `0.98738` | `0.00920 mV` |
| iPhone image 26 | yes | `0.99999` | `0.99999` | `0.00087 mV` |

Both low-fidelity reconstructions remained extremely stable. Shadow
normalization is therefore rejected as the perturbation for this feature.

## Next Pilot

Evaluate a bounded grid-preserving scale or translation perturbation on tune
records. Select it only if disagreement separates failures without
destabilizing good records or adding unacceptable runtime.
