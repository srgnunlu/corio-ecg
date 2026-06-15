# Phase 3 Experiment: Layout Segments v1

**Date:** 2026-06-15
**Status:** Rejected after pilot
**Factor:** Ignore the full-width rhythm strip during vendor canonicalization
**Version:** `layout-segments-v1`

## Scope

The PMcardio pages use `3x4+1R`. This experiment preserves the actual layout in
audit metadata but constrains the vendor canonicalizer to `3x4`, testing whether
short layout-specific segments are more faithful than allowing the full-width
Lead II rhythm strip to overwrite the Lead II canonical row.

The experiment does not reinterpret `raw_lines`. Those values are pixel
Y-coordinates, not microvolts, and the previous position-based override was
already proven invalid.

## Pilot Result

The pilot used 14 matched images: two ECG identities across all seven capture
categories.

- Digitization success: `14/14`
- Target-category aggregate improvement: none
- Locked pilot promotion gate: failed on supported-category runtime
- Crumpled image 10 Lead II correlation delta: `-0.169`

Most category fidelity changes were approximately zero. Removing the rhythm
strip did not systematically improve Lead II or aggregate fidelity.

## Decision

Reject after pilot and do not run the full 70-image benchmark. The candidate
does not demonstrate a target-category benefit and worsens a reviewable
crumpled Lead II result.

Keep `layout_segments` available only as an audited experiment mode. Do not
enable it in the default pipeline or UI.

## Pilot Artifacts

Pilot artifacts are local-only under:

```text
results/digitization-experiments/layout-segments-v1-pilot/
```
