# Phase 3 Experiment: Vendor Dewarping Retry v1

**Date:** 2026-06-15
**Status:** Rejected after early pilot stop
**Factor:** Existing vendor dewarping retry
**Version:** `vendor-dewarping-retry-v1`

## Scope

The experiment used the existing vendor dewarping retry without algorithm
changes. Every image ran in an isolated worker with a strict `60 second` hard
timeout. The pilot was limited to the first 14 balanced PMcardio images and
would stop immediately on a timeout.

## Pilot Result

- Bent image 4: success in approximately `8.3 seconds`; default result retained
- Crumpled image 4: hard timeout after approximately `61.1 seconds`
- Remaining pilot images: not run

No worker processes remained after stopping the pilot.

## Decision

Reject on the current 24 GB macOS development machine. The predefined timeout
stop condition occurred on the second pilot image, so a full benchmark would
be unsafe and unnecessary.

Do not enable vendor dewarping retry in the default pipeline or UI. Future
dewarping work requires a separately budgeted remote job with explicit memory,
timeout, and resume controls.

## Pilot Artifacts

Pilot artifacts are local-only under:

```text
results/digitization-experiments/dewarping-retry-v1-pilot/
```
