# Real Phone ECG Photo Validation Protocol

This protocol evaluates Corio ECG on real phone photographs. It is a research
validation workflow, not a clinical decision tool.

## Privacy

- Remove or crop patient name, ID, date of birth, barcode, and institution ID.
- Do not upload photographs containing identifiable patient information.
- Keep the complete ECG grid, lead labels, calibration pulse, and page edges.

## Minimum Useful Batch

Start with at least 5 different ECG sheets and photograph each sheet in three
conditions, for a minimum of 15 photographs:

1. `front`: straight-on, evenly lit, full page visible.
2. `angle`: approximately 20-35 degrees off-axis.
3. `lowlight`: uneven or weaker indoor lighting, without flash glare.

An ideal validation batch contains 20 or more different ECG sheets from more
than one printer/device.

## Reference Levels

### Level A: Photos Only

Photos alone allow measurement of:

- digitization success/failure rate;
- detected layout and lead count;
- number of active digitized leads;
- Einthoven consistency;
- processing time and retry behavior;
- visible failure patterns.

### Level B: Photos Plus Reference

For each photographed sheet, also provide one of the following:

- original digital PDF or image exported by the ECG device;
- corresponding WFDB/digital waveform;
- straight, high-resolution flatbed scan.

This enables matched printed-segment signal fidelity and diagnosis-drift
measurement. A reference is required for a defensible accuracy claim.

## Capture Requirements

- Upload original-resolution JPG or PNG files; avoid messaging-app compression.
- Keep all page edges visible where possible.
- Avoid fingers, shadows, glare, folds, and objects covering traces or labels.
- Do not crop individual leads; photograph the complete 12-lead sheet.
- Record the paper layout when known, such as `3x4+1R` or `6x2+1R`.

## Naming

Use a shared anonymous case ID for all variants:

```text
case001__front.jpg
case001__angle.jpg
case001__lowlight.jpg
case001__reference.png
```

Optional metadata:

```csv
case_id,variant,layout,device_model,reference_available,notes
case001,front,3x4+1R,unknown,true,even indoor light
case001,angle,3x4+1R,unknown,true,approximately 25 degrees
case001,lowlight,3x4+1R,unknown,true,no flash
```

## Level B Manifest

Store the local Level B dataset under `data/level-b-matched/`. The directory is
gitignored. Its `manifest.json` must follow the locked contract in
`configs/matched_photo_manifest_v1.yaml`.

Each photo record must include:

- anonymous `case_id` and unique `photo_id`;
- capture variant, layout, phone/device model, and printer model;
- safe relative photo and matched-reference paths with SHA-256 hashes;
- reference type, pre-registered `tune` or `test` split, and
  `phi_reviewed: true`.

All variants of one case must remain in one split and point to the same matched
reference. Validate the complete local dataset with:

```bash
.venv/bin/python scripts/validate_matched_photo_dataset.py
```

The aggregate validation report contains no source paths or per-case records.

## Initial Acceptance Criteria

- At least 90% of photos digitize without an exception.
- At least 90% of successful outputs contain 12 active leads.
- Median Einthoven consistency is at least 0.70.
- No silent reuse of stale or unaudited outputs.
- Level B cases include matched printed-segment fidelity and diagnosis-drift
  results.
