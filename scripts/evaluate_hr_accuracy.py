"""Validate heart-rate accuracy against PMcardio rhythm-strip ground truth.

Compares three heart-rate estimates per matched ECG image:

  * reference   — beats detected on the clean reference rhythm strip
                  (data/reference/.../rhythms.npz), the ground truth.
  * baseline    — current production path: estimate_heart_rate_bpm on the
                  cropped/tiled diagnosis signal (digitized-calibrated cache).
  * rhythm_fix  — Phase C.1 path: estimate_rhythm_hr on the uncropped
                  full-duration rhythm strip recovered by the digitizer.

The C.1 success criterion (master-roadmap-v1.md) is, on the tune set:
median absolute HR error < 5 bpm and a >20 bpm deviation rate < 10%
(the tiled path sits near 30%).

Re-digitization runs in an isolated subprocess per image (the CPU digitizer
does not release its working set), and rhythm strips are cached so reruns are
cheap.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.rhythm import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    estimate_heart_rate_bpm,
    estimate_rhythm_hr,
)

DEFAULT_DATA_DIR = Path("data/reference/pmcardio")
DEFAULT_OUTPUT_DIR = Path("results/hr-accuracy")
DEFAULT_TIMEOUT_SECONDS = 240
_MIN_HR = 25.0
_MAX_HR = 220.0
LARGE_ERROR_BPM = 20.0


def reference_hr(rhythm_lead: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> float | None:
    """Ground-truth HR from a clean reference rhythm lead via median RR.

    Independent of the production estimators: bandpass to QRS energy, take a
    smoothed absolute envelope, detect peaks with a 300 ms refractory gap, and
    report 60 / median(RR) over physiologic intervals.
    """
    lead = np.asarray(rhythm_lead, dtype=np.float64)
    if lead.size < sample_rate or float(np.std(lead)) < 1e-6:
        return None

    nyquist = sample_rate / 2.0
    sos = butter(N=3, Wn=[5.0 / nyquist, 30.0 / nyquist], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, lead)
    envelope = np.abs(filtered)
    window = max(int(sample_rate * 0.05), 1)
    envelope = np.convolve(envelope, np.ones(window) / window, mode="same")

    peaks, _ = find_peaks(
        envelope,
        distance=int(sample_rate * 0.3),
        prominence=float(np.std(envelope)) * 0.5,
    )
    if len(peaks) < 3:
        return None

    rr_seconds = np.diff(peaks) / sample_rate
    min_rr, max_rr = 60.0 / _MAX_HR, 60.0 / _MIN_HR
    rr_seconds = rr_seconds[(rr_seconds >= min_rr) & (rr_seconds <= max_rr)]
    if rr_seconds.size == 0:
        return None
    bpm = 60.0 / float(np.median(rr_seconds))
    if bpm < _MIN_HR or bpm > _MAX_HR:
        return None
    return bpm


def _digitize(
    image_path: Path,
    category: str,
    *,
    layout_hint: str | None,
    timeout_seconds: int,
    output_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Re-digitize one image in an isolated process.

    Returns (model_input, rhythm_strip). Baseline (tiled) and fix (strip) HR are
    both derived from this single digitization, so the comparison isolates the
    C.1 change (uncropped rhythm strip + RR method) from any digitization noise.
    Uses a single cheap pass (no retries) for speed; both arms see the same
    signal, so the comparison stays fair.
    """
    model_path = output_dir / "model" / category / f"{image_path.stem}.npy"
    rhythm_path = output_dir / "rhythm-strips" / category / f"{image_path.stem}.npy"
    if model_path.exists():
        strip = np.load(rhythm_path) if rhythm_path.exists() else None
        return np.load(model_path), strip

    calib_out = output_dir / "scratch" / category / f"{image_path.stem}_calib.npy"
    command = [
        sys.executable,
        "-m",
        "scripts.digitize_one",
        "--image",
        str(image_path),
        "--model-output",
        str(model_path),
        "--calibrated-output",
        str(calib_out),
        "--rhythm-output",
        str(rhythm_path),
        "--no-retries",
    ]
    if layout_hint:
        command += ["--layout-hint", layout_hint]

    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            timeout=timeout_seconds,
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"  digitization failed: {exc}", flush=True)
        return None, None

    model = np.load(model_path) if model_path.exists() else None
    strip = np.load(rhythm_path) if rhythm_path.exists() else None
    return model, strip


def _error_stats(errors: list[float]) -> dict[str, Any]:
    if not errors:
        return {"n": 0}
    array = np.asarray(errors, dtype=np.float64)
    return {
        "n": int(array.size),
        "median_abs_err_bpm": round(float(np.median(array)), 2),
        "mean_abs_err_bpm": round(float(np.mean(array)), 2),
        "large_error_rate": round(float(np.mean(array > LARGE_ERROR_BPM)), 4),
        "within_5bpm_rate": round(float(np.mean(array <= 5.0)), 4),
    }


def evaluate(
    data_dir: Path,
    output_dir: Path,
    *,
    timeout_seconds: int,
    max_images: int | None,
) -> dict[str, Any]:
    metadata = pd.read_csv(data_dir / "subset_metadata.csv")
    if max_images is not None:
        metadata = metadata.head(max_images)
    rhythms = np.load(data_dir / "rhythms.npz")

    records: list[dict[str, Any]] = []
    for index, row in metadata.iterrows():
        relative_path = str(row["Image relative path"])
        category = str(row["category"])
        layout = str(row["ECG format"])
        reference_key = str(row["reference_key"])
        image_path = data_dir / "images" / relative_path
        stem = image_path.stem
        print(f"{index + 1}/{len(metadata)} {category}/{stem}", flush=True)

        reference_rhythm = rhythms[reference_key]
        if reference_rhythm.size == 0:
            continue
        ref_hr = reference_hr(reference_rhythm[:, 0])
        if ref_hr is None:
            continue

        model_input, strip = _digitize(
            image_path,
            category,
            layout_hint=layout,
            timeout_seconds=timeout_seconds,
            output_dir=output_dir,
        )
        # Both arms derive from the same digitization to isolate the C.1 change.
        baseline_hr = estimate_heart_rate_bpm(model_input) if model_input is not None else None
        fix_hr = estimate_rhythm_hr(strip) if strip is not None else None

        records.append(
            {
                "category": category,
                "image_id": stem,
                "layout": layout,
                "reference_hr": round(ref_hr, 1),
                "baseline_hr": round(baseline_hr, 1) if baseline_hr else None,
                "rhythm_fix_hr": round(fix_hr, 1) if fix_hr else None,
                "baseline_abs_err": (
                    round(abs(baseline_hr - ref_hr), 1) if baseline_hr else None
                ),
                "rhythm_fix_abs_err": (
                    round(abs(fix_hr - ref_hr), 1) if fix_hr else None
                ),
            }
        )

    rhythms.close()

    baseline_errors = [r["baseline_abs_err"] for r in records if r["baseline_abs_err"] is not None]
    fix_errors = [r["rhythm_fix_abs_err"] for r in records if r["rhythm_fix_abs_err"] is not None]
    return {
        "criterion": {
            "median_abs_err_bpm": "< 5",
            "large_error_rate": "< 0.10",
            "large_error_threshold_bpm": LARGE_ERROR_BPM,
        },
        "baseline_tiled": _error_stats(baseline_errors),
        "rhythm_fix": _error_stats(fix_errors),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate(
        args.data_dir,
        args.output_dir,
        timeout_seconds=args.timeout,
        max_images=args.max_images,
    )
    report_path = args.output_dir / "hr_accuracy.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\nReport: {report_path}")
    print(f"Baseline (tiled):  {report['baseline_tiled']}")
    print(f"Rhythm-strip fix:  {report['rhythm_fix']}")


if __name__ == "__main__":
    main()
