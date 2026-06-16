"""Tune-only pilot: does perturbation disagreement flag low-fidelity records?

For each selected PMcardio tune image we digitize an unperturbed reference arm
and a perturbed arm, then score them with the reference-free reconstruction
disagreement metric. The question is whether the two known false-accept records
(doogee img_74, iphone img_26) show markedly higher worst-case disagreement
than genuinely high-fidelity in-scope controls.

Research only. Operates exclusively on the tune split; the locked 60-group test
split must stay closed. See docs/experiments/phase3-reconstruction-disagreement-v1.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.perturbation_harness import (  # noqa: E402
    PerturbationSpec,
    apply_perturbation,
    default_perturbation_set,
)
from src.evaluation.reconstruction_disagreement import (  # noqa: E402
    evaluate_reconstruction_pair,
)

# Identity arm: a PNG round-trip with no geometric change. It shares the
# perturbed arm's decode-and-re-encode path, so any residual disagreement is
# the metric's noise floor rather than a JPEG-vs-PNG artefact.
IDENTITY_SPEC = PerturbationSpec(label="identity")
DIGITIZE_WORKER = PROJECT_ROOT / "scripts" / "digitize_one.py"

DEFAULT_FIDELITY = Path("results/pmcardio-holdout/tune/pmcardio_reference_fidelity.json")
DEFAULT_OUTPUT_DIR = Path("results/pmcardio-holdout/tune/reconstruction-disagreement")

# The two layout-in-scope records the frozen gate wrongly accepted.
KNOWN_FALSE_ACCEPTS: tuple[tuple[str, int], ...] = (
    ("photos_doogee", 74),
    ("photos_iphone", 26),
)
# Controls are drawn from the same layout family so the comparison is fair.
CONTROL_LAYOUTS: tuple[str, ...] = ("3x4+1R", "3x4+3R")


def _true_fidelity(record: dict[str, Any]) -> float | None:
    fidelity = record.get("fidelity") or {}
    value = fidelity.get("median_correlation")
    return float(value) if value is not None else None


def _select_records(records: list[dict[str, Any]], control_count: int) -> list[dict[str, Any]]:
    """Pick the two false accepts plus the highest-fidelity in-scope controls."""
    by_key = {(r["category"], int(r["image_id"])): r for r in records}
    selected: list[dict[str, Any]] = []
    chosen_keys: set[tuple[str, int]] = set()
    for key in KNOWN_FALSE_ACCEPTS:
        record = by_key.get(key)
        if record is None:
            raise ValueError(f"false-accept record {key} not found in tune fidelity")
        record = {**record, "role": "false_accept"}
        selected.append(record)
        chosen_keys.add(key)

    controls = [
        r
        for r in records
        if (r["category"], int(r["image_id"])) not in chosen_keys
        and r.get("status") == "success"
        and r.get("layout") in CONTROL_LAYOUTS
        and _true_fidelity(r) is not None
    ]
    controls.sort(key=lambda r: _true_fidelity(r) or 0.0, reverse=True)
    for record in controls[:control_count]:
        selected.append({**record, "role": "control"})
    return selected


def _digitize(image_array: np.ndarray, work_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Digitize an RGB array in an isolated subprocess so the OS reclaims memory.

    Returns the model-input and calibrated mV signals. Raises RuntimeError if
    the worker fails (e.g. no signal detected).
    """
    image_path = work_dir / f"{name}.png"
    model_path = work_dir / f"{name}_model.npy"
    calibrated_path = work_dir / f"{name}_calibrated.npy"
    Image.fromarray(image_array).save(image_path)
    # A single no-retry pass keeps memory bounded (the dewarping retry balloons
    # the CPU working set past 100 GB) and is deterministic — both arms share
    # the same setting, so the comparison stays fair.
    result = subprocess.run(
        [
            sys.executable,
            str(DIGITIZE_WORKER),
            "--no-retries",
            "--image",
            str(image_path),
            "--model-output",
            str(model_path),
            "--calibrated-output",
            str(calibrated_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not model_path.exists():
        raise RuntimeError(
            f"digitize worker failed for {name}: {result.stderr.strip()[-300:]}"
        )
    return np.load(model_path), np.load(calibrated_path)


def _worst_case(disagreements: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse per-perturbation scores into the least stable observation."""
    median_corrs = [d["median_correlation"] for d in disagreements]
    min_corrs = [d["minimum_correlation"] for d in disagreements]
    unstable = [d["unstable_leads_below_0_8"] for d in disagreements]
    worst_index = int(np.argmin(median_corrs))
    summary = {
        "worst_perturbation": disagreements[worst_index]["perturbation"],
        "min_median_correlation": float(np.min(median_corrs)),
        "min_minimum_correlation": float(np.min(min_corrs)),
        "max_unstable_leads_below_0_8": int(np.max(unstable)),
    }
    calibrated_rmse = [
        d["calibrated"]["median_rmse_mv"]
        for d in disagreements
        if d.get("calibrated") is not None
    ]
    if calibrated_rmse:
        summary["max_calibrated_median_rmse_mv"] = float(np.max(calibrated_rmse))
    return summary


def _score_record(
    record: dict[str, Any],
    perturbations: tuple[PerturbationSpec, ...],
    work_dir: Path,
) -> dict[str, Any]:
    source = PROJECT_ROOT / record["source_image"]
    image_array = np.asarray(Image.open(source).convert("RGB"))
    reference_model, reference_calibrated = _digitize(
        apply_perturbation(image_array, IDENTITY_SPEC), work_dir, "reference"
    )

    disagreements: list[dict[str, Any]] = []
    for spec in perturbations:
        perturbed_array = apply_perturbation(image_array, spec)
        perturbed_model, perturbed_calibrated = _digitize(perturbed_array, work_dir, spec.label)
        scores = evaluate_reconstruction_pair(
            reference_model,
            perturbed_model,
            first_calibrated=reference_calibrated,
            second_calibrated=perturbed_calibrated,
        )
        disagreements.append(
            {
                "perturbation": spec.label,
                "median_correlation": scores["median_correlation"],
                "minimum_correlation": scores["minimum_correlation"],
                "median_normalized_rmse": scores["median_normalized_rmse"],
                "unstable_leads_below_0_8": scores["unstable_leads_below_0_8"],
                "calibrated": scores.get("calibrated"),
            }
        )

    return {
        "category": record["category"],
        "image_id": int(record["image_id"]),
        "layout": record.get("layout"),
        "role": record["role"],
        "true_median_correlation": _true_fidelity(record),
        "per_perturbation": disagreements,
        "worst_case": _worst_case(disagreements),
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def stable_metric(role: str) -> list[float]:
        return [
            r["worst_case"]["min_median_correlation"]
            for r in results
            if r["role"] == role
        ]

    false_accept = stable_metric("false_accept")
    control = stable_metric("control")
    summary: dict[str, Any] = {
        "false_accept_min_median_correlation": false_accept,
        "control_worst_case_count": len(control),
    }
    if control:
        summary["control_min_median_correlation_min"] = float(np.min(control))
        summary["control_min_median_correlation_median"] = float(np.median(control))
    if false_accept and control:
        # Separation: lowest control stability minus highest false-accept stability.
        summary["separation_margin"] = float(np.min(control) - np.max(false_accept))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity", type=Path, default=DEFAULT_FIDELITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--control-count", type=int, default=10)
    args = parser.parse_args()

    payload = json.loads(args.fidelity.read_text())
    records = [r for r in payload["records"] if r.get("split") == "tune"]
    if not records:
        raise SystemExit("no tune records found in fidelity report")

    selected = _select_records(records, args.control_count)
    perturbations = default_perturbation_set()

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="perturbation_pilot_") as tmp:
        work_dir = Path(tmp)
        for index, record in enumerate(selected, start=1):
            label = f"{record['category']}/img_{record['image_id']}"
            print(f"[{index}/{len(selected)}] digitizing {label} ({record['role']})", flush=True)
            try:
                results.append(_score_record(record, perturbations, work_dir))
            except (RuntimeError, ValueError) as error:
                print(f"  skipped {label}: {error}")
                results.append(
                    {
                        "category": record["category"],
                        "image_id": int(record["image_id"]),
                        "role": record["role"],
                        "error": str(error),
                    }
                )

    summary = _summarize([r for r in results if "worst_case" in r])
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "fidelity_source": str(args.fidelity),
        "perturbations": [spec.label for spec in perturbations],
        "summary": summary,
        "records": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "perturbation_pilot.json"
    output_path.write_text(json.dumps(report, indent=2))

    print("\n=== Perturbation pilot summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
