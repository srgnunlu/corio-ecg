"""Tune-only pilot: does image-space re-projection fidelity flag the failures?

Every internal-consistency probe (stability, perturbation, Goldberger) failed
because a digitization can be self-consistent yet wrong. This pilot tests the
one reference-free *fidelity* signal: how well the reconstructed traces overlay
the ink the segmentation model detected on the page.

For each selected PMcardio tune image we digitize once and score the overlay.
The question is whether the two known false-accept records (doogee img_74,
iphone img_26) show markedly lower overlay fidelity (higher residual) than
genuinely high-fidelity in-scope controls.

Research only. Operates exclusively on the tune split; the locked 60-group test
split must stay closed. Requires the Open-ECG-Digitizer vendor patch that
exposes the segmentation probability map (see vendor/patches/).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPROJECT_WORKER = PROJECT_ROOT / "scripts" / "reproject_one.py"
DEFAULT_FIDELITY = Path("results/pmcardio-holdout/tune/pmcardio_reference_fidelity.json")
DEFAULT_OUTPUT_DIR = Path("results/pmcardio-holdout/tune/reprojection")

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
        selected.append({**record, "role": "false_accept"})
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


def _reproject(source: Path) -> dict[str, Any]:
    """Run the isolated worker and parse its JSON overlay scores."""
    result = subprocess.run(
        [sys.executable, str(REPROJECT_WORKER), "--image", str(source)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"reproject worker failed: {result.stderr.strip()[-300:]}")
    last_line = result.stdout.strip().splitlines()[-1]
    return dict(json.loads(last_line))


def _score_record(record: dict[str, Any]) -> dict[str, Any]:
    source = PROJECT_ROOT / record["source_image"]
    scores = _reproject(source)
    return {
        "category": record["category"],
        "image_id": int(record["image_id"]),
        "layout": record.get("layout"),
        "role": record["role"],
        "true_median_correlation": _true_fidelity(record),
        "scores": scores,
    }


def _separation(results: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Separation margin for one residual/score metric (higher residual = worse)."""

    def values(role: str) -> list[float]:
        return [float(r["scores"][metric]) for r in results if r["role"] == role]

    false_accept = values("false_accept")
    control = values("control")
    summary: dict[str, Any] = {
        "false_accept_values": false_accept,
        "control_count": len(control),
    }
    if control:
        summary["control_min"] = float(np.min(control))
        summary["control_max"] = float(np.max(control))
        summary["control_median"] = float(np.median(control))
    if false_accept and control:
        # residual: false accepts should exceed controls -> min(FA) - max(ctrl).
        summary["separation_margin_residual"] = float(np.min(false_accept) - np.max(control))
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

    results: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        label = f"{record['category']}/img_{record['image_id']}"
        print(f"[{index}/{len(selected)}] digitizing {label} ({record['role']})", flush=True)
        try:
            results.append(_score_record(record))
        except (RuntimeError, ValueError, json.JSONDecodeError) as error:
            print(f"  skipped {label}: {error}")
            results.append(
                {
                    "category": record["category"],
                    "image_id": int(record["image_id"]),
                    "role": record["role"],
                    "error": str(error),
                }
            )

    scored = [r for r in results if "scores" in r]
    summary = {
        "residual": _separation(scored, "residual"),
        "precision": _separation(scored, "precision"),
        "recall": _separation(scored, "recall"),
    }
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "fidelity_source": str(args.fidelity),
        "metric": "image-space re-projection overlay (precision/recall/f1)",
        "summary": summary,
        "records": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "reprojection_pilot.json"
    output_path.write_text(json.dumps(report, indent=2))

    print("\n=== Re-projection pilot summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
