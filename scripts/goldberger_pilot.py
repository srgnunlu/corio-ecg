"""Tune-only pilot: does limb-lead redundancy flag low-fidelity records?

The shadow and perturbation pilots failed because both probed *stability*, not
*fidelity* — a wrong-but-stable digitization slipped through. This pilot probes
absolute physiological plausibility instead: a faithful 12-lead reconstruction
must satisfy the Einthoven/Goldberger limb-lead derivation identities, while a
distorted or mis-assigned reconstruction violates them no matter how stable it
is.

For each selected PMcardio tune image we digitize once and score the calibrated
signal with the full limb-lead redundancy metric. The question is whether the
two known false-accept records (doogee img_74, iphone img_26) show markedly
higher Goldberger residuals than genuinely high-fidelity in-scope controls.

Research only. Operates exclusively on the tune split; the locked 60-group test
split must stay closed. See docs/experiments/.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.signal_clean import goldberger_consistency  # noqa: E402

DIGITIZE_WORKER = PROJECT_ROOT / "scripts" / "digitize_one.py"
DEFAULT_FIDELITY = Path("results/pmcardio-holdout/tune/pmcardio_reference_fidelity.json")
DEFAULT_OUTPUT_DIR = Path("results/pmcardio-holdout/tune/physiological-consistency")

# The two layout-in-scope records the frozen gate wrongly accepted.
KNOWN_FALSE_ACCEPTS: tuple[tuple[str, int], ...] = (
    ("photos_doogee", 74),
    ("photos_iphone", 26),
)
# Controls are drawn from the same layout family so the comparison is fair.
CONTROL_LAYOUTS: tuple[str, ...] = ("3x4+1R", "3x4+3R")

# Lag-tolerant (handles cross-column phase) and strict zero-lag variants. The
# strict variant is reported only to show why phase tolerance is required.
LAG_VARIANTS: tuple[tuple[str, float], ...] = (
    ("lag_tolerant", 1.3),
    ("zero_lag", 0.0),
)


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


def _digitize(source: Path, work_dir: Path) -> np.ndarray:
    """Digitize one image in an isolated subprocess; return calibrated mV signal.

    The CPU digitizer does not return its multi-GB working set between calls, so
    each image runs in its own worker. Retries are disabled to keep the footprint
    bounded (the dewarping retry balloons RSS past 100 GB).
    """
    model_path = work_dir / "model.npy"
    calibrated_path = work_dir / "calibrated.npy"
    result = subprocess.run(
        [
            sys.executable,
            str(DIGITIZE_WORKER),
            "--no-retries",
            "--image",
            str(source),
            "--model-output",
            str(model_path),
            "--calibrated-output",
            str(calibrated_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not calibrated_path.exists():
        raise RuntimeError(f"digitize worker failed: {result.stderr.strip()[-300:]}")
    return np.asarray(np.load(calibrated_path))


def _score_record(record: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    source = PROJECT_ROOT / record["source_image"]
    signal = _digitize(source, work_dir)

    variants: dict[str, Any] = {}
    for name, max_lag_seconds in LAG_VARIANTS:
        consistency = goldberger_consistency(signal, max_lag_seconds=max_lag_seconds)
        variants[name] = {
            "correlations": consistency.correlations,
            "residuals": consistency.residuals,
            "mean_residual": consistency.mean_residual,
            "worst_residual": consistency.worst_residual,
        }

    return {
        "category": record["category"],
        "image_id": int(record["image_id"]),
        "layout": record.get("layout"),
        "role": record["role"],
        "true_median_correlation": _true_fidelity(record),
        "variants": variants,
    }


def _separation(results: list[dict[str, Any]], variant: str, metric: str) -> dict[str, Any]:
    """Separation margin for one (lag variant, residual metric) pair.

    Higher residual = worse, so a clean separation means every false accept sits
    above every control. The margin is ``min(false_accept) - max(control)``;
    positive means a threshold cleanly isolates the failures.
    """

    def values(role: str) -> list[float]:
        return [
            float(r["variants"][variant][metric]) for r in results if r["role"] == role
        ]

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
        summary["separation_margin"] = float(np.min(false_accept) - np.max(control))
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
    with tempfile.TemporaryDirectory(prefix="goldberger_pilot_") as tmp:
        for index, record in enumerate(selected, start=1):
            label = f"{record['category']}/img_{record['image_id']}"
            print(f"[{index}/{len(selected)}] digitizing {label} ({record['role']})", flush=True)
            record_dir = Path(tmp) / f"rec_{index}"
            record_dir.mkdir()
            try:
                results.append(_score_record(record, record_dir))
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

    scored = [r for r in results if "variants" in r]
    summary = {
        variant: {
            "mean_residual": _separation(scored, variant, "mean_residual"),
            "worst_residual": _separation(scored, variant, "worst_residual"),
        }
        for variant, _ in LAG_VARIANTS
    }
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "fidelity_source": str(args.fidelity),
        "lag_variants": {name: seconds for name, seconds in LAG_VARIANTS},
        "rules": ["II = I+III", "aVR = -(I+II)/2", "aVL = (I-III)/2", "aVF = (II+III)/2"],
        "summary": summary,
        "records": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "goldberger_pilot.json"
    output_path.write_text(json.dumps(report, indent=2))

    print("\n=== Goldberger pilot summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
