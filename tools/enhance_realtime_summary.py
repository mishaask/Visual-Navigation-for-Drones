from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def safe_float(value: Any) -> float:
    try:
        if value in (None, "", "nan", "None"):
            return math.nan
        x = float(value)
        return x if math.isfinite(x) else math.nan
    except Exception:
        return math.nan


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def is_valid_estimate(row: dict[str, str]) -> bool:
    v = str(row.get("valid_estimate", "1")).strip().lower()
    if v in ("0", "false", "no", "none", ""):
        return False
    return math.isfinite(safe_float(row.get("estimated_ground_latitude"))) and math.isfinite(
        safe_float(row.get("estimated_ground_longitude"))
    )


def metric_block(values: list[float], prefix: str = "") -> dict[str, Any]:
    vals = [v for v in values if math.isfinite(v)]
    out: dict[str, Any] = {
        f"{prefix}evaluated_frames": len(vals),
    }
    if not vals:
        out.update(
            {
                f"{prefix}mean_error_m": None,
                f"{prefix}median_error_m": None,
                f"{prefix}p90_error_m": None,
                f"{prefix}p95_error_m": None,
                f"{prefix}max_error_m": None,
                f"{prefix}pct_under_100m": None,
                f"{prefix}pct_under_50m": None,
                f"{prefix}pct_under_10m": None,
                f"{prefix}pct_under_5m": None,
            }
        )
        return out

    vals_sorted = sorted(vals)

    def percentile(p: float) -> float:
        if len(vals_sorted) == 1:
            return vals_sorted[0]
        pos = (len(vals_sorted) - 1) * p / 100.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals_sorted[lo]
        frac = pos - lo
        return vals_sorted[lo] * (1 - frac) + vals_sorted[hi] * frac

    def pct_under(threshold: float) -> float:
        # Inclusive threshold: <= threshold metres.
        return 100.0 * sum(1 for v in vals if v <= threshold) / len(vals)

    out.update(
        {
            f"{prefix}mean_error_m": sum(vals) / len(vals),
            f"{prefix}median_error_m": percentile(50),
            f"{prefix}p90_error_m": percentile(90),
            f"{prefix}p95_error_m": percentile(95),
            f"{prefix}max_error_m": max(vals),
            f"{prefix}pct_under_100m": pct_under(100.0),
            f"{prefix}pct_under_50m": pct_under(50.0),
            f"{prefix}pct_under_10m": pct_under(10.0),
            f"{prefix}pct_under_5m": pct_under(5.0),
        }
    )
    return out


def fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def write_markdown(path: Path, summary: dict[str, Any], test_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_truth = bool(summary.get("has_ground_truth"))

    lines: list[str] = []
    lines.append(f"# Realtime V7 Summary — {test_name}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Frames processed: `{summary.get('frames_processed', 0)}`")
    lines.append(f"- Valid estimates: `{summary.get('valid_estimate_frames', 0)}`")
    lines.append(f"- No-estimate frames: `{summary.get('no_estimate_frames', 0)}`")
    lines.append(f"- Coverage: `{fmt(summary.get('coverage_pct'), '%')}`")
    lines.append("")

    if has_truth:
        lines.append("## Look-at / camera-center error")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        lines.append(f"| Mean error | {fmt(summary.get('mean_error_m'), ' m')} |")
        lines.append(f"| Median error | {fmt(summary.get('median_error_m'), ' m')} |")
        lines.append(f"| P90 error | {fmt(summary.get('p90_error_m'), ' m')} |")
        lines.append(f"| P95 error | {fmt(summary.get('p95_error_m'), ' m')} |")
        lines.append(f"| Max error | {fmt(summary.get('max_error_m'), ' m')} |")
        lines.append(f"| % under 100 m | {fmt(summary.get('pct_under_100m'), '%')} |")
        lines.append(f"| % under 50 m | {fmt(summary.get('pct_under_50m'), '%')} |")
        lines.append(f"| % under 10 m | {fmt(summary.get('pct_under_10m'), '%')} |")
        lines.append(f"| % under 5 m | {fmt(summary.get('pct_under_5m'), '%')} |")
        lines.append("")

        if summary.get("drone_evaluated_frames", 0):
            lines.append("## Drone-position error")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| --- | ---: |")
            lines.append(f"| Mean drone error | {fmt(summary.get('drone_mean_error_m'), ' m')} |")
            lines.append(f"| Median drone error | {fmt(summary.get('drone_median_error_m'), ' m')} |")
            lines.append(f"| P90 drone error | {fmt(summary.get('drone_p90_error_m'), ' m')} |")
            lines.append(f"| P95 drone error | {fmt(summary.get('drone_p95_error_m'), ' m')} |")
            lines.append(f"| Max drone error | {fmt(summary.get('drone_max_error_m'), ' m')} |")
            lines.append(f"| Drone % under 100 m | {fmt(summary.get('drone_pct_under_100m'), '%')} |")
            lines.append(f"| Drone % under 50 m | {fmt(summary.get('drone_pct_under_50m'), '%')} |")
            lines.append(f"| Drone % under 10 m | {fmt(summary.get('drone_pct_under_10m'), '%')} |")
            lines.append(f"| Drone % under 5 m | {fmt(summary.get('drone_pct_under_5m'), '%')} |")
            lines.append("")
    else:
        lines.append("## Accuracy")
        lines.append("")
        lines.append("No ground-truth SRT/manifest was supplied for this query, so error statistics cannot be computed.")
        lines.append("The output contains estimated paths only.")
        lines.append("")

    lines.append("## Metric notes")
    lines.append("")
    for note in summary["metric_notes"]:
        lines.append(f"- {note}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Enhance realtime summary with extra threshold percentages and human-readable notes.")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--base-summary", type=Path)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    ap.add_argument("--test-name", required=True)
    args = ap.parse_args()

    rows = read_csv(args.predictions)
    base = load_json(args.base_summary)

    valid_rows = [r for r in rows if is_valid_estimate(r)]
    no_estimate_rows = [r for r in rows if not is_valid_estimate(r)]

    look_errors = [safe_float(r.get("position_error_m")) for r in valid_rows]
    drone_errors = [safe_float(r.get("drone_position_error_m")) for r in valid_rows]

    has_truth = any(math.isfinite(v) for v in look_errors)

    summary: dict[str, Any] = {}
    summary.update(base)
    summary["test_name"] = args.test_name
    summary["frames_processed"] = len(rows)
    summary["valid_estimate_frames"] = len(valid_rows)
    summary["no_estimate_frames"] = len(no_estimate_rows)
    summary["coverage_pct"] = 100.0 * len(valid_rows) / len(rows) if rows else 0.0
    summary["has_ground_truth"] = has_truth

    if has_truth:
        summary.update(metric_block(look_errors, prefix=""))
        summary.update(metric_block(drone_errors, prefix="drone_"))
    else:
        # Keep fields explicit so readers do not confuse missing truth with bad accuracy.
        summary.update(metric_block([], prefix=""))
        summary.update(metric_block([], prefix="drone_"))

    # Put notes at the bottom of the JSON by adding them last.
    summary["metric_notes"] = [
        "Valid estimate frames are frames where the realtime localizer output an estimated coordinate. NO_ESTIMATE frames are intentionally skipped because the system was uncertain.",
        "Coverage is valid_estimate_frames divided by frames_processed.",
        "Mean error is the arithmetic average distance between the estimated look-at point and the SRT-derived ground-truth look-at point, in metres.",
        "Median error is the middle error value; half of evaluated estimates are below it and half are above it.",
        "P90 error means 90 percent of evaluated estimates have error less than or equal to this value.",
        "P95 error means 95 percent of evaluated estimates have error less than or equal to this value.",
        "% under 100m / 50m / 10m / 5m is the percentage of evaluated valid estimates whose error is less than or equal to that threshold. It is not divided by all frames, only by frames that have a valid estimate and ground truth.",
        "Drone-position error compares estimated drone GPS position to SRT drone GPS position. Look-at/camera-center error compares estimated camera-center ground coordinate to the SRT-derived camera-center ground coordinate.",
        "When no SRT/ground truth is available, accuracy metrics are N/A and only estimated paths are exported.",
    ]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_md, summary, args.test_name)

    print(json.dumps(summary, indent=2))
    print(f"wrote: {args.output_json}")
    print(f"wrote: {args.output_md}")


if __name__ == "__main__":
    main()
