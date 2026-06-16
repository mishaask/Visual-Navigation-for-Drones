"""Inspect whether a DJI SRT contains direct yaw/gimbal direction fields.

If no direct yaw/gimbal-yaw field exists, this computes course-over-ground from
successive GPS samples. That is flight direction, not guaranteed camera look direction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

TIME_RE = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s+-->\s+(\d\d):(\d\d):(\d\d),(\d\d\d)")
BRACKET_RE = re.compile(r"\[([^\]]+)\]")
YAW_KEYS = ("yaw", "gimbal_yaw", "gb_yaw", "gimbal yaw", "flight_yaw", "attitude_yaw")


def ts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def safe_float(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def parse_srt(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text)
    rows = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        time_line = next((ln for ln in lines if "-->" in ln), "")
        m = TIME_RE.search(time_line)
        if not m:
            continue
        row: dict[str, Any] = {"start_seconds": ts_to_sec(*m.groups()[:4])}
        joined = " ".join(lines)
        # frame count
        fm = re.search(r"(?:FrameCnt|SrtCnt)\s*:?\s*(\d+)", joined)
        if fm:
            row["frame_count"] = int(fm.group(1))
        for item in BRACKET_RE.findall(joined):
            # split multiple key-value pairs inside the same bracket, e.g. rel_alt + abs_alt
            for key, val in re.findall(r"([A-Za-z_ ]+)\s*:\s*(-?\d+(?:\.\d+)?)", item):
                row[key.strip().lower().replace(" ", "_")] = float(val)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--srt", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    args = ap.parse_args()
    rows = parse_srt(args.srt)
    present_yaw_keys = sorted({k for r in rows for k in r.keys() if any(y in k for y in YAW_KEYS)})
    out = []
    prev_good = None
    for r in rows:
        lat = safe_float(r.get("latitude"))
        lon = safe_float(r.get("longitude"))
        if lat is None or lon is None or abs(lat) < 1e-9 or abs(lon) < 1e-9:
            continue
        course = None
        if prev_good is not None:
            plat, plon = prev_good["latitude"], prev_good["longitude"]
            # skip identical coordinates
            if abs(lat - plat) > 1e-9 or abs(lon - plon) > 1e-9:
                course = bearing_deg(plat, plon, lat, lon)
        row = {
            "start_seconds": r.get("start_seconds", ""),
            "frame_count": r.get("frame_count", ""),
            "latitude": lat,
            "longitude": lon,
            "rel_alt": r.get("rel_alt", ""),
            "course_over_ground_deg": "" if course is None else course,
        }
        for k in present_yaw_keys:
            row[k] = r.get(k, "")
        out.append(row)
        prev_good = {"latitude": lat, "longitude": lon}
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if out:
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            writer.writeheader()
            writer.writerows(out)
    summary = {
        "srt": str(args.srt),
        "rows_total": len(rows),
        "valid_gps_rows": len(out),
        "direct_yaw_or_gimbal_yaw_fields_found": present_yaw_keys,
        "has_direct_camera_look_direction": bool(present_yaw_keys),
        "fallback_available": "course_over_ground_deg from GPS deltas",
        "warning": "course_over_ground is drone movement direction, not guaranteed camera/gimbal look direction",
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote: {args.output_csv}")
    print(f"wrote: {args.summary_json}")


if __name__ == "__main__":
    main()
