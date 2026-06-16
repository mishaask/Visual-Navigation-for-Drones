from __future__ import annotations

import argparse
import csv
import html
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


def is_valid_estimate(row: dict[str, str]) -> bool:
    v = str(row.get("valid_estimate", "1")).strip().lower()
    if v in ("0", "false", "no", "none", ""):
        return False
    return math.isfinite(safe_float(row.get("estimated_ground_latitude"))) and math.isfinite(
        safe_float(row.get("estimated_ground_longitude"))
    )


def coord(row: dict[str, str], lat_key: str, lon_key: str) -> tuple[float, float] | None:
    lat = safe_float(row.get(lat_key))
    lon = safe_float(row.get(lon_key))
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    return lat, lon


def dedupe_consecutive(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    last: tuple[float, float] | None = None
    for p in points:
        if last is None or abs(p[0] - last[0]) > 1e-12 or abs(p[1] - last[1]) > 1e-12:
            out.append(p)
            last = p
    return out


def truth_points(rows: list[dict[str, str]], lat_key: str, lon_key: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        p = coord(row, lat_key, lon_key)
        if p is None:
            continue
        # Use truth frame/time when available to avoid duplicating same 1fps truth
        # for a 2fps query stream.
        key = str(row.get("truth_frame_count") or row.get("query_video_time_s") or i)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return dedupe_consecutive(out)


def estimate_segments(rows: list[dict[str, str]], lat_key: str, lon_key: str) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    prev_sample: int | None = None

    for row in rows:
        sample_text = row.get("query_sample_index", "")
        try:
            sample = int(float(sample_text))
        except Exception:
            sample = None

        if not is_valid_estimate(row):
            if len(current) >= 2:
                segments.append(current)
            current = []
            prev_sample = None
            continue

        p = coord(row, lat_key, lon_key)
        if p is None:
            if len(current) >= 2:
                segments.append(current)
            current = []
            prev_sample = None
            continue

        # Split estimated paths across NO_ESTIMATE gaps.
        if prev_sample is not None and sample is not None and sample != prev_sample + 1:
            if len(current) >= 2:
                segments.append(current)
            current = []

        current.append(p)
        prev_sample = sample

    if len(current) >= 2:
        segments.append(current)
    return segments


def coordinates_text(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{lon:.10f},{lat:.10f},0" for lat, lon in points)


def line_placemark(name: str, style_id: str, points: list[tuple[float, float]]) -> str:
    if len(points) < 2:
        return ""
    return f"""
    <Placemark>
      <name>{html.escape(name)}</name>
      <styleUrl>#{style_id}</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>{coordinates_text(points)}</coordinates>
      </LineString>
    </Placemark>
"""


def multisegment_placemark(name: str, style_id: str, segments: list[list[tuple[float, float]]]) -> str:
    segments = [s for s in segments if len(s) >= 2]
    if not segments:
        return ""
    if len(segments) == 1:
        return line_placemark(name, style_id, segments[0])
    parts = []
    for seg in segments:
        parts.append(
            f"""
        <LineString>
          <tessellate>1</tessellate>
          <coordinates>{coordinates_text(seg)}</coordinates>
        </LineString>"""
        )
    return f"""
    <Placemark>
      <name>{html.escape(name)}</name>
      <styleUrl>#{style_id}</styleUrl>
      <MultiGeometry>{''.join(parts)}
      </MultiGeometry>
    </Placemark>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Export final clean KML for realtime runs.")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--name", default="Realtime visual navigation")
    args = ap.parse_args()

    rows = read_csv(args.predictions)

    truth_drone = truth_points(rows, "truth_drone_latitude", "truth_drone_longitude")
    truth_look = truth_points(rows, "truth_ground_latitude", "truth_ground_longitude")

    est_drone_segments = estimate_segments(rows, "estimated_drone_latitude", "estimated_drone_longitude")
    est_look_segments = estimate_segments(rows, "estimated_ground_latitude", "estimated_ground_longitude")

    has_truth = len(truth_drone) >= 2 or len(truth_look) >= 2

    placemarks = []
    if has_truth:
        placemarks.append(line_placemark("calculated SRT drone path", "truthDroneGreen", truth_drone))
        placemarks.append(line_placemark("calculated SRT drone look-at path", "truthLookBlue", truth_look))

    placemarks.append(multisegment_placemark("estimated drone path", "estimatedDroneRed", est_drone_segments))
    placemarks.append(multisegment_placemark("estimated look-at path", "estimatedLookYellow", est_look_segments))

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{html.escape(args.name)}</name>

    <!-- KML colors are AABBGGRR -->
    <Style id="truthDroneGreen">
      <LineStyle><color>ff00ff00</color><width>4</width></LineStyle>
    </Style>
    <Style id="truthLookBlue">
      <LineStyle><color>ffff0000</color><width>4</width></LineStyle>
    </Style>
    <Style id="estimatedDroneRed">
      <LineStyle><color>ff0000ff</color><width>4</width></LineStyle>
    </Style>
    <Style id="estimatedLookYellow">
      <LineStyle><color>ff00ffff</color><width>4</width></LineStyle>
    </Style>

    {''.join(placemarks)}
  </Document>
</kml>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(kml, encoding="utf-8")
    print(f"wrote: {args.output}")
    if has_truth:
        print("KML includes SRT drone path, SRT look-at path, estimated drone path, estimated look-at path.")
    else:
        print("KML includes estimated drone path and estimated look-at path only; no SRT truth was available.")


if __name__ == "__main__":
    main()
