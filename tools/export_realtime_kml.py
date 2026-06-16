"""Export realtime localization outputs to Google Earth KML.

Layers:
- calculated drone path from query SRT/manifest
- calculated look-at path from query SRT/manifest
- estimated drone path from matched reference drone positions
- estimated look-at path from realtime predictions
- per-frame connecting lines from true look-at to estimated look-at
"""
from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_manifest_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        p = Path(value)
        return p.stem, p
    ds, path = value.split("=", 1)
    return ds, Path(path)


def safe_float(v: Any) -> float | None:
    try:
        if v in (None, "", "nan"):
            return None
        x = float(v)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def coord(lon: float, lat: float, alt: float = 0) -> str:
    return f"{lon:.8f},{lat:.8f},{alt:.2f}"


def line_string(name: str, points: list[tuple[float, float]], style: str) -> str:
    if len(points) < 2:
        return ""
    coords = "\n".join(coord(lon, lat) for lat, lon in points)
    return f"""
    <Placemark>
      <name>{html.escape(name)}</name>
      <styleUrl>#{style}</styleUrl>
      <LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString>
    </Placemark>
"""


def point_mark(name: str, lat: float, lon: float, style: str, description: str = "") -> str:
    return f"""
    <Placemark>
      <name>{html.escape(name)}</name>
      <styleUrl>#{style}</styleUrl>
      <description>{html.escape(description)}</description>
      <Point><coordinates>{coord(lon, lat)}</coordinates></Point>
    </Placemark>
"""


def folder(name: str, content: str, visible: int = 1) -> str:
    return f"""
  <Folder>
    <name>{html.escape(name)}</name>
    <visibility>{visible}</visibility>
    {content}
  </Folder>
"""


def load_reference_lookup(values: list[str]) -> dict[tuple[str, int], dict[str, str]]:
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    for value in values:
        ds, path = parse_manifest_arg(value)
        for row in read_csv(path):
            try:
                frame = int(float(row["frame_count"]))
            except Exception:
                continue
            row = dict(row)
            row["dataset_id"] = ds
            lookup[(ds, frame)] = row
    return lookup


def true_rows_by_time(truth_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(truth_rows, key=lambda r: safe_float(r.get("start_seconds")) or 0.0)


def nearest_truth(truth_rows: list[dict[str, str]], time_s: float) -> dict[str, str] | None:
    if not truth_rows:
        return None
    return min(truth_rows, key=lambda r: abs((safe_float(r.get("start_seconds")) or 0.0) - time_s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--truth-manifest", type=Path, required=True)
    ap.add_argument("--reference-manifest", action="append", default=[])
    ap.add_argument("--output-kml", type=Path, required=True)
    ap.add_argument("--line-every", type=int, default=1)
    ap.add_argument("--point-every", type=int, default=20)
    args = ap.parse_args()

    preds = read_csv(args.predictions)
    truth = true_rows_by_time(read_csv(args.truth_manifest))
    ref_lookup = load_reference_lookup(args.reference_manifest)

    true_drone_pts: list[tuple[float, float]] = []
    true_look_pts: list[tuple[float, float]] = []
    for row in truth:
        dlat, dlon = safe_float(row.get("drone_latitude")), safe_float(row.get("drone_longitude"))
        glat, glon = safe_float(row.get("ground_latitude")), safe_float(row.get("ground_longitude"))
        if dlat is not None and dlon is not None:
            true_drone_pts.append((dlat, dlon))
        if glat is not None and glon is not None:
            true_look_pts.append((glat, glon))

    est_drone_pts: list[tuple[float, float]] = []
    est_look_pts: list[tuple[float, float]] = []
    connect = ""
    est_points = ""
    for i, row in enumerate(preds):
        eglat, eglon = safe_float(row.get("estimated_ground_latitude")), safe_float(row.get("estimated_ground_longitude"))
        edlat, edlon = safe_float(row.get("estimated_drone_latitude")), safe_float(row.get("estimated_drone_longitude"))
        if edlat is None or edlon is None:
            ds = str(row.get("reference_dataset", ""))
            try:
                frame = int(float(row.get("reference_frame_count", "nan")))
            except Exception:
                frame = -1
            ref = ref_lookup.get((ds, frame))
            if ref is not None:
                edlat, edlon = safe_float(ref.get("drone_latitude")), safe_float(ref.get("drone_longitude"))
        if eglat is not None and eglon is not None:
            est_look_pts.append((eglat, eglon))
        if edlat is not None and edlon is not None:
            est_drone_pts.append((edlat, edlon))

        if i % max(args.line_every, 1) == 0 and eglat is not None and eglon is not None:
            t = nearest_truth(truth, safe_float(row.get("query_video_time_s")) or float(i))
            if t is not None:
                tglat, tglon = safe_float(t.get("ground_latitude")), safe_float(t.get("ground_longitude"))
                if tglat is not None and tglon is not None:
                    desc = f"frame={i}, ref={row.get('reference_dataset')}:{row.get('reference_frame_count')}, error_m={row.get('position_error_m','')}"
                    connect += f"""
    <Placemark>
      <name>match_{i:04d}</name>
      <styleUrl>#connectionStyle</styleUrl>
      <description>{html.escape(desc)}</description>
      <LineString><tessellate>1</tessellate><coordinates>{coord(tglon, tglat)} {coord(eglon, eglat)}</coordinates></LineString>
    </Placemark>
"""
        if i % max(args.point_every, 1) == 0 and eglat is not None and eglon is not None:
            desc = f"time={row.get('query_video_time_s')}, ref={row.get('reference_dataset')}:{row.get('reference_frame_count')}, error_m={row.get('position_error_m','')}"
            est_points += point_mark(f"estimated_look_{i:04d}", eglat, eglon, "estLookPointStyle", desc)

    styles = """
  <Style id="trueDroneStyle"><LineStyle><color>ffff0000</color><width>4</width></LineStyle></Style>
  <Style id="trueLookStyle"><LineStyle><color>ff00aa00</color><width>4</width></LineStyle></Style>
  <Style id="estDroneStyle"><LineStyle><color>ff00ffff</color><width>4</width></LineStyle></Style>
  <Style id="estLookStyle"><LineStyle><color>ff0000ff</color><width>4</width></LineStyle></Style>
  <Style id="connectionStyle"><LineStyle><color>660000ff</color><width>1</width></LineStyle></Style>
  <Style id="estLookPointStyle"><IconStyle><scale>0.6</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon></IconStyle></Style>
"""
    body = styles
    body += folder("calculated_drone_path_from_srt", line_string("calculated_drone_path", true_drone_pts, "trueDroneStyle"))
    body += folder("calculated_look_at_path_from_srt", line_string("calculated_look_at_path", true_look_pts, "trueLookStyle"))
    body += folder("estimated_drone_path_from_realtime_matches", line_string("estimated_drone_path", est_drone_pts, "estDroneStyle"))
    body += folder("estimated_look_at_path_from_realtime", line_string("estimated_look_at_path", est_look_pts, "estLookStyle"))
    body += folder("per_frame_true_to_estimated_look_at_lines", connect, visible=0)
    body += folder("estimated_look_at_sample_points", est_points, visible=0)

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>{html.escape(args.output_kml.stem)}</name>
{body}
</Document>
</kml>
"""
    args.output_kml.parent.mkdir(parents=True, exist_ok=True)
    args.output_kml.write_text(kml, encoding="utf-8")
    print(f"wrote: {args.output_kml}")
    print(f"predictions: {len(preds)}")
    print(f"truth rows: {len(truth)}")


if __name__ == "__main__":
    main()
