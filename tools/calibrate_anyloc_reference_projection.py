import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

EARTH_R = 6378137.0


def parse_manifest_arg(value):
    if "=" in value:
        dataset, path = value.split("=", 1)
        return dataset, Path(path)
    path = Path(value)
    return path.stem, path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def first_float(row, keys, default=None):
    for k in keys:
        if k in row:
            v = safe_float(row.get(k), None)
            if v is not None:
                return v
    return default


def normalize_path(path_text):
    return str(Path(str(path_text).replace("\\", "/")).as_posix()).lower()


def pick_prefix(fieldnames):
    if "motion_viterbi_reference_dataset" in fieldnames:
        return "motion_viterbi"
    if "temporal_reference_dataset" in fieldnames:
        return "temporal"
    if "dino_reference_dataset" in fieldnames:
        return "dino"
    raise RuntimeError("Could not detect result type: expected motion_viterbi, temporal, or dino columns")


def distance_m(lat1, lon1, lat2, lon2):
    lat0 = math.radians((lat1 + lat2) * 0.5)
    dx = math.radians(lon2 - lon1) * math.cos(lat0) * EARTH_R
    dy = math.radians(lat2 - lat1) * EARTH_R
    return math.hypot(dx, dy)


def offset_latlon(lat, lon, east_m, north_m):
    lat2 = lat + math.degrees(north_m / EARTH_R)
    lon2 = lon + math.degrees(east_m / (EARTH_R * math.cos(math.radians(lat))))
    return lat2, lon2


def project_camera_center(drone_lat, drone_lon, heading_deg, alt_m, angle_deg, bearing_offset_deg, convention):
    alt_m = abs(float(alt_m))
    angle_rad = math.radians(max(0.01, min(89.99, float(angle_deg))))
    if convention == "from-horizon":
        ground_m = alt_m / math.tan(angle_rad)
    elif convention == "from-nadir":
        ground_m = alt_m * math.tan(angle_rad)
    else:
        raise ValueError(f"Unsupported angle convention: {convention}")
    bearing = math.radians((heading_deg + bearing_offset_deg) % 360.0)
    east = math.sin(bearing) * ground_m
    north = math.cos(bearing) * ground_m
    return offset_latlon(drone_lat, drone_lon, east, north)


def parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def frange(start, stop, step):
    vals = []
    x = float(start)
    stop = float(stop)
    step = float(step)
    if step <= 0:
        raise ValueError("step must be positive")
    while x <= stop + 1e-9:
        vals.append(round(x, 10))
        x += step
    return vals


def percentile(values, p):
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def summarize(errors):
    return {
        "count": len(errors),
        "mean_m": mean(errors),
        "median_m": median(errors),
        "rmse_m": math.sqrt(mean([e * e for e in errors])),
        "min_m": min(errors),
        "max_m": max(errors),
        "p75_m": percentile(errors, 75),
        "p90_m": percentile(errors, 90),
        "p95_m": percentile(errors, 95),
        "pct_under_10m": 100.0 * sum(e < 10 for e in errors) / len(errors),
        "pct_under_25m": 100.0 * sum(e < 25 for e in errors) / len(errors),
        "pct_under_50m": 100.0 * sum(e < 50 for e in errors) / len(errors),
        "pct_under_100m": 100.0 * sum(e < 100 for e in errors) / len(errors),
    }


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def enrich_reference_rows(rows):
    # Adds flexible internal fields: __drone_lat, __drone_lon, __alt_m, __heading_deg.
    out = []
    for r in rows:
        rr = dict(r)
        rr["__drone_lat"] = first_float(rr, ["latitude", "drone_latitude", "reference_drone_latitude", "matched_reference_drone_latitude"])
        rr["__drone_lon"] = first_float(rr, ["longitude", "drone_longitude", "reference_drone_longitude", "matched_reference_drone_longitude"])
        rr["__alt_m"] = first_float(rr, ["rel_alt_m", "relative_altitude_m", "altitude_m", "reference_rel_alt_m", "matched_reference_rel_alt_m"])
        rr["__heading_deg"] = first_float(rr, ["heading_deg", "heading", "course_deg", "reference_heading_deg", "matched_reference_heading_deg"])
        out.append(rr)

    # Fallback heading from neighboring drone positions if heading is absent.
    for i, rr in enumerate(out):
        if rr.get("__heading_deg") is not None:
            continue
        lat = rr.get("__drone_lat")
        lon = rr.get("__drone_lon")
        if lat is None or lon is None:
            continue
        candidates = []
        for j in (i + 1, i - 1, i + 2, i - 2):
            if 0 <= j < len(out):
                lat2 = out[j].get("__drone_lat")
                lon2 = out[j].get("__drone_lon")
                if lat2 is not None and lon2 is not None and distance_m(lat, lon, lat2, lon2) > 1.0:
                    if j > i:
                        candidates.append(bearing_deg(lat, lon, lat2, lon2))
                    else:
                        candidates.append(bearing_deg(lat2, lon2, lat))
        if candidates:
            rr["__heading_deg"] = candidates[0]
    return out


def build_reference_index(ref_args):
    index = {}
    manifests = {}
    for arg in ref_args:
        dataset, path = parse_manifest_arg(arg)
        rows = enrich_reference_rows(read_csv(path))
        manifests[dataset] = rows
        for r in rows:
            frame = str(r.get("frame_count", ""))
            if frame:
                index[(dataset, frame)] = r
    return index, manifests


def rolling_smooth(coords, window, method):
    if window <= 1:
        return coords[:]
    if window % 2 == 0:
        window += 1
    half = window // 2
    lat0 = coords[0][0]
    lon0 = coords[0][1]
    xs = []
    ys = []
    for lat, lon in coords:
        x = math.radians(lon - lon0) * math.cos(math.radians(lat0)) * EARTH_R
        y = math.radians(lat - lat0) * EARTH_R
        xs.append(x)
        ys.append(y)
    out = []
    for i in range(len(coords)):
        lo = max(0, i - half)
        hi = min(len(coords), i + half + 1)
        if method == "mean":
            x = mean(xs[lo:hi])
            y = mean(ys[lo:hi])
        elif method == "median":
            x = median(xs[lo:hi])
            y = median(ys[lo:hi])
        else:
            raise ValueError(f"Unsupported smoothing method: {method}")
        lat = lat0 + math.degrees(y / EARTH_R)
        lon = lon0 + math.degrees(x / (EARTH_R * math.cos(math.radians(lat0))))
        out.append((lat, lon))
    return out


def eval_calibration(items, angle, offset, convention, fallback_altitude_m):
    errors = []
    for item in items:
        ref = item["ref_row"]
        lat = ref.get("__drone_lat")
        lon = ref.get("__drone_lon")
        heading = ref.get("__heading_deg")
        alt = ref.get("__alt_m")
        if alt is None:
            alt = fallback_altitude_m
        if None in (lat, lon, heading, alt):
            continue
        pred_lat, pred_lon = project_camera_center(lat, lon, heading, alt, angle, offset, convention)
        errors.append(distance_m(pred_lat, pred_lon, item["true_lat"], item["true_lon"]))
    return errors


def best_grid(items, angles, offsets, convention, fallback_altitude_m):
    best = None
    for angle in angles:
        for offset in offsets:
            errors = eval_calibration(items, angle, offset, convention, fallback_altitude_m)
            if not errors:
                continue
            stats = summarize(errors)
            cand = {
                "angle_deg": angle,
                "bearing_offset_deg": offset,
                **stats,
            }
            key = (cand["median_m"], cand["mean_m"], cand["p90_m"])
            if best is None or key < best[0]:
                best = (key, cand)
    return None if best is None else best[1]


def write_kml(path, true_coords, raw_coords, cal_coords, smooth_coords):
    def coord_line(coords):
        return "\n".join(f"{lon:.8f},{lat:.8f},0" for lat, lon in coords)
    smooth_block = ""
    if smooth_coords:
        smooth_block = f"""
  <Placemark>
    <name>Smoothed calibrated estimated path</name>
    <styleUrl>#smoothStyle</styleUrl>
    <LineString><tessellate>1</tessellate><coordinates>
{coord_line(smooth_coords)}
    </coordinates></LineString>
  </Placemark>
"""
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>AnyLoc segment-calibrated path</name>
  <Style id="trueStyle"><LineStyle><color>ff00ff00</color><width>4</width></LineStyle></Style>
  <Style id="rawStyle"><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>
  <Style id="calStyle"><LineStyle><color>ffff0000</color><width>4</width></LineStyle></Style>
  <Style id="smoothStyle"><LineStyle><color>ff00ffff</color><width>5</width></LineStyle></Style>

  <Placemark>
    <name>True camera-center path</name>
    <styleUrl>#trueStyle</styleUrl>
    <LineString><tessellate>1</tessellate><coordinates>
{coord_line(true_coords)}
    </coordinates></LineString>
  </Placemark>

  <Placemark>
    <name>Raw estimated path</name>
    <styleUrl>#rawStyle</styleUrl>
    <LineString><tessellate>1</tessellate><coordinates>
{coord_line(raw_coords)}
    </coordinates></LineString>
  </Placemark>

  <Placemark>
    <name>Segment-calibrated estimated path</name>
    <styleUrl>#calStyle</styleUrl>
    <LineString><tessellate>1</tessellate><coordinates>
{coord_line(cal_coords)}
    </coordinates></LineString>
  </Placemark>
{smooth_block}
</Document>
</kml>
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(kml, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--query-manifest", required=True, type=Path)
    parser.add_argument("--reference-manifest", action="append", required=True)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--calibration-csv", required=True, type=Path)
    parser.add_argument("--output-kml", type=Path)
    parser.add_argument("--angles", default="35,40,45,50,55,60,65,70")
    parser.add_argument("--offset-start", type=float, default=-180.0)
    parser.add_argument("--offset-stop", type=float, default=180.0)
    parser.add_argument("--offset-step", type=float, default=15.0)
    parser.add_argument("--segment-frame-span", type=int, default=3000)
    parser.add_argument("--min-segment-count", type=int, default=8)
    parser.add_argument("--default-angle", type=float, default=60.0)
    parser.add_argument("--default-offset", type=float, default=0.0)
    parser.add_argument("--angle-convention", choices=["from-horizon", "from-nadir"], default="from-horizon")
    parser.add_argument("--fallback-altitude-m", type=float, default=119.5)
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--smooth-method", choices=["median", "mean"], default="median")
    args = parser.parse_args()

    result_rows = read_csv(args.results)
    if not result_rows:
        raise RuntimeError("No result rows")
    prefix = pick_prefix(result_rows[0].keys())

    query_rows = read_csv(args.query_manifest)
    q_by_path = {normalize_path(r.get("frame_path", "")): r for r in query_rows}
    q_by_frame = {str(r.get("frame_count", "")): r for r in query_rows}

    ref_index, _ = build_reference_index(args.reference_manifest)

    items = []
    dropped = 0
    for i, row in enumerate(result_rows):
        qrow = q_by_path.get(normalize_path(row.get("query_frame_path", "")))
        if qrow is None:
            qrow = q_by_frame.get(str(row.get("query_frame_count", "")))
        dataset = str(row.get(f"{prefix}_reference_dataset", ""))
        ref_frame = str(row.get(f"{prefix}_reference_frame_count", ""))
        refrow = ref_index.get((dataset, ref_frame))
        true_lat = safe_float(qrow.get("ground_latitude")) if qrow else None
        true_lon = safe_float(qrow.get("ground_longitude")) if qrow else None
        if qrow is None or refrow is None or true_lat is None or true_lon is None:
            dropped += 1
            continue
        raw_lat = safe_float(refrow.get("ground_latitude"))
        raw_lon = safe_float(refrow.get("ground_longitude"))
        if raw_lat is None or raw_lon is None:
            # Fallback to default projection if stored ground projection is unavailable.
            raw_lat, raw_lon = project_camera_center(
                refrow["__drone_lat"], refrow["__drone_lon"], refrow["__heading_deg"],
                refrow.get("__alt_m") or args.fallback_altitude_m,
                args.default_angle, args.default_offset, args.angle_convention,
            )
        seg_id = safe_int(ref_frame, 0) // args.segment_frame_span
        items.append({
            "i": i,
            "row": row,
            "qrow": qrow,
            "ref_row": refrow,
            "dataset": dataset,
            "ref_frame": ref_frame,
            "segment_id": seg_id,
            "true_lat": true_lat,
            "true_lon": true_lon,
            "raw_lat": raw_lat,
            "raw_lon": raw_lon,
        })

    if not items:
        raise RuntimeError("No joined rows. Check dataset IDs and manifest frame_count values.")

    angles = parse_float_list(args.angles)
    offsets = frange(args.offset_start, args.offset_stop, args.offset_step)

    # Flight-level fallback calibration.
    by_flight = {}
    for it in items:
        by_flight.setdefault(it["dataset"], []).append(it)
    flight_cal = {}
    cal_rows = []
    for dataset, group_items in sorted(by_flight.items()):
        best = best_grid(group_items, angles, offsets, args.angle_convention, args.fallback_altitude_m)
        if best:
            flight_cal[dataset] = best
            cal_rows.append({
                "level": "flight",
                "dataset": dataset,
                "segment_id": "",
                "segment_start": "",
                "segment_end": "",
                "angle_deg": best["angle_deg"],
                "bearing_offset_deg": best["bearing_offset_deg"],
                **{k: best[k] for k in best if k not in ("angle_deg", "bearing_offset_deg")},
            })

    # Segment-level calibration.
    by_segment = {}
    for it in items:
        by_segment.setdefault((it["dataset"], it["segment_id"]), []).append(it)
    segment_cal = {}
    for (dataset, sid), group_items in sorted(by_segment.items()):
        if len(group_items) < args.min_segment_count:
            continue
        best = best_grid(group_items, angles, offsets, args.angle_convention, args.fallback_altitude_m)
        if best:
            segment_cal[(dataset, sid)] = best
            cal_rows.append({
                "level": "segment",
                "dataset": dataset,
                "segment_id": sid,
                "segment_start": sid * args.segment_frame_span,
                "segment_end": (sid + 1) * args.segment_frame_span - 1,
                "angle_deg": best["angle_deg"],
                "bearing_offset_deg": best["bearing_offset_deg"],
                **{k: best[k] for k in best if k not in ("angle_deg", "bearing_offset_deg")},
            })

    raw_errors = []
    cal_errors = []
    true_coords = []
    raw_coords = []
    cal_coords = []
    out_rows = []

    for it in items:
        row = dict(it["row"])
        ref = it["ref_row"]
        dataset = it["dataset"]
        sid = it["segment_id"]
        cal = segment_cal.get((dataset, sid))
        level = "segment"
        if cal is None:
            cal = flight_cal.get(dataset)
            level = "flight"
        if cal is None:
            cal = {"angle_deg": args.default_angle, "bearing_offset_deg": args.default_offset}
            level = "default"
        alt = ref.get("__alt_m")
        if alt is None:
            alt = args.fallback_altitude_m
        cal_lat, cal_lon = project_camera_center(
            ref["__drone_lat"], ref["__drone_lon"], ref["__heading_deg"], alt,
            cal["angle_deg"], cal["bearing_offset_deg"], args.angle_convention,
        )
        raw_err = distance_m(it["raw_lat"], it["raw_lon"], it["true_lat"], it["true_lon"])
        cal_err = distance_m(cal_lat, cal_lon, it["true_lat"], it["true_lon"])
        raw_errors.append(raw_err)
        cal_errors.append(cal_err)
        true_coords.append((it["true_lat"], it["true_lon"]))
        raw_coords.append((it["raw_lat"], it["raw_lon"]))
        cal_coords.append((cal_lat, cal_lon))

        row["true_latitude"] = f"{it['true_lat']:.8f}"
        row["true_longitude"] = f"{it['true_lon']:.8f}"
        row["raw_est_latitude"] = f"{it['raw_lat']:.8f}"
        row["raw_est_longitude"] = f"{it['raw_lon']:.8f}"
        row["raw_error_m"] = f"{raw_err:.3f}"
        row["cal_est_latitude"] = f"{cal_lat:.8f}"
        row["cal_est_longitude"] = f"{cal_lon:.8f}"
        row["cal_error_m"] = f"{cal_err:.3f}"
        row["calibration_level"] = level
        row["calibration_segment_id"] = str(sid)
        row["calibration_angle_deg"] = f"{float(cal['angle_deg']):.6f}"
        row["calibration_bearing_offset_deg"] = f"{float(cal['bearing_offset_deg']):.6f}"
        out_rows.append(row)

    smooth_errors = []
    smooth_coords = []
    if args.smooth_window and args.smooth_window > 1:
        smooth_coords = rolling_smooth(cal_coords, args.smooth_window, args.smooth_method)
        for row, (slat, slon), (tlat, tlon) in zip(out_rows, smooth_coords, true_coords):
            e = distance_m(slat, slon, tlat, tlon)
            smooth_errors.append(e)
            row["smooth_cal_est_latitude"] = f"{slat:.8f}"
            row["smooth_cal_est_longitude"] = f"{slon:.8f}"
            row["smooth_cal_error_m"] = f"{e:.3f}"
            row["smooth_window"] = str(args.smooth_window)
            row["smooth_method"] = args.smooth_method

    summary = {
        "results": str(args.results),
        "prefix": prefix,
        "joined_rows": len(items),
        "dropped_rows": dropped,
        "segment_frame_span": args.segment_frame_span,
        "min_segment_count": args.min_segment_count,
        "angles": angles,
        "offset_start": args.offset_start,
        "offset_stop": args.offset_stop,
        "offset_step": args.offset_step,
        "raw": summarize(raw_errors),
        "calibrated": summarize(cal_errors),
    }
    if smooth_errors:
        summary["calibrated_smoothed"] = summarize(smooth_errors)

    write_csv(args.output_csv, out_rows)
    write_csv(args.calibration_csv, cal_rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_kml:
        write_kml(args.output_kml, true_coords, raw_coords, cal_coords, smooth_coords)

    print(json.dumps(summary, indent=2))
    print(f"wrote: {args.output_csv}")
    print(f"wrote: {args.calibration_csv}")
    print(f"wrote: {args.summary_json}")
    if args.output_kml:
        print(f"wrote: {args.output_kml}")


if __name__ == "__main__":
    main()
