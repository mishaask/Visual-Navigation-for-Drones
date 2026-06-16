"""Optical-flow diagnostics for realtime visual-localization predictions.

This does not change the prediction path. It analyzes whether the image-to-image
motion between query frames agrees with the estimated path jumps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

EARTH_RADIUS_M = 6_378_137.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(v: Any, default: float = math.nan) -> float:
    try:
        if v in (None, "", "nan"):
            return default
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def local_xy_from_latlon(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    lat0 = math.radians(origin_lat)
    lon0 = math.radians(origin_lon)
    return ((lon_r - lon0) * math.cos(lat0) * EARTH_RADIUS_M, (lat_r - lat0) * EARTH_RADIUS_M)


def distance_latlon_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    x, y = local_xy_from_latlon(lat2, lon2, lat1, lon1)
    return math.hypot(x, y)


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def load_gray(path: Path, max_width: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"could not read image: {path}")
    if max_width and img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return img


def flow_between(prev_gray: np.ndarray, cur_gray: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=args.max_corners,
        qualityLevel=args.quality_level,
        minDistance=args.min_distance,
        blockSize=args.block_size,
    )
    if p0 is None or len(p0) == 0:
        return {"points_detected": 0, "points_tracked": 0, "flow_quality": 0.0, "median_dx_px": 0.0, "median_dy_px": 0.0, "median_mag_px": 0.0, "mean_mag_px": 0.0, "angle_deg_img": math.nan, "p0": [], "p1": []}
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        cur_gray,
        p0,
        None,
        winSize=(args.lk_win_size, args.lk_win_size),
        maxLevel=args.lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, args.lk_iter, args.lk_eps),
    )
    if p1 is None or st is None:
        return {"points_detected": int(len(p0)), "points_tracked": 0, "flow_quality": 0.0, "median_dx_px": 0.0, "median_dy_px": 0.0, "median_mag_px": 0.0, "mean_mag_px": 0.0, "angle_deg_img": math.nan, "p0": [], "p1": []}
    good0 = p0[st.ravel() == 1].reshape(-1, 2)
    good1 = p1[st.ravel() == 1].reshape(-1, 2)
    if len(good0) == 0:
        return {"points_detected": int(len(p0)), "points_tracked": 0, "flow_quality": 0.0, "median_dx_px": 0.0, "median_dy_px": 0.0, "median_mag_px": 0.0, "mean_mag_px": 0.0, "angle_deg_img": math.nan, "p0": [], "p1": []}
    d = good1 - good0
    dx = d[:, 0]
    dy = d[:, 1]
    mag = np.sqrt(dx * dx + dy * dy)
    med_dx = float(np.median(dx))
    med_dy = float(np.median(dy))
    return {
        "points_detected": int(len(p0)),
        "points_tracked": int(len(good0)),
        "flow_quality": float(len(good0) / max(len(p0), 1)),
        "median_dx_px": med_dx,
        "median_dy_px": med_dy,
        "median_mag_px": float(np.median(mag)),
        "mean_mag_px": float(np.mean(mag)),
        "angle_deg_img": float((math.degrees(math.atan2(med_dy, med_dx)) + 360.0) % 360.0),
        "p0": good0,
        "p1": good1,
    }


def draw_flow(prev_bgr: np.ndarray, cur_bgr: np.ndarray, f: dict[str, Any], row: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    if prev_bgr.shape[:2] != cur_bgr.shape[:2]:
        cur_bgr = cv2.resize(cur_bgr, (prev_bgr.shape[1], prev_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    canvas = cur_bgr.copy()
    p0 = f.get("p0", [])
    p1 = f.get("p1", [])
    if len(p0):
        step = max(1, len(p0) // max(args.max_drawn_tracks, 1))
        for a, b in zip(p0[::step], p1[::step]):
            x0, y0 = int(a[0]), int(a[1])
            x1, y1 = int(b[0]), int(b[1])
            cv2.line(canvas, (x0, y0), (x1, y1), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (x1, y1), 2, (0, 255, 0), -1)
    lines = [
        f"i={row['query_sample_index']} state={row.get('state','')}",
        f"flow med=({row['flow_median_dx_px']:.1f},{row['flow_median_dy_px']:.1f}) mag={row['flow_median_mag_px']:.1f}px q={row['flow_quality']:.2f}",
        f"est_jump={row['estimated_jump_m']:.1f}m truth_jump={row['truth_jump_m']:.1f}m suspicious={row['suspicious_jump']}",
    ]
    y = 24
    for line in lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return canvas


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"pairs": 0}
    def arr(k: str) -> np.ndarray:
        return np.array([float(r[k]) for r in rows if math.isfinite(float(r[k]))], dtype=float)
    est = arr("estimated_jump_m")
    mag = arr("flow_median_mag_px")
    q = arr("flow_quality")
    suspicious = [r for r in rows if str(r.get("suspicious_jump")) == "1"]
    return {
        "pairs": len(rows),
        "mean_estimated_jump_m": float(est.mean()) if len(est) else 0.0,
        "median_estimated_jump_m": float(np.median(est)) if len(est) else 0.0,
        "p90_estimated_jump_m": float(np.percentile(est, 90)) if len(est) else 0.0,
        "max_estimated_jump_m": float(est.max()) if len(est) else 0.0,
        "median_flow_mag_px": float(np.median(mag)) if len(mag) else 0.0,
        "mean_flow_quality": float(q.mean()) if len(q) else 0.0,
        "suspicious_jump_count": len(suspicious),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    ap.add_argument("--output-video", type=Path)
    ap.add_argument("--video-fps", type=float, default=1.0)
    ap.add_argument("--max-width", type=int, default=960)
    ap.add_argument("--max-corners", type=int, default=500)
    ap.add_argument("--quality-level", type=float, default=0.01)
    ap.add_argument("--min-distance", type=int, default=7)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--lk-win-size", type=int, default=21)
    ap.add_argument("--lk-max-level", type=int, default=3)
    ap.add_argument("--lk-iter", type=int, default=30)
    ap.add_argument("--lk-eps", type=float, default=0.01)
    ap.add_argument("--jump-threshold-m", type=float, default=80.0)
    ap.add_argument("--jump-vs-truth-factor", type=float, default=4.0)
    ap.add_argument("--small-flow-px", type=float, default=20.0)
    ap.add_argument("--max-drawn-tracks", type=int, default=120)
    args = ap.parse_args()

    preds = read_csv(args.predictions)
    out_rows: list[dict[str, Any]] = []
    writer = None
    prev_gray = None
    prev_bgr = None
    prev_pred = None

    for row in tqdm(preds, desc="flow diagnostics"):
        frame_path = Path(row["query_frame_path"])
        gray = load_gray(frame_path, args.max_width)
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not read image: {frame_path}")
        if args.max_width and bgr.shape[1] > args.max_width:
            scale = args.max_width / bgr.shape[1]
            bgr = cv2.resize(bgr, (args.max_width, int(round(bgr.shape[0] * scale))), interpolation=cv2.INTER_AREA)

        if prev_gray is None or prev_pred is None:
            prev_gray, prev_bgr, prev_pred = gray, bgr, row
            continue

        f = flow_between(prev_gray, gray, args)
        e1lat, e1lon = safe_float(prev_pred.get("estimated_ground_latitude")), safe_float(prev_pred.get("estimated_ground_longitude"))
        e2lat, e2lon = safe_float(row.get("estimated_ground_latitude")), safe_float(row.get("estimated_ground_longitude"))
        t1lat, t1lon = safe_float(prev_pred.get("truth_ground_latitude")), safe_float(prev_pred.get("truth_ground_longitude"))
        t2lat, t2lon = safe_float(row.get("truth_ground_latitude")), safe_float(row.get("truth_ground_longitude"))
        est_jump = distance_latlon_m(e1lat, e1lon, e2lat, e2lon) if all(math.isfinite(x) for x in [e1lat, e1lon, e2lat, e2lon]) else math.nan
        truth_jump = distance_latlon_m(t1lat, t1lon, t2lat, t2lon) if all(math.isfinite(x) for x in [t1lat, t1lon, t2lat, t2lon]) else math.nan
        est_bearing = bearing_deg(e1lat, e1lon, e2lat, e2lon) if math.isfinite(est_jump) and est_jump > 0.5 else math.nan
        truth_bearing = bearing_deg(t1lat, t1lon, t2lat, t2lon) if math.isfinite(truth_jump) and truth_jump > 0.5 else math.nan
        suspicious = 0
        if math.isfinite(est_jump):
            too_big_abs = est_jump > args.jump_threshold_m
            too_big_vs_truth = math.isfinite(truth_jump) and est_jump > args.jump_vs_truth_factor * max(truth_jump, 1.0)
            low_image_motion = f["median_mag_px"] < args.small_flow_px and f["points_tracked"] >= 30
            suspicious = int((too_big_abs and low_image_motion) or too_big_vs_truth)
        out = {
            "query_sample_index": int(float(row.get("query_sample_index", 0))),
            "query_video_time_s": safe_float(row.get("query_video_time_s")),
            "state": row.get("state", ""),
            "reference_dataset": row.get("reference_dataset", ""),
            "reference_frame_count": row.get("reference_frame_count", ""),
            "position_error_m": safe_float(row.get("position_error_m")),
            "estimated_jump_m": est_jump,
            "truth_jump_m": truth_jump,
            "estimated_bearing_deg": est_bearing,
            "truth_bearing_deg": truth_bearing,
            "points_detected": f["points_detected"],
            "points_tracked": f["points_tracked"],
            "flow_quality": f["flow_quality"],
            "flow_median_dx_px": f["median_dx_px"],
            "flow_median_dy_px": f["median_dy_px"],
            "flow_median_mag_px": f["median_mag_px"],
            "flow_mean_mag_px": f["mean_mag_px"],
            "flow_angle_deg_img": f["angle_deg_img"],
            "suspicious_jump": suspicious,
        }
        out_rows.append(out)
        if args.output_video is not None:
            canvas = draw_flow(prev_bgr, bgr, f, out, args)
            if writer is None:
                args.output_video.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.output_video), fourcc, args.video_fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
        prev_gray, prev_bgr, prev_pred = gray, bgr, row

    if writer is not None:
        writer.release()
    write_csv(args.output_csv, out_rows)
    summary = summarize(out_rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote: {args.output_csv}")
    print(f"wrote: {args.summary_json}")
    if args.output_video is not None:
        print(f"wrote: {args.output_video}")


if __name__ == "__main__":
    main()
