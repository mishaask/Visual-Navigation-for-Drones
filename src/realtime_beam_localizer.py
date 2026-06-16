"""Rolling-window beam-search realtime visual localizer.

This script keeps multiple plausible reference-path hypotheses alive over a
causal rolling window. It is meant to replace greedy frame-by-frame selection.

Design:
- DINO retrieval every sampled frame.
- LightGlue on bounded top candidates only on anchor frames or strong frames.
- Beam search scores appearance + geometric verification + motion continuity.
- Outputs the oldest stable state from the best beam after a small delay.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anyloc_dino_retrieval import (  # noqa: E402
    choose_device,
    compute_descriptors,
    load_dinov2,
    mean_pool_descriptor,
    patch_descriptors_for_image,
)
from lightglue import LightGlue, SuperPoint  # noqa: E402
from lightglue.utils import load_image  # noqa: E402

EARTH_RADIUS_M = 6_378_137.0


@dataclass
class BeamStep:
    sample_index: int
    video_time_s: float
    query_frame_path: str
    reference_index: int
    reference_row: dict[str, str]
    rank: int
    dino_similarity: float
    lg_match_count: float = 0.0
    lg_mean_score: float = 0.0
    lg_inlier_count: float = 0.0
    lg_inlier_ratio: float = 0.0
    lightglue_verified: bool = False
    unary_cost: float = 0.0
    transition_cost: float = 0.0
    total_increment_cost: float = 0.0
    search_mode: str = "global"
    candidate_pool_size: int = 0
    consensus_votes: int = 0
    consensus_adjustment: float = 0.0
    consensus_cluster: str = ""
    homography_inliers: int = 0
    homography_inlier_ratio: float = 0.0
    homography_reproj_rmse: float = 0.0
    projected_center_inside: bool = False
    projected_center_x_frac: float = math.nan
    projected_center_y_frac: float = math.nan
    projected_center_offset_frac: float = math.nan
    homography_quad_area_frac: float = 0.0
    homography_area_ok: bool = False
    query_inlier_bbox_area_frac: float = 0.0
    query_inlier_bbox_width_frac: float = 0.0
    query_inlier_bbox_height_frac: float = 0.0
    reference_inlier_bbox_area_frac: float = 0.0
    reference_inlier_bbox_width_frac: float = 0.0
    reference_inlier_bbox_height_frac: float = 0.0
    expected_spread_ratio_query_over_ref: float = math.nan
    observed_spread_ratio_query_over_ref: float = math.nan
    spread_balance_error: float = math.nan
    spread_balance_ok: bool = False
    same_altitude_spread_mismatch: bool = False
    query_inlier_spread_ok: bool = False
    reference_inlier_spread_ok: bool = False
    geometry_verified: bool = False
    query_altitude_m: float = math.nan
    reference_altitude_m: float = math.nan
    altitude_ratio_query_over_ref: float = math.nan


@dataclass
class BeamPath:
    steps: list[BeamStep] = field(default_factory=list)
    score: float = 0.0
    verified_count: int = 0

    @property
    def last(self) -> BeamStep | None:
        return self.steps[-1] if self.steps else None

    def copy_extend(self, step: BeamStep, increment: float, max_window: int) -> "BeamPath":
        steps = self.steps + [step]
        if len(steps) > max_window:
            steps = steps[-max_window:]
        return BeamPath(
            steps=steps,
            score=self.score + increment,
            verified_count=self.verified_count + int(step.lightglue_verified),
        )


def parse_manifest_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        p = Path(value)
        return p.stem, p
    ds, path = value.split("=", 1)
    return ds, Path(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_manifest(path: Path, dataset_id: str) -> list[dict[str, str]]:
    rows = read_csv(path)
    for row in rows:
        row["dataset_id"] = dataset_id
    return rows


def load_reference_rows(values: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for value in values:
        ds, path = parse_manifest_arg(value)
        rows = load_manifest(path, ds)
        out.extend(rows)
        print(f"reference {ds}: {len(rows)} rows from {path}")
    if not out:
        raise RuntimeError("No reference rows loaded")
    return out


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


def dist_latlon_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    x, y = local_xy_from_latlon(lat2, lon2, lat1, lon1)
    return math.hypot(x, y)


def row_xy(row: dict[str, Any], origin: tuple[float, float]) -> tuple[float, float]:
    return local_xy_from_latlon(float(row["ground_latitude"]), float(row["ground_longitude"]), origin[0], origin[1])


def row_dist(a: dict[str, Any], b: dict[str, Any], origin: tuple[float, float]) -> float:
    ax, ay = row_xy(a, origin)
    bx, by = row_xy(b, origin)
    return math.hypot(ax - bx, ay - by)


def segment_key(row: dict[str, str], span: int) -> str:
    try:
        frame = int(float(row.get("frame_count", 0)))
    except Exception:
        frame = 0
    return f"{row.get('dataset_id','')}:{frame // max(span, 1)}"


def nearest_truth(truth_rows: list[dict[str, str]], time_s: float) -> dict[str, str] | None:
    if not truth_rows:
        return None
    return min(truth_rows, key=lambda r: abs((safe_float(r.get("start_seconds"), 0.0)) - time_s))


def ensure_reference_descriptors(ref_rows: list[dict[str, str]], args: argparse.Namespace) -> np.ndarray:
    cache = args.reference_descriptor_cache
    if cache.exists() and not args.recompute_reference_descriptors:
        desc = np.load(cache)
        if len(desc) == len(ref_rows):
            print(f"loaded reference descriptors: {cache} ({desc.shape})")
            return desc
        print(f"descriptor length mismatch for {cache}; recomputing")
    cache.parent.mkdir(parents=True, exist_ok=True)
    weights = args.weights_path if args.weights_path.exists() else None
    desc = compute_descriptors(
        ref_rows,
        args.model_name,
        args.max_size,
        args.dinov2_repo,
        weights,
        "mean",
        32,
        100000,
        7,
        3.0,
    )
    np.save(cache, desc)
    print(f"wrote reference descriptors: {cache} ({desc.shape})")
    return desc


def is_verified(step: BeamStep, args: argparse.Namespace) -> bool:
    light_ok = step.lightglue_verified and (
        step.lg_inlier_count >= args.lg_min_inliers or step.lg_inlier_ratio >= args.lg_min_ratio
    )
    if args.require_geometry_verified:
        return light_ok and step.geometry_verified
    return light_ok




def row_altitude_m(row: dict[str, Any]) -> float:
    for key in (
        "rel_alt_m", "rel_alt", "relative_altitude", "relative_altitude_m",
        "drone_rel_altitude", "drone_rel_altitude_m", "altitude_m", "height_m",
    ):
        if key in row:
            v = safe_float(row.get(key))
            if math.isfinite(v) and abs(v) > 0.01:
                return abs(v)
    return math.nan


def extract_features_cached(
    image_path: Path,
    extractor: SuperPoint,
    device: torch.device,
    image_resize: int,
    cache: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    key = str(image_path)
    if key not in cache:
        image = load_image(image_path, resize=image_resize).to(device)
        with torch.no_grad():
            cache[key] = extractor.extract(image)
    return cache[key]


def bbox_fracs(points: np.ndarray, width: float, height: float) -> tuple[float, float, float]:
    if points is None or len(points) == 0 or width <= 1 or height <= 1:
        return 0.0, 0.0, 0.0
    xs = points[:, 0]
    ys = points[:, 1]
    w = float(max(0.0, xs.max() - xs.min())) / float(width)
    h = float(max(0.0, ys.max() - ys.min())) / float(height)
    return w * h, w, h


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def altitude_scaled_spread_thresholds(q_alt: float, r_alt: float, args: argparse.Namespace) -> tuple[float, float, float, float]:
    """Return min width/height fractions for query and reference inlier spread.

    If the query is much lower than the reference, the matched region is allowed
    to occupy a smaller fraction of the reference image. If the reference is
    much lower, the matched region is allowed to occupy a smaller fraction of
    the query image. When altitudes are similar, both thresholds stay near the
    base thresholds.
    """
    q_scale = 1.0
    r_scale = 1.0
    if math.isfinite(q_alt) and math.isfinite(r_alt) and q_alt > 1.0 and r_alt > 1.0:
        q_scale = min(1.0, max(args.min_altitude_spread_scale, r_alt / q_alt))
        r_scale = min(1.0, max(args.min_altitude_spread_scale, q_alt / r_alt))
    return (
        args.min_query_inlier_width_frac * q_scale,
        args.min_query_inlier_height_frac * q_scale,
        args.min_reference_inlier_width_frac * r_scale,
        args.min_reference_inlier_height_frac * r_scale,
    )


def lightglue_geometry_score(
    query_path: Path,
    reference_path: Path,
    query_alt_m: float,
    reference_alt_m: float,
    extractor: SuperPoint,
    matcher: LightGlue,
    device: torch.device,
    image_resize: int,
    feature_cache: dict[str, dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    feats0 = extract_features_cached(query_path, extractor, device, image_resize, feature_cache)
    feats1 = extract_features_cached(reference_path, extractor, device, image_resize, feature_cache)
    with torch.no_grad():
        matches01 = matcher({"image0": feats0, "image1": feats1})

    matches = matches01["matches"][0].detach().cpu().numpy()
    scores = matches01["scores"][0].detach().cpu().numpy()
    match_count = int(len(matches))
    mean_score = float(scores.mean()) if len(scores) else 0.0
    out: dict[str, Any] = {
        "lg_match_count": float(match_count),
        "lg_mean_score": mean_score,
        "lg_inlier_count": 0.0,
        "lg_inlier_ratio": 0.0,
        "homography_inliers": 0,
        "homography_inlier_ratio": 0.0,
        "homography_reproj_rmse": 9999.0,
        "projected_center_inside": False,
        "projected_center_x_frac": math.nan,
        "projected_center_y_frac": math.nan,
        "projected_center_offset_frac": math.nan,
        "homography_quad_area_frac": 0.0,
        "homography_area_ok": False,
        "query_inlier_bbox_area_frac": 0.0,
        "query_inlier_bbox_width_frac": 0.0,
        "query_inlier_bbox_height_frac": 0.0,
        "reference_inlier_bbox_area_frac": 0.0,
        "reference_inlier_bbox_width_frac": 0.0,
        "reference_inlier_bbox_height_frac": 0.0,
        "expected_spread_ratio_query_over_ref": math.nan,
        "observed_spread_ratio_query_over_ref": math.nan,
        "spread_balance_error": math.nan,
        "spread_balance_ok": False,
        "same_altitude_spread_mismatch": False,
        "query_inlier_spread_ok": False,
        "reference_inlier_spread_ok": False,
        "geometry_verified": False,
        "query_altitude_m": query_alt_m,
        "reference_altitude_m": reference_alt_m,
        "altitude_ratio_query_over_ref": (query_alt_m / reference_alt_m) if math.isfinite(query_alt_m) and math.isfinite(reference_alt_m) and reference_alt_m > 1 else math.nan,
    }
    if match_count < 4:
        return out

    keypoints0 = feats0["keypoints"][0].detach().cpu().numpy()
    keypoints1 = feats1["keypoints"][0].detach().cpu().numpy()
    pts0 = keypoints0[matches[:, 0]].astype(np.float32)
    pts1 = keypoints1[matches[:, 1]].astype(np.float32)
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, args.ransac_reproj_threshold)
    if H is None or mask is None:
        return out
    mask = mask.ravel().astype(bool)
    inlier_count = int(mask.sum())
    inlier_ratio = float(inlier_count / max(match_count, 1))
    out["lg_inlier_count"] = float(inlier_count)
    out["lg_inlier_ratio"] = inlier_ratio
    out["homography_inliers"] = inlier_count
    out["homography_inlier_ratio"] = inlier_ratio
    if inlier_count < 4:
        return out

    in0 = pts0[mask]
    in1 = pts1[mask]
    # LightGlue features are extracted on resized images. Use original feature tensor size if available.
    # Fallback to image_resize square-ish thresholds. The spread checks are relative, so this is good enough.
    q_img = cv2.imread(str(query_path))
    r_img = cv2.imread(str(reference_path))
    qh, qw = q_img.shape[:2] if q_img is not None else (image_resize, image_resize)
    rh, rw = r_img.shape[:2] if r_img is not None else (image_resize, image_resize)
    # If load_image resized the coordinates, normalize by the max feature coordinate instead of original image size.
    qw_eff = max(float(np.max(pts0[:,0]) + 1), 1.0)
    qh_eff = max(float(np.max(pts0[:,1]) + 1), 1.0)
    rw_eff = max(float(np.max(pts1[:,0]) + 1), 1.0)
    rh_eff = max(float(np.max(pts1[:,1]) + 1), 1.0)

    q_area, q_w, q_h = bbox_fracs(in0, qw_eff, qh_eff)
    r_area, r_w, r_h = bbox_fracs(in1, rw_eff, rh_eff)
    out.update({
        "query_inlier_bbox_area_frac": q_area,
        "query_inlier_bbox_width_frac": q_w,
        "query_inlier_bbox_height_frac": q_h,
        "reference_inlier_bbox_area_frac": r_area,
        "reference_inlier_bbox_width_frac": r_w,
        "reference_inlier_bbox_height_frac": r_h,
    })

    min_qw, min_qh, min_rw, min_rh = altitude_scaled_spread_thresholds(query_alt_m, reference_alt_m, args)
    q_spread_ok = q_w >= min_qw and q_h >= min_qh and q_area >= args.min_query_inlier_area_frac * min(1.0, max(args.min_altitude_spread_scale, (reference_alt_m / query_alt_m) if math.isfinite(query_alt_m) and math.isfinite(reference_alt_m) and query_alt_m > 1 else 1.0))
    r_spread_ok = r_w >= min_rw and r_h >= min_rh and r_area >= args.min_reference_inlier_area_frac * min(1.0, max(args.min_altitude_spread_scale, (query_alt_m / reference_alt_m) if math.isfinite(query_alt_m) and math.isfinite(reference_alt_m) and reference_alt_m > 1 else 1.0))

    # Spread-balance consistency: if altitudes are similar, inlier support should
    # have similar image coverage. If query is much lower than reference, query
    # support may be wider and reference support may be clustered.
    spread_balance_ok = True
    same_altitude_spread_mismatch = False
    observed_ratio = math.nan
    expected_ratio = math.nan
    balance_error = math.nan
    if q_area > 1e-6 and r_area > 1e-6 and math.isfinite(query_alt_m) and math.isfinite(reference_alt_m) and query_alt_m > 1.0 and reference_alt_m > 1.0:
        observed_ratio = math.sqrt(q_area) / max(math.sqrt(r_area), 1e-6)
        expected_ratio = max(args.min_altitude_spread_scale, min(1.0 / max(args.min_altitude_spread_scale,1e-6), reference_alt_m / query_alt_m))
        balance_error = max(observed_ratio / max(expected_ratio,1e-6), expected_ratio / max(observed_ratio,1e-6))
        spread_balance_ok = balance_error <= args.spread_balance_tolerance
        alt_ratio = max(query_alt_m, reference_alt_m) / max(min(query_alt_m, reference_alt_m), 1e-6)
        same_altitude_spread_mismatch = bool(alt_ratio <= args.same_altitude_ratio_threshold and not spread_balance_ok)
    out["expected_spread_ratio_query_over_ref"] = float(expected_ratio) if math.isfinite(expected_ratio) else math.nan
    out["observed_spread_ratio_query_over_ref"] = float(observed_ratio) if math.isfinite(observed_ratio) else math.nan
    out["spread_balance_error"] = float(balance_error) if math.isfinite(balance_error) else math.nan
    out["spread_balance_ok"] = bool(spread_balance_ok)
    out["same_altitude_spread_mismatch"] = bool(same_altitude_spread_mismatch)
    out["query_inlier_spread_ok"] = bool(q_spread_ok)
    out["reference_inlier_spread_ok"] = bool(r_spread_ok)

    proj = cv2.perspectiveTransform(pts0[mask].reshape(-1,1,2), H).reshape(-1,2)
    rmse = float(np.sqrt(np.mean(np.sum((proj - in1) ** 2, axis=1)))) if len(proj) else 9999.0
    out["homography_reproj_rmse"] = rmse

    # Project query image center and corners into reference coordinates.
    center = np.float32([[[qw_eff/2.0, qh_eff/2.0]]])
    pc = cv2.perspectiveTransform(center, H).reshape(2)
    pad_x = args.projected_center_padding_frac * rw_eff
    pad_y = args.projected_center_padding_frac * rh_eff
    out["projected_center_inside"] = bool((-pad_x <= pc[0] <= rw_eff + pad_x) and (-pad_y <= pc[1] <= rh_eff + pad_y))
    out["projected_center_x_frac"] = float(pc[0] / max(rw_eff, 1.0))
    out["projected_center_y_frac"] = float(pc[1] / max(rh_eff, 1.0))
    out["projected_center_offset_frac"] = float(math.hypot(out["projected_center_x_frac"] - 0.5, out["projected_center_y_frac"] - 0.5))

    corners = np.float32([[[0,0]], [[qw_eff-1,0]], [[qw_eff-1,qh_eff-1]], [[0,qh_eff-1]]])
    quad = cv2.perspectiveTransform(corners, H).reshape(-1,2)
    quad_area_frac = polygon_area(quad) / max(1.0, rw_eff * rh_eff)
    out["homography_quad_area_frac"] = quad_area_frac
    out["homography_area_ok"] = bool(args.min_homography_quad_area_frac <= quad_area_frac <= args.max_homography_quad_area_frac)

    out["geometry_verified"] = bool(
        inlier_count >= args.geometry_min_inliers
        and inlier_ratio >= args.geometry_min_inlier_ratio
        and rmse <= args.max_homography_reproj_rmse
        and out["projected_center_inside"]
        and out["homography_area_ok"]
        and q_spread_ok
        and r_spread_ok
        and ((not getattr(args, "require_spread_balance", False)) or out["spread_balance_ok"])
    )
    return out

def unary_cost(c: dict[str, Any], args: argparse.Namespace) -> float:
    # Lower is better. DINO similarity is the main cheap signal, LightGlue adds confidence.
    cost = -args.dino_weight * float(c.get("dino_similarity", 0.0))
    cost -= args.inlier_weight * math.log1p(float(c.get("lg_inlier_count", 0.0)))
    cost -= args.ratio_weight * float(c.get("lg_inlier_ratio", 0.0))
    if not bool(c.get("lightglue_verified", False)):
        cost += args.unverified_penalty
    if args.use_geometry_scoring:
        if bool(c.get("geometry_verified", False)):
            cost -= args.geometry_verified_bonus
        elif bool(c.get("lightglue_verified", False)):
            cost += args.geometry_failed_penalty
    return cost


def transition_cost(prev: BeamStep | None, cur: BeamStep, origin: tuple[float, float], args: argparse.Namespace) -> float:
    if prev is None:
        return 0.0
    d = row_dist(prev.reference_row, cur.reference_row, origin)
    soft_excess = max(0.0, d - args.max_step_m)
    cost = args.motion_weight * (soft_excess / max(args.max_step_m, 1e-6)) ** 2
    if cur.reference_row.get("dataset_id") != prev.reference_row.get("dataset_id"):
        cost += args.dataset_switch_penalty
    if segment_key(cur.reference_row, args.segment_frame_span) != segment_key(prev.reference_row, args.segment_frame_span):
        cost += args.segment_switch_penalty
    if d > args.hard_jump_m and not is_verified(cur, args):
        cost += args.hard_jump_penalty
    return cost


def spatial_cluster_id(row: dict[str, str], origin: tuple[float, float], cluster_m: float) -> tuple[int, int]:
    x, y = row_xy(row, origin)
    return (int(round(x / cluster_m)), int(round(y / cluster_m)))


def consensus_adjustment(path: BeamPath, cur: BeamStep, origin: tuple[float, float], args: argparse.Namespace) -> tuple[float, int, tuple[int, int]]:
    """Return consensus cost adjustment, vote count including current, and cluster id.

    This is the explicit multi-frame consensus patch: a spatial cluster can get
    a bonus when it repeats in the recent beam history, while unverified
    one-frame spikes can receive an additional penalty when strict mode is on.
    """
    cur_cluster = spatial_cluster_id(cur.reference_row, origin, args.consensus_cluster_m)
    if not path.steps:
        votes = 1
    else:
        lookback = path.steps[-max(args.consensus_lookback - 1, 0):]
        votes = 1 + sum(
            1 for s in lookback
            if spatial_cluster_id(s.reference_row, origin, args.consensus_cluster_m) == cur_cluster
        )

    adj = 0.0
    if votes >= args.consensus_min_votes:
        adj -= args.consensus_bonus
    elif args.strict_consensus_for_unverified and not is_verified(cur, args):
        # This is the "2 of last 3" style guard: if a candidate has no
        # geometric verification and its spatial cluster has not repeated
        # recently, do not let it win just because of a single DINO spike.
        adj += args.strict_consensus_penalty
    return adj, votes, cur_cluster


def should_run_lightglue(sample_idx: int, sims: np.ndarray, args: argparse.Namespace) -> tuple[bool, str]:
    if args.lightglue_every <= 1:
        return True, "scheduled_every_frame"
    if sample_idx % args.lightglue_every == 0:
        return True, "scheduled_anchor"
    # Optional early anchor: if top candidates are unusually separated, verify now.
    top = np.sort(sims)[::-1]
    if len(top) >= 2 and (top[0] - top[1]) >= args.dino_margin_trigger:
        return True, "dino_margin_trigger"
    if top[0] >= args.dino_abs_trigger:
        return True, "dino_abs_trigger"
    return False, "dino_only_skip_lg"


def candidate_indices(sims: np.ndarray, beam: list[BeamPath], ref_rows: list[dict[str, str]], origin: tuple[float, float], args: argparse.Namespace) -> list[int]:
    global_idx = list(np.argsort(sims)[::-1][: args.global_topk])
    local_idx: list[int] = []
    if beam and args.local_expand_topk > 0:
        seen_centers = [p.last.reference_row for p in beam if p.last is not None]
        for center in seen_centers[: args.local_expand_beams]:
            near = []
            for i, r in enumerate(ref_rows):
                d = row_dist(center, r, origin)
                if d <= args.local_radius_m:
                    near.append((float(sims[i]), i))
            near_sorted = [i for _s, i in sorted(near, key=lambda x: x[0], reverse=True)[: args.local_expand_topk]]
            local_idx.extend(near_sorted)
    out = []
    seen = set()
    for idx in global_idx + local_idx:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(int(idx))
        if len(out) >= args.candidate_pool_limit:
            break
    return out




def candidate_indices_anchor_region(
    sims: np.ndarray,
    beam: list[BeamPath],
    ref_rows: list[dict[str, str]],
    origin: tuple[float, float],
    args: argparse.Namespace,
    anchor_row: dict[str, str] | None,
    anchor_fail_count: int,
) -> tuple[list[int], str]:
    """Candidate selection with hard anchor-region gating.

    In anchor-region mode, global search is allowed only while acquiring an
    anchor or after N consecutive failed local frames. During normal lock, only
    reference frames inside the anchor radius can compete.
    """
    if not args.anchor_region_mode:
        return candidate_indices(sims, beam, ref_rows, origin, args), "GLOBAL_PLUS_BEAM_LOCAL"

    if anchor_row is None:
        global_idx = list(np.argsort(sims)[::-1][: args.global_topk])
        return [int(i) for i in global_idx[: args.candidate_pool_limit]], "ACQUIRE_GLOBAL"

    if anchor_fail_count >= args.anchor_fail_limit:
        global_idx = list(np.argsort(sims)[::-1][: args.global_topk])
        return [int(i) for i in global_idx[: args.candidate_pool_limit]], "REACQUIRE_GLOBAL_AFTER_FAIL_LIMIT"

    radius = args.anchor_radius_m
    mode = "LOCKED_LOCAL_REGION"
    if anchor_fail_count >= args.anchor_recovery_after:
        radius = args.anchor_recovery_radius_m
        mode = "RECOVERY_WIDE_LOCAL_REGION"

    local: list[tuple[float, int]] = []
    for i, r in enumerate(ref_rows):
        try:
            d = row_dist(anchor_row, r, origin)
        except Exception:
            continue
        if d <= radius:
            local.append((float(sims[i]), i))
    local_idx = [int(i) for _score, i in sorted(local, key=lambda x: x[0], reverse=True)[: args.candidate_pool_limit]]

    if not local_idx:
        nearest = sorted(
            [(row_dist(anchor_row, r, origin), i) for i, r in enumerate(ref_rows)],
            key=lambda x: x[0],
        )[: max(1, args.anchor_min_local_candidates)]
        local_idx = [int(i) for _d, i in nearest]
        mode = "LOCKED_LOCAL_FALLBACK_NEAREST"
    return local_idx, mode


def good_anchor_step(step: BeamStep | None, args: argparse.Namespace) -> bool:
    if step is None:
        return False
    return is_verified(step, args)

def score_candidates(frame_path: Path, sample_idx: int, video_time_s: float, sims: np.ndarray, indices: list[int], ref_rows: list[dict[str, str]], query_truth: dict[str, str] | None, run_lg: bool, lg_reason: str, extractor: SuperPoint, matcher: LightGlue, device: torch.device, args: argparse.Namespace) -> list[BeamStep]:
    feature_cache: dict[str, dict[str, torch.Tensor]] = {}
    steps: list[BeamStep] = []
    for rank, idx in enumerate(indices, start=1):
        row = ref_rows[idx]
        step = BeamStep(
            sample_index=sample_idx,
            video_time_s=video_time_s,
            query_frame_path=str(frame_path),
            reference_index=idx,
            reference_row=row,
            rank=rank,
            dino_similarity=float(sims[idx]),
            search_mode=lg_reason,
            candidate_pool_size=len(indices),
        )
        steps.append(step)
    if run_lg:
        query_alt = row_altitude_m(query_truth or {})
        for step in steps[: args.lg_topk]:
            ref_alt = row_altitude_m(step.reference_row)
            lg = lightglue_geometry_score(
                frame_path, Path(step.reference_row["frame_path"]), query_alt, ref_alt,
                extractor, matcher, device, args.image_resize, feature_cache, args
            )
            step.lg_match_count = float(lg.get("lg_match_count", 0.0))
            step.lg_mean_score = float(lg.get("lg_mean_score", 0.0))
            step.lg_inlier_count = float(lg.get("lg_inlier_count", 0.0))
            step.lg_inlier_ratio = float(lg.get("lg_inlier_ratio", 0.0))
            step.homography_inliers = int(lg.get("homography_inliers", 0))
            step.homography_inlier_ratio = float(lg.get("homography_inlier_ratio", 0.0))
            step.homography_reproj_rmse = float(lg.get("homography_reproj_rmse", 0.0))
            step.projected_center_inside = bool(lg.get("projected_center_inside", False))
            step.projected_center_x_frac = float(lg.get("projected_center_x_frac", math.nan))
            step.projected_center_y_frac = float(lg.get("projected_center_y_frac", math.nan))
            step.projected_center_offset_frac = float(lg.get("projected_center_offset_frac", math.nan))
            step.homography_quad_area_frac = float(lg.get("homography_quad_area_frac", 0.0))
            step.homography_area_ok = bool(lg.get("homography_area_ok", False))
            step.query_inlier_bbox_area_frac = float(lg.get("query_inlier_bbox_area_frac", 0.0))
            step.query_inlier_bbox_width_frac = float(lg.get("query_inlier_bbox_width_frac", 0.0))
            step.query_inlier_bbox_height_frac = float(lg.get("query_inlier_bbox_height_frac", 0.0))
            step.reference_inlier_bbox_area_frac = float(lg.get("reference_inlier_bbox_area_frac", 0.0))
            step.reference_inlier_bbox_width_frac = float(lg.get("reference_inlier_bbox_width_frac", 0.0))
            step.reference_inlier_bbox_height_frac = float(lg.get("reference_inlier_bbox_height_frac", 0.0))
            step.expected_spread_ratio_query_over_ref = float(lg.get("expected_spread_ratio_query_over_ref", math.nan))
            step.observed_spread_ratio_query_over_ref = float(lg.get("observed_spread_ratio_query_over_ref", math.nan))
            step.spread_balance_error = float(lg.get("spread_balance_error", math.nan))
            step.spread_balance_ok = bool(lg.get("spread_balance_ok", False))
            step.same_altitude_spread_mismatch = bool(lg.get("same_altitude_spread_mismatch", False))
            step.query_inlier_spread_ok = bool(lg.get("query_inlier_spread_ok", False))
            step.reference_inlier_spread_ok = bool(lg.get("reference_inlier_spread_ok", False))
            step.geometry_verified = bool(lg.get("geometry_verified", False))
            step.query_altitude_m = float(lg.get("query_altitude_m", math.nan))
            step.reference_altitude_m = float(lg.get("reference_altitude_m", math.nan))
            step.altitude_ratio_query_over_ref = float(lg.get("altitude_ratio_query_over_ref", math.nan))
            step.lightglue_verified = True
    for step in steps:
        step.unary_cost = unary_cost(step.__dict__, args)
    return steps


def update_beam(beam: list[BeamPath], candidates: list[BeamStep], origin: tuple[float, float], args: argparse.Namespace) -> list[BeamPath]:
    new_paths: list[BeamPath] = []
    if not beam:
        for c in candidates[: args.beam_width]:
            inc = c.unary_cost
            c.transition_cost = 0.0
            c.consensus_votes = 1
            c.consensus_adjustment = 0.0
            c.consensus_cluster = "bootstrap"
            c.total_increment_cost = inc
            new_paths.append(BeamPath(steps=[c], score=inc, verified_count=int(is_verified(c, args))))
    else:
        # Limit candidate expansion to keep latency bounded.
        cand_subset = candidates[: args.expand_candidates]
        for path in beam:
            prev = path.last
            for c in cand_subset:
                tc = transition_cost(prev, c, origin, args)
                cb, votes, cluster = consensus_adjustment(path, c, origin, args)
                inc = c.unary_cost + tc + cb
                c2 = BeamStep(**{**c.__dict__})
                c2.transition_cost = tc
                c2.consensus_votes = votes
                c2.consensus_adjustment = cb
                c2.consensus_cluster = f"{cluster[0]},{cluster[1]}"
                c2.total_increment_cost = inc
                new_paths.append(path.copy_extend(c2, inc, args.window))
    # Prefer lower score; mild bonus for paths with verified matches.
    new_paths.sort(key=lambda p: p.score - args.verified_path_bonus * p.verified_count)
    return new_paths[: args.beam_width]


def output_row_from_step(step: BeamStep, output_state: str, beam_rank: int, best_score: float, truth_rows: list[dict[str, str]]) -> dict[str, Any]:
    ref = step.reference_row
    row: dict[str, Any] = {
        "query_sample_index": step.sample_index,
        "query_video_time_s": round(step.video_time_s, 3),
        "query_frame_path": step.query_frame_path,
        "state": output_state,
        "beam_rank": beam_rank,
        "beam_score": best_score,
        "search_mode": step.search_mode,
        "candidate_pool_size": step.candidate_pool_size,
        "consensus_votes": step.consensus_votes,
        "consensus_adjustment": step.consensus_adjustment,
        "consensus_cluster": step.consensus_cluster,
        "reference_dataset": ref.get("dataset_id", ""),
        "reference_frame_count": ref.get("frame_count", ""),
        "reference_frame_path": ref.get("frame_path", ""),
        "reference_segment_key": segment_key(ref, 3000),
        "rank": step.rank,
        "dino_similarity": step.dino_similarity,
        "lg_match_count": int(step.lg_match_count),
        "lg_inlier_count": int(step.lg_inlier_count),
        "lg_inlier_ratio": step.lg_inlier_ratio,
        "lightglue_verified": int(step.lightglue_verified),
        "homography_inliers": step.homography_inliers,
        "homography_inlier_ratio": step.homography_inlier_ratio,
        "homography_reproj_rmse": step.homography_reproj_rmse,
        "projected_center_inside": int(step.projected_center_inside),
        "projected_center_x_frac": step.projected_center_x_frac,
        "projected_center_y_frac": step.projected_center_y_frac,
        "projected_center_offset_frac": step.projected_center_offset_frac,
        "homography_quad_area_frac": step.homography_quad_area_frac,
        "homography_area_ok": int(step.homography_area_ok),
        "query_inlier_bbox_area_frac": step.query_inlier_bbox_area_frac,
        "query_inlier_bbox_width_frac": step.query_inlier_bbox_width_frac,
        "query_inlier_bbox_height_frac": step.query_inlier_bbox_height_frac,
        "reference_inlier_bbox_area_frac": step.reference_inlier_bbox_area_frac,
        "reference_inlier_bbox_width_frac": step.reference_inlier_bbox_width_frac,
        "reference_inlier_bbox_height_frac": step.reference_inlier_bbox_height_frac,
        "expected_spread_ratio_query_over_ref": step.expected_spread_ratio_query_over_ref,
        "observed_spread_ratio_query_over_ref": step.observed_spread_ratio_query_over_ref,
        "spread_balance_error": step.spread_balance_error,
        "spread_balance_ok": int(step.spread_balance_ok),
        "same_altitude_spread_mismatch": int(step.same_altitude_spread_mismatch),
        "query_inlier_spread_ok": int(step.query_inlier_spread_ok),
        "reference_inlier_spread_ok": int(step.reference_inlier_spread_ok),
        "geometry_verified": int(step.geometry_verified),
        "query_altitude_m": step.query_altitude_m,
        "reference_altitude_m": step.reference_altitude_m,
        "altitude_ratio_query_over_ref": step.altitude_ratio_query_over_ref,
        "unary_cost": step.unary_cost,
        "transition_cost": step.transition_cost,
        "increment_cost": step.total_increment_cost,
        "estimated_drone_latitude": safe_float(ref.get("drone_latitude")),
        "estimated_drone_longitude": safe_float(ref.get("drone_longitude")),
        "estimated_ground_latitude": safe_float(ref.get("ground_latitude")),
        "estimated_ground_longitude": safe_float(ref.get("ground_longitude")),
    }
    truth = nearest_truth(truth_rows, step.video_time_s)
    if truth is not None:
        tlat = safe_float(truth.get("ground_latitude"))
        tlon = safe_float(truth.get("ground_longitude"))
        row["truth_frame_count"] = truth.get("frame_count", "")
        row["truth_ground_latitude"] = tlat
        row["truth_ground_longitude"] = tlon
        tdlat = safe_float(truth.get("drone_latitude"))
        tdlon = safe_float(truth.get("drone_longitude"))
        row["truth_drone_latitude"] = tdlat
        row["truth_drone_longitude"] = tdlon
        if math.isfinite(row["estimated_drone_latitude"]) and math.isfinite(row["estimated_drone_longitude"]) and math.isfinite(tdlat) and math.isfinite(tdlon):
            row["drone_position_error_m"] = dist_latlon_m(row["estimated_drone_latitude"], row["estimated_drone_longitude"], tdlat, tdlon)
        else:
            row["drone_position_error_m"] = ""
        if math.isfinite(row["estimated_ground_latitude"]) and math.isfinite(row["estimated_ground_longitude"]):
            row["position_error_m"] = dist_latlon_m(row["estimated_ground_latitude"], row["estimated_ground_longitude"], tlat, tlon)
        else:
            row["position_error_m"] = ""
    else:
        row["truth_frame_count"] = ""
        row["truth_ground_latitude"] = ""
        row["truth_ground_longitude"] = ""
        row["truth_drone_latitude"] = ""
        row["truth_drone_longitude"] = ""
        row["drone_position_error_m"] = ""
        row["position_error_m"] = ""
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Different realtime states can produce different diagnostic columns.
    # Use the union of keys so later geometry/debug fields do not crash CSV writing.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], latencies: list[float]) -> dict[str, Any]:
    e = np.array([float(r["position_error_m"]) for r in rows if r.get("position_error_m", "") not in ("", None)], dtype=float)
    de = np.array([float(r["drone_position_error_m"]) for r in rows if r.get("drone_position_error_m", "") not in ("", None)], dtype=float)
    lat = np.array(latencies, dtype=float) if latencies else np.array([])
    out: dict[str, Any] = {
        "frames_processed": len(rows),
        "mean_latency_ms": float(lat.mean()) if len(lat) else 0.0,
        "median_latency_ms": float(np.median(lat)) if len(lat) else 0.0,
        "p90_latency_ms": float(np.percentile(lat, 90)) if len(lat) else 0.0,
        "max_latency_ms": float(lat.max()) if len(lat) else 0.0,
    }
    if len(e):
        out.update({
            "evaluated_frames": len(e),
            "mean_error_m": float(e.mean()),
            "median_error_m": float(np.median(e)),
            "p90_error_m": float(np.percentile(e, 90)),
            "p95_error_m": float(np.percentile(e, 95)),
            "max_error_m": float(e.max()),
            "pct_under_50m": float((e < 50).mean() * 100.0),
            "pct_under_100m": float((e < 100).mean() * 100.0),
        })
    if len(de):
        out.update({
            "drone_evaluated_frames": len(de),
            "drone_mean_error_m": float(de.mean()),
            "drone_median_error_m": float(np.median(de)),
            "drone_p90_error_m": float(np.percentile(de, 90)),
            "drone_p95_error_m": float(np.percentile(de, 95)),
            "drone_max_error_m": float(de.max()),
            "drone_pct_under_50m": float((de < 50).mean() * 100.0),
            "drone_pct_under_100m": float((de < 100).mean() * 100.0),
        })
    states: dict[str, int] = {}
    verified = 0
    geometry_verified = 0
    for r in rows:
        states[str(r.get("state", ""))] = states.get(str(r.get("state", "")), 0) + 1
        verified += int(str(r.get("lightglue_verified", "0")) == "1")
        geometry_verified += int(str(r.get("geometry_verified", "0")) == "1")
    out["state_counts"] = states
    out["verified_output_frames"] = verified
    out["geometry_verified_output_frames"] = geometry_verified
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-manifest", action="append", required=True)
    ap.add_argument("--query-video", type=Path, required=True)
    ap.add_argument("--truth-manifest", type=Path)
    ap.add_argument("--reference-descriptor-cache", type=Path, required=True)
    ap.add_argument("--query-frame-dir", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    ap.add_argument("--beam-debug-jsonl", type=Path, required=True)
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=0)

    ap.add_argument("--model-name", default="dinov2_vits14")
    ap.add_argument("--dinov2-repo", type=Path, default=Path("third_party/dinov2"))
    ap.add_argument("--weights-path", type=Path, default=Path("outputs/models/dinov2/dinov2_vits14_pretrain.pth"))
    ap.add_argument("--max-size", type=int, default=518)
    ap.add_argument("--recompute-reference-descriptors", action="store_true")

    ap.add_argument("--beam-width", type=int, default=12)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--global-topk", type=int, default=50)
    ap.add_argument("--candidate-pool-limit", type=int, default=70)
    ap.add_argument("--expand-candidates", type=int, default=20)
    ap.add_argument("--local-expand-topk", type=int, default=12)
    ap.add_argument("--local-expand-beams", type=int, default=4)
    ap.add_argument("--local-radius-m", type=float, default=140.0)
    ap.add_argument("--lg-topk", type=int, default=10)
    ap.add_argument("--lightglue-every", type=int, default=1)
    ap.add_argument("--dino-margin-trigger", type=float, default=0.08)
    ap.add_argument("--dino-abs-trigger", type=float, default=0.92)
    ap.add_argument("--image-resize", type=int, default=768)
    ap.add_argument("--max-keypoints", type=int, default=1024)

    ap.add_argument("--max-step-m", type=float, default=35.0)
    ap.add_argument("--hard-jump-m", type=float, default=140.0)
    ap.add_argument("--segment-frame-span", type=int, default=3000)
    ap.add_argument("--lg-min-inliers", type=float, default=12.0)
    ap.add_argument("--lg-min-ratio", type=float, default=0.08)

    ap.add_argument("--dino-weight", type=float, default=4.0)
    ap.add_argument("--inlier-weight", type=float, default=1.2)
    ap.add_argument("--ratio-weight", type=float, default=1.0)
    ap.add_argument("--motion-weight", type=float, default=8.0)
    ap.add_argument("--dataset-switch-penalty", type=float, default=3.0)
    ap.add_argument("--segment-switch-penalty", type=float, default=0.8)
    ap.add_argument("--unverified-penalty", type=float, default=2.0)
    ap.add_argument("--hard-jump-penalty", type=float, default=1000.0)
    ap.add_argument("--verified-path-bonus", type=float, default=0.25)
    ap.add_argument("--consensus-lookback", type=int, default=4)
    ap.add_argument("--consensus-min-votes", type=int, default=2)
    ap.add_argument("--consensus-cluster-m", type=float, default=80.0)
    ap.add_argument("--consensus-bonus", type=float, default=1.0)
    ap.add_argument("--strict-consensus-for-unverified", action="store_true")
    ap.add_argument("--strict-consensus-penalty", type=float, default=2.5)

    ap.add_argument("--use-geometry-scoring", action="store_true")
    ap.add_argument("--require-geometry-verified", action="store_true")
    ap.add_argument("--geometry-verified-bonus", type=float, default=2.0)
    ap.add_argument("--geometry-failed-penalty", type=float, default=2.5)
    ap.add_argument("--geometry-min-inliers", type=int, default=10)
    ap.add_argument("--geometry-min-inlier-ratio", type=float, default=0.10)
    ap.add_argument("--ransac-reproj-threshold", type=float, default=5.0)
    ap.add_argument("--max-homography-reproj-rmse", type=float, default=8.0)
    ap.add_argument("--projected-center-padding-frac", type=float, default=0.20)
    ap.add_argument("--min-homography-quad-area-frac", type=float, default=0.005)
    ap.add_argument("--max-homography-quad-area-frac", type=float, default=1.50)
    ap.add_argument("--min-query-inlier-width-frac", type=float, default=0.16)
    ap.add_argument("--min-query-inlier-height-frac", type=float, default=0.10)
    ap.add_argument("--min-query-inlier-area-frac", type=float, default=0.012)
    ap.add_argument("--min-reference-inlier-width-frac", type=float, default=0.16)
    ap.add_argument("--min-reference-inlier-height-frac", type=float, default=0.10)
    ap.add_argument("--min-reference-inlier-area-frac", type=float, default=0.012)
    ap.add_argument("--min-altitude-spread-scale", type=float, default=0.25)

    ap.add_argument("--anchor-region-mode", action="store_true")
    ap.add_argument("--anchor-radius-m", type=float, default=180.0)
    ap.add_argument("--anchor-recovery-after", type=int, default=5)
    ap.add_argument("--anchor-recovery-radius-m", type=float, default=300.0)
    ap.add_argument("--anchor-fail-limit", type=int, default=10)
    ap.add_argument("--anchor-confirm-frames", type=int, default=2)
    ap.add_argument("--anchor-min-local-candidates", type=int, default=8)
    args = ap.parse_args()

    ref_rows = load_reference_rows(args.reference_manifest)
    origin = (float(ref_rows[0]["ground_latitude"]), float(ref_rows[0]["ground_longitude"]))
    ref_desc = ensure_reference_descriptors(ref_rows, args)

    truth_rows: list[dict[str, str]] = []
    if args.truth_manifest is not None and args.truth_manifest.exists():
        truth_rows = load_manifest(args.truth_manifest, "truth")
        print(f"truth rows for evaluation only: {len(truth_rows)}")

    device = choose_device()
    print(f"runtime device: {device}")
    weights = args.weights_path if args.weights_path.exists() else None
    model = load_dinov2(args.model_name, device, args.dinov2_repo, weights)
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    cap = cv2.VideoCapture(str(args.query_video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.query_video}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(native_fps / max(args.sample_fps, 1e-6))))
    print(f"query video fps={native_fps:.3f}; sampling every {frame_step} frames (~{args.sample_fps} fps)")
    print(f"beam_width={args.beam_width}, window={args.window}, lightglue_every={args.lightglue_every}, strict_consensus={args.strict_consensus_for_unverified}, anchor_region_mode={args.anchor_region_mode}")

    args.query_frame_dir.mkdir(parents=True, exist_ok=True)
    args.beam_debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
    debug_f = args.beam_debug_jsonl.open("w", encoding="utf-8")

    beam: list[BeamPath] = []
    pending_outputs: list[BeamStep] = []
    output_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    frame_idx = -1
    sample_idx = 0
    anchor_row: dict[str, str] | None = None
    anchor_fail_count = 0
    pending_anchor_cluster: tuple[int, int] | None = None
    pending_anchor_count = 0
    pbar = tqdm(desc="beam realtime frames")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue
        if args.max_frames and sample_idx >= args.max_frames:
            break
        video_time_s = frame_idx / native_fps
        t0 = time.perf_counter()
        frame_path = args.query_frame_dir / f"query_{sample_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        patches = patch_descriptors_for_image(model, frame_path, device=device, max_size=args.max_size)
        q_desc = mean_pool_descriptor(patches)
        sims = ref_desc @ q_desc
        run_lg, lg_reason = should_run_lightglue(sample_idx, sims, args)
        idxs, anchor_search_mode = candidate_indices_anchor_region(
            sims, beam, ref_rows, origin, args, anchor_row, anchor_fail_count
        )
        query_truth = nearest_truth(truth_rows, video_time_s) if truth_rows else None
        candidates = score_candidates(
            frame_path, sample_idx, video_time_s, sims, idxs, ref_rows, query_truth,
            run_lg, f"{anchor_search_mode}:{lg_reason}", extractor, matcher, device, args
        )
        beam = update_beam(beam, candidates, origin, args)
        best = beam[0]

        # Hard anchor-region state update. This controls where the NEXT frame
        # is allowed to search. Global search is not allowed while locked unless
        # anchor_fail_count reaches anchor_fail_limit.
        if args.anchor_region_mode:
            current = best.last
            current_good = good_anchor_step(current, args)
            if anchor_row is None or anchor_fail_count >= args.anchor_fail_limit:
                if current_good and current is not None:
                    cl = spatial_cluster_id(current.reference_row, origin, args.consensus_cluster_m)
                    if pending_anchor_cluster == cl:
                        pending_anchor_count += 1
                    else:
                        pending_anchor_cluster = cl
                        pending_anchor_count = 1
                    if pending_anchor_count >= args.anchor_confirm_frames:
                        anchor_row = current.reference_row
                        anchor_fail_count = 0
                        pending_anchor_count = 0
                else:
                    pending_anchor_count = 0
                    pending_anchor_cluster = None
            else:
                if current_good and current is not None:
                    anchor_row = current.reference_row
                    anchor_fail_count = 0
                else:
                    anchor_fail_count += 1

        # Commit oldest state from best path once the rolling window is full.
        if len(best.steps) >= args.window:
            committed = best.steps[0]
            out_state = "ANCHOR_REGION_TRACK" if args.anchor_region_mode else "BEAM_TRACK"
            row = output_row_from_step(committed, out_state, 0, best.score, truth_rows)
            row["anchor_fail_count_at_output"] = anchor_fail_count
            output_rows.append(row)
            # Drop committed step from all beam paths so the next output can advance.
            for p in beam:
                if p.steps:
                    p.steps = p.steps[1:]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)
        top_debug = []
        for rank, p in enumerate(beam[: min(5, len(beam))]):
            if p.last is None:
                continue
            top_debug.append({
                "rank": rank,
                "score": p.score,
                "last_ref_dataset": p.last.reference_row.get("dataset_id"),
                "last_ref_frame": p.last.reference_row.get("frame_count"),
                "last_rank": p.last.rank,
                "last_dino": p.last.dino_similarity,
                "last_lg_inliers": p.last.lg_inlier_count,
                "last_verified": p.last.lightglue_verified,
                "last_geometry_verified": p.last.geometry_verified,
                "last_homography_inliers": p.last.homography_inliers,
            })
        debug_f.write(json.dumps({
            "sample_index": sample_idx,
            "video_time_s": video_time_s,
            "run_lightglue": run_lg,
            "lightglue_reason": lg_reason,
            "candidate_count": len(candidates),
            "anchor_search_mode": anchor_search_mode,
            "anchor_fail_count": anchor_fail_count,
            "anchor_dataset": anchor_row.get("dataset_id") if anchor_row else None,
            "anchor_frame": anchor_row.get("frame_count") if anchor_row else None,
            "latency_ms": latency_ms,
            "top_beam": top_debug,
        }) + "\n")
        debug_f.flush()
        write_csv(args.output_csv, output_rows)
        sample_idx += 1
        pbar.update(1)

    # Flush remaining delayed states from best beam.
    if beam:
        while beam[0].steps:
            committed = beam[0].steps.pop(0)
            row = output_row_from_step(committed, "BEAM_FLUSH", 0, beam[0].score, truth_rows)
            output_rows.append(row)
    pbar.close()
    cap.release()
    debug_f.close()
    output_rows.sort(key=lambda r: int(r["query_sample_index"]))
    write_csv(args.output_csv, output_rows)
    summary = summarize(output_rows, latencies)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote: {args.output_csv}")
    print(f"wrote: {args.summary_json}")
    print(f"wrote: {args.beam_debug_jsonl}")


if __name__ == "__main__":
    main()
