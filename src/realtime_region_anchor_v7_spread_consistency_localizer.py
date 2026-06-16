"""Region-anchor realtime localizer v6: stale-hold gaps + speed cap.

Compared with v2, this version fixes two failure modes observed in Test1:
- no fake ACQUIRE_HOLD path: while acquiring/reacquiring, unconfirmed frames can
  be written as NO_ESTIMATE instead of repeating stale coordinates;
- dynamic local regions: when locked, the search ring expands according to
  optical-flow speed, altitude, recent accepted motion, and local failure count.

FLOW is used as a motion prior, not as the final localizer. DINO/LightGlue/
homography still decide whether a candidate is visually and geometrically valid.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anyloc_dino_retrieval import choose_device, load_dinov2, mean_pool_descriptor, patch_descriptors_for_image, compute_descriptors
from realtime_beam_localizer import (
    BeamStep,
    dist_latlon_m,
    ensure_reference_descriptors,
    good_anchor_step,
    is_verified,
    lightglue_geometry_score,
    local_xy_from_latlon,
    nearest_truth,
    output_row_from_step,
    read_csv,
    row_altitude_m,
    row_dist,
    safe_float,
    score_candidates,
    summarize,
    spatial_cluster_id,
    unary_cost,
    write_csv,
)

EARTH_RADIUS_M = 6_378_137.0


def parse_manifest_arg(value: str) -> tuple[str, Path]:
    if '=' not in value:
        p = Path(value)
        return p.stem, p
    ds, path = value.split('=', 1)
    return ds, Path(path)


def load_reference_rows_from_regions(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    for i, r in enumerate(rows):
        r.setdefault('reference_index', str(i))
        r.setdefault('dataset_id', r.get('dataset_id', 'refs'))
    return rows


def load_manifest(path: Path, dataset_id: str) -> list[dict[str, str]]:
    rows = read_csv(path)
    for r in rows:
        r['dataset_id'] = dataset_id
    return rows


def load_reference_rows(values: list[str]) -> list[dict[str, str]]:
    out=[]
    for v in values:
        ds, path = parse_manifest_arg(v)
        rows = load_manifest(path, ds)
        out.extend(rows)
        print(f'reference {ds}: {len(rows)} rows from {path}')
    for i, r in enumerate(out):
        r.setdefault('reference_index', str(i))
    return out


def row_region_id(row: dict[str, Any]) -> str:
    return str(row.get('region_id') or '')


def row_region_xy(row: dict[str, Any]) -> tuple[int, int] | None:
    try:
        return int(float(row['region_x'])), int(float(row['region_y']))
    except Exception:
        return None


def add_regions_if_missing(rows: list[dict[str, str]], grid_m: float) -> None:
    if rows and rows[0].get('region_id'):
        return
    pts=[]
    for r in rows:
        lat=safe_float(r.get('ground_latitude'))
        lon=safe_float(r.get('ground_longitude'))
        if math.isfinite(lat) and math.isfinite(lon): pts.append((lat,lon))
    if not pts: raise RuntimeError('reference rows need ground_latitude/ground_longitude')
    origin=(sum(a for a,_ in pts)/len(pts), sum(b for _,b in pts)/len(pts))
    for r in rows:
        lat=safe_float(r.get('ground_latitude'))
        lon=safe_float(r.get('ground_longitude'))
        x,y=local_xy_from_latlon(lat,lon,origin[0],origin[1])
        gx=int(math.floor(x/grid_m)); gy=int(math.floor(y/grid_m))
        r['region_id']=f'r_{gx}_{gy}'
        r['region_x']=str(gx); r['region_y']=str(gy); r['region_grid_m']=str(grid_m)


def region_candidates(ref_rows: list[dict[str,str]], accepted_region: tuple[int,int], ring: int) -> list[int]:
    """Return rows inside Chebyshev ring around one region."""
    ax, ay = accepted_region
    out=[]
    for i,r in enumerate(ref_rows):
        xy=row_region_xy(r)
        if xy is None: continue
        if max(abs(xy[0]-ax), abs(xy[1]-ay)) <= ring:
            out.append(i)
    return out


def region_candidates_multi(ref_rows: list[dict[str,str]], centers: list[tuple[int,int]], ring: int) -> list[int]:
    """Return rows near any center. Used for anchor region + flow-predicted region."""
    centers=[c for c in centers if c is not None]
    if not centers:
        return []
    out=[]
    seen=set()
    for c in centers:
        for i in region_candidates(ref_rows, c, ring):
            if i not in seen:
                seen.add(i); out.append(i)
    return out


def row_xy_m(row: dict[str, Any], origin: tuple[float,float]) -> tuple[float,float] | None:
    lat=safe_float(row.get('ground_latitude'))
    lon=safe_float(row.get('ground_longitude'))
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    return local_xy_from_latlon(lat, lon, origin[0], origin[1])


def estimate_fallback_m_per_px(altitude_m: float, image_width_px: int, args: argparse.Namespace) -> float:
    """Approximate ground meters per pixel from altitude and horizontal FOV.

    This is intentionally rough. It is only used to choose a reachable search
    radius/ring, not to output absolute position.
    """
    alt = altitude_m if math.isfinite(altitude_m) and altitude_m > 1.0 else args.default_altitude_m
    ground_width = 2.0 * alt * math.tan(math.radians(args.camera_hfov_deg) / 2.0)
    mpp = args.oblique_scale * ground_width / max(float(image_width_px), 1.0)
    return float(min(args.max_m_per_px, max(args.min_m_per_px, mpp)))


def flow_stats(prev_gray: np.ndarray | None, gray: np.ndarray | None) -> dict[str, float | int]:
    """Robust LK optical flow summary between consecutive sampled query frames."""
    out={'dx_px':0.0,'dy_px':0.0,'mag_px':0.0,'quality':0.0,'points':0}
    if prev_gray is None or gray is None:
        return out
    pts0=cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=8, blockSize=7)
    if pts0 is None or len(pts0)<10:
        return out
    pts1, st, err=cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts0, None, winSize=(21,21), maxLevel=3)
    if pts1 is None or st is None:
        return out
    good=st.ravel().astype(bool)
    if good.sum()<5:
        out['quality']=float(good.mean()) if len(good) else 0.0
        out['points']=int(good.sum())
        return out
    d=(pts1[good].reshape(-1,2)-pts0[good].reshape(-1,2)).astype(float)
    # Robustly trim extreme flow vectors caused by independently moving objects or bad tracks.
    mags=np.linalg.norm(d,axis=1)
    if len(mags) >= 10:
        med=float(np.median(mags)); mad=float(np.median(np.abs(mags-med))) + 1e-6
        keep=np.abs(mags-med) <= max(8.0, 3.5*mad)
        if keep.sum() >= 5:
            d=d[keep]; mags=mags[keep]
    dx=float(np.median(d[:,0])); dy=float(np.median(d[:,1])); mag=float(np.median(mags))
    out.update({'dx_px':dx,'dy_px':dy,'mag_px':mag,'quality':float(good.mean()),'points':int(good.sum())})
    return out


def predicted_region_from_history(good_history: deque, origin: tuple[float,float], grid_m: float, dt: float, flow_speed_m_s: float | None, args: argparse.Namespace) -> tuple[int,int] | None:
    """Predict next region from recent accepted map motion.

    Direction comes from recent accepted map positions. Flow supplies speed
    magnitude when available. This avoids pretending image dx/dy is a compass
    direction when yaw/gimbal yaw is missing.
    """
    if len(good_history) < 2:
        return None
    a=good_history[-2]; b=good_history[-1]
    xy0=row_xy_m(a.reference_row, origin); xy1=row_xy_m(b.reference_row, origin)
    if xy0 is None or xy1 is None:
        return None
    dt_hist=max(1e-6, float(b.video_time_s-a.video_time_s))
    vx=(xy1[0]-xy0[0])/dt_hist; vy=(xy1[1]-xy0[1])/dt_hist
    vmag=math.hypot(vx,vy)
    if vmag < 0.1:
        return row_region_xy(b.reference_row)
    pred_speed=vmag
    if flow_speed_m_s is not None and math.isfinite(flow_speed_m_s) and flow_speed_m_s > 0:
        # Do not let noisy flow create insane predictions; use it as a bounded speed hint.
        pred_speed=min(args.max_flow_pred_speed_mps, max(0.25*vmag, flow_speed_m_s))
    ux,uy=vx/vmag, vy/vmag
    px=xy1[0] + ux*pred_speed*dt
    py=xy1[1] + uy*pred_speed*dt
    return (int(math.floor(px/grid_m)), int(math.floor(py/grid_m)))


def dynamic_region_ring(flow_mag_px: float, flow_speed_m_s: float, altitude_m: float, bad_count: int, args: argparse.Namespace) -> int:
    ring=int(args.locked_region_ring)
    if math.isfinite(altitude_m) and altitude_m > 1.0 and altitude_m <= args.low_altitude_m:
        ring += args.low_altitude_ring_bonus
    if flow_mag_px >= args.fast_flow_px:
        ring += args.fast_flow_ring_bonus
    if flow_mag_px >= args.very_fast_flow_px:
        ring += args.very_fast_flow_ring_bonus
    if math.isfinite(flow_speed_m_s) and flow_speed_m_s >= args.fast_speed_mps:
        ring += args.fast_speed_ring_bonus
    if math.isfinite(flow_speed_m_s) and flow_speed_m_s >= args.very_fast_speed_mps:
        ring += args.very_fast_speed_ring_bonus
    if bad_count >= args.recovery_after:
        ring=max(ring + args.bad_frame_ring_bonus, args.recovery_region_ring)
    return int(max(1, min(args.max_dynamic_region_ring, ring)))


def select_candidate_indices(
    sims: np.ndarray,
    ref_rows: list[dict[str,str]],
    accepted_region: tuple[int,int] | None,
    predicted_region: tuple[int,int] | None,
    bad_count: int,
    dynamic_ring: int,
    args: argparse.Namespace,
) -> tuple[list[int], str]:
    if accepted_region is None:
        idx=list(np.argsort(sims)[::-1][:args.global_topk])
        return [int(i) for i in idx[:args.candidate_pool_limit]], 'ACQUIRE_GLOBAL'
    if bad_count >= args.fail_limit:
        idx=list(np.argsort(sims)[::-1][:args.global_topk])
        return [int(i) for i in idx[:args.candidate_pool_limit]], 'REACQUIRE_GLOBAL_AFTER_FAIL_LIMIT'
    centers=[accepted_region]
    if predicted_region is not None:
        centers.append(predicted_region)
    mode='LOCKED_REGION_FLOW_DYNAMIC'
    if bad_count >= args.recovery_after:
        mode='RECOVERY_LOCAL_REGION_FLOW_DYNAMIC'
    pool = region_candidates_multi(ref_rows, centers, dynamic_ring)
    if not pool:
        idx=list(np.argsort(sims)[::-1][:args.global_topk])
        return [int(i) for i in idx[:args.candidate_pool_limit]], 'EMPTY_DYNAMIC_REGION_GLOBAL_FALLBACK'
    ranked=sorted(pool, key=lambda i: float(sims[i]), reverse=True)[:args.candidate_pool_limit]
    return [int(i) for i in ranked], mode


def region_votes_from_candidates(candidates: list[BeamStep], args: argparse.Namespace) -> tuple[str, dict[str,float], dict[str,int], dict[str,BeamStep]]:
    """Vote over regions using top-N candidates, not just the selected best frame."""
    votes: dict[str,float]={}
    geom: dict[str,int]={}
    best_by_region: dict[str,BeamStep]={}
    for c in candidates[:args.acquire_vote_topn]:
        rid=row_region_id(c.reference_row)
        if not rid:
            continue
        # Rank/DINO creates recall; geometry gives trust. Keep this bounded so one single frame cannot dominate forever.
        v=max(0.0, float(c.dino_similarity)) + 1.0/max(1.0,float(c.rank))
        if c.lightglue_verified:
            v += args.acquire_verified_vote_bonus
        if c.geometry_verified:
            v += args.acquire_geometry_vote_bonus
            geom[rid]=geom.get(rid,0)+1
        votes[rid]=votes.get(rid,0.0)+v
        old=best_by_region.get(rid)
        if old is None or c.unary_cost < old.unary_cost:
            best_by_region[rid]=c
    if not votes:
        return '', votes, geom, best_by_region
    top=max(votes.items(), key=lambda kv: kv[1])[0]
    return top, votes, geom, best_by_region


def make_no_estimate_row(sample_index:int, video_time_s:float, frame_path:Path, state:str, truth_rows:list[dict[str,str]], extra:dict[str,Any] | None=None) -> dict[str,Any]:
    row: dict[str, Any] = {
        'query_sample_index': sample_index,
        'query_video_time_s': round(video_time_s,3),
        'query_frame_path': str(frame_path),
        'state': state,
        'valid_estimate': 0,
        'beam_rank': '', 'beam_score': '', 'search_mode': '', 'candidate_pool_size': '',
        'consensus_votes': '', 'consensus_adjustment': '', 'consensus_cluster': '',
        'reference_dataset': '', 'reference_frame_count': '', 'reference_frame_path': '', 'reference_segment_key': '',
        'rank': '', 'dino_similarity': '', 'lg_match_count': '', 'lg_inlier_count': '', 'lg_inlier_ratio': '',
        'lightglue_verified': 0, 'homography_inliers': '', 'homography_inlier_ratio': '', 'homography_reproj_rmse': '',
        'projected_center_inside': 0, 'homography_quad_area_frac': '', 'homography_area_ok': 0,
        'query_inlier_bbox_area_frac': '', 'query_inlier_bbox_width_frac': '', 'query_inlier_bbox_height_frac': '',
        'reference_inlier_bbox_area_frac': '', 'reference_inlier_bbox_width_frac': '', 'reference_inlier_bbox_height_frac': '',
        'query_inlier_spread_ok': 0, 'reference_inlier_spread_ok': 0, 'geometry_verified': 0,
        'query_altitude_m': '', 'reference_altitude_m': '', 'altitude_ratio_query_over_ref': '',
        'unary_cost': '', 'transition_cost': '', 'increment_cost': '',
        'estimated_drone_latitude': '', 'estimated_drone_longitude': '',
        'estimated_ground_latitude': '', 'estimated_ground_longitude': '',
        'truth_frame_count': '', 'truth_ground_latitude': '', 'truth_ground_longitude': '',
        'truth_drone_latitude': '', 'truth_drone_longitude': '',
        'drone_position_error_m': '', 'position_error_m': '',
    }
    truth=nearest_truth(truth_rows, video_time_s)
    if truth is not None:
        row['truth_frame_count']=truth.get('frame_count','')
        row['truth_ground_latitude']=safe_float(truth.get('ground_latitude'))
        row['truth_ground_longitude']=safe_float(truth.get('ground_longitude'))
        row['truth_drone_latitude']=safe_float(truth.get('drone_latitude'))
        row['truth_drone_longitude']=safe_float(truth.get('drone_longitude'))
    if extra:
        row.update(extra)
    return row


def clone_step_for_time(step: BeamStep, sample_index: int, video_time_s: float, frame_path: Path, state_note: str) -> BeamStep:
    """Copy an accepted step to the current timestamp for locked-only HOLD states.

    v3 still allows HOLD while already locked, but no longer uses this during
    acquisition/reacquisition when no estimate is trusted.
    """
    return replace(step, sample_index=sample_index, video_time_s=video_time_s, query_frame_path=str(frame_path), search_mode=state_note)


def latlon_add_m(lat: float, lon: float, dx_m: float, dy_m: float) -> tuple[float, float]:
    """Move lat/lon by local ENU meters: dx=east, dy=north."""
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return math.nan, math.nan
    dlat = dy_m / EARTH_RADIUS_M
    dlon = dx_m / (EARTH_RADIUS_M * max(1e-9, math.cos(math.radians(lat))))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def recent_motion_unit_and_speed(good_history: deque, origin: tuple[float,float]) -> tuple[float, float, float] | None:
    """Direction and speed from recent accepted geometry-valid estimates."""
    if len(good_history) < 2:
        return None
    a=good_history[-2]; b=good_history[-1]
    xy0=row_xy_m(a.reference_row, origin); xy1=row_xy_m(b.reference_row, origin)
    if xy0 is None or xy1 is None:
        return None
    dx=xy1[0]-xy0[0]; dy=xy1[1]-xy0[1]
    d=math.hypot(dx,dy)
    dt=max(1e-6, float(b.video_time_s-a.video_time_s))
    if d < 0.25:
        return None
    return dx/d, dy/d, d/dt


def flow_propagated_row_from_step(
    step: BeamStep,
    sample_index: int,
    video_time_s: float,
    frame_path: Path,
    state: str,
    truth_rows: list[dict[str,str]],
    dx_m: float,
    dy_m: float,
    extra: dict[str,Any] | None=None,
) -> dict[str,Any]:
    """Create a propagated estimate row by shifting the last accepted coordinate."""
    st=clone_step_for_time(step, sample_index, video_time_s, frame_path, state)
    row=output_row_from_step(st,state,0,float(step.total_increment_cost),truth_rows)
    gd_lat=safe_float(row.get('estimated_ground_latitude'))
    gd_lon=safe_float(row.get('estimated_ground_longitude'))
    dr_lat=safe_float(row.get('estimated_drone_latitude'))
    dr_lon=safe_float(row.get('estimated_drone_longitude'))
    nglat, nglon=latlon_add_m(gd_lat, gd_lon, dx_m, dy_m)
    ndlat, ndlon=latlon_add_m(dr_lat, dr_lon, dx_m, dy_m)
    row['estimated_ground_latitude']=nglat
    row['estimated_ground_longitude']=nglon
    row['estimated_drone_latitude']=ndlat
    row['estimated_drone_longitude']=ndlon
    row['flow_propagated_dx_m']=dx_m
    row['flow_propagated_dy_m']=dy_m
    row['flow_propagated_step_m']=math.hypot(dx_m,dy_m)
    row['valid_estimate']=1
    # Recompute errors after replacing coordinates.
    tg_lat=safe_float(row.get('truth_ground_latitude'))
    tg_lon=safe_float(row.get('truth_ground_longitude'))
    td_lat=safe_float(row.get('truth_drone_latitude'))
    td_lon=safe_float(row.get('truth_drone_longitude'))
    if math.isfinite(nglat) and math.isfinite(nglon) and math.isfinite(tg_lat) and math.isfinite(tg_lon):
        row['position_error_m']=dist_latlon_m(nglat,nglon,tg_lat,tg_lon)
    if math.isfinite(ndlat) and math.isfinite(ndlon) and math.isfinite(td_lat) and math.isfinite(td_lon):
        row['drone_position_error_m']=dist_latlon_m(ndlat,ndlon,td_lat,td_lon)
    if extra:
        row.update(extra)
    return row

def transition_m(last: BeamStep | None, cur: BeamStep, origin: tuple[float,float]) -> float:
    if last is None: return 0.0
    try: return row_dist(last.reference_row, cur.reference_row, origin)
    except Exception: return 0.0


def score_for_region(step: BeamStep, last_good: BeamStep | None, origin: tuple[float,float], args: argparse.Namespace) -> float:
    score=step.unary_cost
    d=transition_m(last_good, step, origin)
    excess=max(0.0, d-args.max_step_m)
    score += args.motion_weight * (excess / max(args.max_step_m, 1e-6))**2
    if step.geometry_verified:
        score -= args.geometry_verified_bonus
    else:
        score += args.geometry_failed_penalty
    return float(score)


def make_summary_extra(rows: list[dict[str,Any]], debug_counts: dict[str,int]) -> dict[str,Any]:
    return {'search_mode_counts': debug_counts}


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-manifest', action='append')
    ap.add_argument('--reference-regions', type=Path)
    ap.add_argument('--query-video', type=Path, required=True)
    ap.add_argument('--truth-manifest', type=Path)
    ap.add_argument('--reference-descriptor-cache', type=Path, required=True)
    ap.add_argument('--query-frame-dir', type=Path, required=True)
    ap.add_argument('--output-csv', type=Path, required=True)
    ap.add_argument('--summary-json', type=Path, required=True)
    ap.add_argument('--debug-jsonl', type=Path, required=True)
    ap.add_argument('--sample-fps', type=float, default=1.0)
    ap.add_argument('--max-frames', type=int, default=0)

    ap.add_argument('--model-name', default='dinov2_vits14')
    ap.add_argument('--dinov2-repo', type=Path, default=Path('third_party/dinov2'))
    ap.add_argument('--weights-path', type=Path, default=Path('outputs/models/dinov2/dinov2_vits14_pretrain.pth'))
    ap.add_argument('--max-size', type=int, default=518)
    ap.add_argument('--recompute-reference-descriptors', action='store_true')

    ap.add_argument('--region-grid-m', type=float, default=90.0)
    ap.add_argument('--global-topk', type=int, default=80)
    ap.add_argument('--candidate-pool-limit', type=int, default=80)
    ap.add_argument('--locked-region-ring', type=int, default=1)
    ap.add_argument('--recovery-region-ring', type=int, default=2)
    ap.add_argument('--recovery-after', type=int, default=5)
    ap.add_argument('--fail-limit', type=int, default=10)
    ap.add_argument('--acquire-window', type=int, default=5)
    ap.add_argument('--acquire-min-region-votes', type=int, default=3)
    ap.add_argument('--acquire-min-geometry-votes', type=int, default=2)
    ap.add_argument('--transition-confirm-window', type=int, default=3)
    ap.add_argument('--transition-min-votes', type=int, default=2)

    ap.add_argument('--lg-topk', type=int, default=12)
    ap.add_argument('--lightglue-every', type=int, default=1)
    ap.add_argument('--image-resize', type=int, default=768)
    ap.add_argument('--max-keypoints', type=int, default=1024)
    ap.add_argument('--lg-min-inliers', type=float, default=10.0)
    ap.add_argument('--lg-min-ratio', type=float, default=0.08)
    ap.add_argument('--require-geometry-verified', action='store_true')
    ap.add_argument('--use-geometry-scoring', action='store_true')
    ap.add_argument('--geometry-verified-bonus', type=float, default=2.2)
    ap.add_argument('--geometry-failed-penalty', type=float, default=3.0)
    ap.add_argument('--geometry-min-inliers', type=int, default=8)
    ap.add_argument('--geometry-min-inlier-ratio', type=float, default=0.08)
    ap.add_argument('--ransac-reproj-threshold', type=float, default=5.0)
    ap.add_argument('--max-homography-reproj-rmse', type=float, default=10.0)
    ap.add_argument('--projected-center-padding-frac', type=float, default=0.20)
    ap.add_argument('--min-homography-quad-area-frac', type=float, default=0.005)
    ap.add_argument('--max-homography-quad-area-frac', type=float, default=1.50)
    ap.add_argument('--min-query-inlier-width-frac', type=float, default=0.14)
    ap.add_argument('--min-query-inlier-height-frac', type=float, default=0.08)
    ap.add_argument('--min-query-inlier-area-frac', type=float, default=0.010)
    ap.add_argument('--min-reference-inlier-width-frac', type=float, default=0.14)
    ap.add_argument('--min-reference-inlier-height-frac', type=float, default=0.08)
    ap.add_argument('--min-reference-inlier-area-frac', type=float, default=0.010)
    ap.add_argument('--min-altitude-spread-scale', type=float, default=0.25)
    ap.add_argument('--require-spread-balance', action='store_true')
    ap.add_argument('--spread-balance-tolerance', type=float, default=2.5)
    ap.add_argument('--same-altitude-ratio-threshold', type=float, default=1.35)

    ap.add_argument('--dino-weight', type=float, default=4.0)
    ap.add_argument('--inlier-weight', type=float, default=1.15)
    ap.add_argument('--ratio-weight', type=float, default=1.0)
    ap.add_argument('--unverified-penalty', type=float, default=2.8)
    ap.add_argument('--motion-weight', type=float, default=7.0)
    ap.add_argument('--max-step-m', type=float, default=45.0)
    ap.add_argument('--strong-jump-m', type=float, default=140.0)
    ap.add_argument('--low-flow-px', type=float, default=12.0)
    ap.add_argument('--flow-contradiction-enabled', action='store_true')

    # v3 flow-motion options. Flow controls reachable-region radius and prediction,
    # not the final coordinate output.
    ap.add_argument('--camera-hfov-deg', type=float, default=82.0)
    ap.add_argument('--default-altitude-m', type=float, default=100.0)
    ap.add_argument('--oblique-scale', type=float, default=1.35)
    ap.add_argument('--min-m-per-px', type=float, default=0.02)
    ap.add_argument('--max-m-per-px', type=float, default=2.5)
    ap.add_argument('--flow-ema-alpha', type=float, default=0.25)
    ap.add_argument('--flow-calib-min-px', type=float, default=4.0)
    ap.add_argument('--flow-calib-max-jump-m', type=float, default=180.0)
    ap.add_argument('--max-flow-pred-speed-mps', type=float, default=35.0)
    ap.add_argument('--low-altitude-m', type=float, default=45.0)
    ap.add_argument('--low-altitude-ring-bonus', type=int, default=1)
    ap.add_argument('--fast-flow-px', type=float, default=22.0)
    ap.add_argument('--very-fast-flow-px', type=float, default=45.0)
    ap.add_argument('--fast-flow-ring-bonus', type=int, default=1)
    ap.add_argument('--very-fast-flow-ring-bonus', type=int, default=2)
    ap.add_argument('--fast-speed-mps', type=float, default=10.0)
    ap.add_argument('--very-fast-speed-mps', type=float, default=20.0)
    ap.add_argument('--fast-speed-ring-bonus', type=int, default=1)
    ap.add_argument('--very-fast-speed-ring-bonus', type=int, default=2)
    ap.add_argument('--bad-frame-ring-bonus', type=int, default=2)
    ap.add_argument('--max-dynamic-region-ring', type=int, default=6)
    ap.add_argument('--acquire-vote-topn', type=int, default=40)
    ap.add_argument('--acquire-verified-vote-bonus', type=float, default=1.0)
    ap.add_argument('--acquire-geometry-vote-bonus', type=float, default=3.0)
    ap.add_argument('--no-estimate-during-acquire', action='store_true')
    ap.add_argument('--flow-propagate-holds', action='store_true')
    ap.add_argument('--flow-propagation-min-quality', type=float, default=0.55)
    ap.add_argument('--flow-propagation-min-points', type=int, default=40)
    ap.add_argument('--flow-propagation-max-step-m', type=float, default=20.0)
    ap.add_argument('--flow-propagation-max-bad-count', type=int, default=3)
    ap.add_argument('--disable-flow-propagation-in-recovery', action='store_true')
    ap.add_argument('--max-flow-hold-speed-mps', type=float, default=35.0)
    ap.add_argument('--max-flow-speed-mps', type=float, default=35.0)
    ap.add_argument('--hold-max-bad-count', type=int, default=3)
    ap.add_argument('--projected-center-offset-warning-frac', type=float, default=0.28)
    args=ap.parse_args()

    if args.reference_regions and args.reference_regions.exists():
        ref_rows=load_reference_rows_from_regions(args.reference_regions)
        print(f'loaded reference regions: {args.reference_regions} ({len(ref_rows)} rows)')
    else:
        if not args.reference_manifest:
            raise RuntimeError('need --reference-regions or --reference-manifest')
        ref_rows=load_reference_rows(args.reference_manifest)
        add_regions_if_missing(ref_rows, args.region_grid_m)
    if not ref_rows:
        raise RuntimeError('no reference rows')
    origin=(safe_float(ref_rows[0]['ground_latitude']), safe_float(ref_rows[0]['ground_longitude']))
    ref_desc=ensure_reference_descriptors(ref_rows,args)

    truth_rows=[]
    if args.truth_manifest and args.truth_manifest.exists():
        truth_rows=load_manifest(args.truth_manifest,'truth')
        print(f'truth rows for evaluation only: {len(truth_rows)}')

    device=choose_device(); print(f'runtime device: {device}')
    weights=args.weights_path if args.weights_path.exists() else None
    model=load_dinov2(args.model_name, device, args.dinov2_repo, weights)
    extractor=SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher=LightGlue(features='superpoint').eval().to(device)

    cap=cv2.VideoCapture(str(args.query_video))
    if not cap.isOpened(): raise RuntimeError(f'could not open {args.query_video}')
    native_fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step=max(1,int(round(native_fps/max(args.sample_fps,1e-6))))
    print(f'query video fps={native_fps:.3f}; sampling every {frame_step} frames (~{args.sample_fps} fps)')
    print(f'region-anchor v7-spread-consistency: acquire {args.acquire_min_region_votes}/{args.acquire_window}, fail_limit={args.fail_limit}')

    args.query_frame_dir.mkdir(parents=True, exist_ok=True)
    args.debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
    debug_f=args.debug_jsonl.open('w',encoding='utf-8')

    accepted_region: tuple[int,int] | None=None
    accepted_region_id=''
    last_good: BeamStep | None=None
    last_output_step: BeamStep | None=None
    bad_count=0
    acquire_hist=deque(maxlen=args.acquire_window)
    transition_hist=deque(maxlen=args.transition_confirm_window)
    good_history=deque(maxlen=4)   # accepted/locked geometry-valid map positions
    output_rows=[]; latencies=[]; mode_counts=Counter()
    prev_gray=None
    flow_m_per_px_ema=math.nan
    frame_idx=-1; sample_idx=0
    pbar=tqdm(desc='region-anchor-v7-spread-consistency realtime frames')
    while True:
        ok, frame=cap.read()
        if not ok: break
        frame_idx += 1
        if frame_idx % frame_step != 0: continue
        if args.max_frames and sample_idx >= args.max_frames: break
        video_time_s=frame_idx/native_fps
        dt_sample=frame_step/max(native_fps,1e-6)
        t0=time.perf_counter()
        frame_path=args.query_frame_dir / f'query_{sample_idx:06d}.jpg'
        cv2.imwrite(str(frame_path), frame)
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow=flow_stats(prev_gray, gray)
        prev_gray=gray

        query_truth=nearest_truth(truth_rows, video_time_s) if truth_rows else None
        query_alt=row_altitude_m(query_truth or {})
        fallback_mpp=estimate_fallback_m_per_px(query_alt, int(frame.shape[1]), args)
        active_mpp=flow_m_per_px_ema if math.isfinite(flow_m_per_px_ema) else fallback_mpp
        flow_speed_m_s_raw=float(flow['mag_px'])*active_mpp/max(dt_sample,1e-6)
        flow_speed_m_s=min(args.max_flow_speed_mps, flow_speed_m_s_raw) if math.isfinite(flow_speed_m_s_raw) else flow_speed_m_s_raw
        dyn_ring=dynamic_region_ring(float(flow['mag_px']), flow_speed_m_s, query_alt, bad_count, args)
        predicted_region=predicted_region_from_history(good_history, origin, args.region_grid_m, dt_sample, flow_speed_m_s, args)

        patches=patch_descriptors_for_image(model, frame_path, device=device, max_size=args.max_size)
        q_desc=mean_pool_descriptor(patches)
        sims=ref_desc @ q_desc
        idxs, search_mode=select_candidate_indices(sims, ref_rows, accepted_region, predicted_region, bad_count, dyn_ring, args)
        mode_counts[search_mode]+=1
        run_lg = (args.lightglue_every <= 1) or (sample_idx % args.lightglue_every == 0) or search_mode.startswith('ACQUIRE') or search_mode.startswith('REACQUIRE')
        candidates=score_candidates(frame_path, sample_idx, video_time_s, sims, idxs, ref_rows, query_truth, run_lg, search_mode, extractor, matcher, device, args)
        if not candidates:
            row=make_no_estimate_row(sample_idx, video_time_s, frame_path, 'NO_ESTIMATE_NO_CANDIDATES', truth_rows)
            output_rows.append(row)
            sample_idx += 1; pbar.update(1); continue

        # Score candidates with local motion continuity.
        ranked=[]
        for c in candidates:
            ranked.append((score_for_region(c,last_good,origin,args), c))
        ranked.sort(key=lambda x:x[0])
        best=ranked[0][1]
        best.total_increment_cost=ranked[0][0]
        best_good=is_verified(best,args)
        best_xy=row_region_xy(best.reference_row)
        jump_m=transition_m(last_good,best,origin)
        contradiction=False
        if last_good is not None and jump_m > args.strong_jump_m:
            contradiction=True
        if args.flow_contradiction_enabled and last_good is not None and jump_m > args.max_step_m and float(flow['mag_px']) < args.low_flow_px and int(flow['points']) > 20:
            contradiction=True
        logical = bool(best_good and not contradiction)

        # Flow-fill vector for HOLD frames. Direction comes from recent accepted
        # map motion; magnitude comes from optical-flow speed when reliable.
        flow_prop_dx_m=0.0
        flow_prop_dy_m=0.0
        can_flow_propagate=False
        motion_hint=recent_motion_unit_and_speed(good_history, origin)
        in_recovery_mode = str(search_mode).startswith('RECOVERY')
        if (args.flow_propagate_holds and motion_hint is not None
                and bad_count <= args.flow_propagation_max_bad_count
                and not (args.disable_flow_propagation_in_recovery and in_recovery_mode)):
            ux,uy,recent_speed_m_s=motion_hint
            raw_speed=flow_speed_m_s if math.isfinite(flow_speed_m_s) and flow_speed_m_s > 0 else recent_speed_m_s
            use_speed=min(args.max_flow_hold_speed_mps, max(0.0, raw_speed))
            step_m=min(args.flow_propagation_max_step_m, max(0.0, use_speed*dt_sample))
            if float(flow['quality']) >= args.flow_propagation_min_quality and int(flow['points']) >= args.flow_propagation_min_points and step_m > 0.05:
                flow_prop_dx_m=ux*step_m
                flow_prop_dy_m=uy*step_m
                can_flow_propagate=True

        output_step: BeamStep | None=None
        state=''
        valid_estimate=True
        top_region_id, region_vote_scores, geom_region_counts, best_by_region = region_votes_from_candidates(candidates, args)

        if accepted_region is None or search_mode.startswith('REACQUIRE'):
            # Region-level acquisition uses votes from top-N candidates, not only one best frame.
            top_region_step = best_by_region.get(top_region_id, best)
            top_region_xy = row_region_xy(top_region_step.reference_row) if top_region_step is not None else None
            top_region_geom = int(geom_region_counts.get(top_region_id,0) > 0)
            acquire_hist.append((top_region_id, bool(top_region_geom), top_region_xy, top_region_step))
            region_counts=Counter(r for r,g,xy,st in acquire_hist if r)
            geom_counts=Counter(r for r,g,xy,st in acquire_hist if r and g)
            accept_region_id=None
            for rid0,n in region_counts.items():
                if n >= args.acquire_min_region_votes and geom_counts[rid0] >= args.acquire_min_geometry_votes:
                    accept_region_id=rid0; break
            candidate_for_accept = best_by_region.get(accept_region_id, None) if accept_region_id else None
            if candidate_for_accept is None and accept_region_id == row_region_id(best.reference_row):
                candidate_for_accept = best
            if candidate_for_accept is not None:
                candidate_for_accept.total_increment_cost = candidate_for_accept.unary_cost
            accept_xy = row_region_xy(candidate_for_accept.reference_row) if candidate_for_accept is not None else None
            accept_logical = bool(candidate_for_accept is not None and is_verified(candidate_for_accept,args) and not contradiction)
            if accept_region_id and accept_logical and accept_xy is not None:
                accepted_region=accept_xy
                accepted_region_id=accept_region_id
                last_good=candidate_for_accept
                good_history.append(candidate_for_accept)
                bad_count=0
                state='ACQUIRE_ACCEPTED_REGION'
                output_step=candidate_for_accept
            else:
                # No fake old path while acquiring/reacquiring. This creates honest KML gaps.
                if args.no_estimate_during_acquire:
                    valid_estimate=False
                    state='NO_ESTIMATE_ACQUIRE' if last_good is None else 'NO_ESTIMATE_REACQUIRE'
                elif logical:
                    output_step=best; state='ACQUIRE_CANDIDATE'
                else:
                    valid_estimate=False; state='NO_ESTIMATE_ACQUIRE_UNCONFIRMED'
        else:
            # Locked: only dynamic local-region candidates were considered. Never replace anchor on weak/contradictory evidence.
            if logical and best_xy is not None:
                if best_xy == accepted_region:
                    # Use geometry-valid local match as current output and speed calibration.
                    if last_good is not None:
                        dcal=transition_m(last_good,best,origin)
                        if float(flow['mag_px']) >= args.flow_calib_min_px and dcal <= args.flow_calib_max_jump_m:
                            mpp_obs=dcal/max(float(flow['mag_px']),1e-6)
                            if args.min_m_per_px <= mpp_obs <= args.max_m_per_px:
                                if math.isfinite(flow_m_per_px_ema):
                                    flow_m_per_px_ema=(1.0-args.flow_ema_alpha)*flow_m_per_px_ema + args.flow_ema_alpha*mpp_obs
                                else:
                                    flow_m_per_px_ema=mpp_obs
                    last_good=best
                    good_history.append(best)
                    bad_count=0
                    transition_hist.clear()
                    output_step=best; state='LOCKED_REGION'
                else:
                    transition_hist.append((row_region_id(best.reference_row), best_xy, bool(best.geometry_verified), best))
                    reg_votes=Counter(r for r,xy,g,st in transition_hist)
                    geom_votes=Counter(r for r,xy,g,st in transition_hist if g)
                    rid=row_region_id(best.reference_row)
                    if reg_votes[rid] >= args.transition_min_votes and geom_votes[rid] >= 1:
                        accepted_region=best_xy
                        accepted_region_id=rid
                        last_good=best
                        good_history.append(best)
                        bad_count=0
                        output_step=best; state='LOCKED_REGION_TRANSITION_ACCEPTED'
                    else:
                        bad_count += 1
                        if bad_count > args.hold_max_bad_count:
                            valid_estimate=False; output_step=None; state='NO_ESTIMATE_REACQUIRE'
                            accepted_region=None; accepted_region_id=''
                            acquire_hist.clear(); transition_hist.clear()
                        elif last_good is not None:
                            if can_flow_propagate:
                                output_step=clone_step_for_time(last_good,sample_idx,video_time_s,frame_path,'FLOW_PROPAGATED_TRANSITION_HOLD')
                                state='FLOW_PROPAGATED_TRANSITION_HOLD'
                            else:
                                output_step=clone_step_for_time(last_good,sample_idx,video_time_s,frame_path,'LOCKED_REGION_TRANSITION_HOLD')
                                state='LOCKED_REGION_TRANSITION_HOLD'
                        else:
                            valid_estimate=False; state='NO_ESTIMATE_TRANSITION_HOLD'
            else:
                bad_count += 1
                if bad_count > args.hold_max_bad_count:
                    valid_estimate=False; output_step=None; state='NO_ESTIMATE_REACQUIRE'
                    accepted_region=None
                    accepted_region_id=''
                    acquire_hist.clear(); transition_hist.clear()
                elif last_good is not None:
                    if can_flow_propagate:
                        output_step=clone_step_for_time(last_good,sample_idx,video_time_s,frame_path,'FLOW_PROPAGATED_HOLD')
                        state='FLOW_PROPAGATED_HOLD'
                    else:
                        output_step=clone_step_for_time(last_good,sample_idx,video_time_s,frame_path,'HOLD_BAD_LOCAL')
                        state='HOLD_BAD_LOCAL'
                else:
                    valid_estimate=False; state='NO_ESTIMATE_NO_GOOD_HOLD_AVAILABLE'

        extra={
            'accepted_region_id':accepted_region_id,
            'accepted_region_x':accepted_region[0] if accepted_region else '',
            'accepted_region_y':accepted_region[1] if accepted_region else '',
            'search_mode':search_mode,
            'bad_count':bad_count,
            'flow_dx_px':float(flow['dx_px']),
            'flow_dy_px':float(flow['dy_px']),
            'flow_mag_px':float(flow['mag_px']),
            'flow_quality':float(flow['quality']),
            'flow_points':int(flow['points']),
            'flow_m_per_px':active_mpp,
            'flow_speed_m_s':flow_speed_m_s,
            'flow_speed_capped_for_hold_m_s': min(args.max_flow_hold_speed_mps, flow_speed_m_s) if math.isfinite(flow_speed_m_s) else '',
            'flow_speed_cap_m_s': args.max_flow_speed_mps,
            'dynamic_region_ring':dyn_ring,
            'projected_center_offset_frac': getattr(best, 'projected_center_offset_frac', math.nan),
            'landmark_pose_mismatch_warning': int(math.isfinite(getattr(best, 'projected_center_offset_frac', math.nan)) and getattr(best, 'projected_center_offset_frac', math.nan) > args.projected_center_offset_warning_frac),
            'predicted_region_x':predicted_region[0] if predicted_region else '',
            'predicted_region_y':predicted_region[1] if predicted_region else '',
            'candidate_region_id':row_region_id(best.reference_row),
            'candidate_jump_m':jump_m,
            'contradiction':int(contradiction),
            'top_voted_region_id':top_region_id,
            'top_voted_region_score':region_vote_scores.get(top_region_id,'') if top_region_id else '',
            'projected_center_x_frac': getattr(best, 'projected_center_x_frac', math.nan),
            'projected_center_y_frac': getattr(best, 'projected_center_y_frac', math.nan),
            'projected_center_offset_frac': getattr(best, 'projected_center_offset_frac', math.nan),
            'landmark_pose_mismatch_warning': int(math.isfinite(getattr(best, 'projected_center_offset_frac', math.nan)) and getattr(best, 'projected_center_offset_frac', math.nan) > args.projected_center_offset_warning_frac),
            'flow_raw_speed_m_s': flow_speed_m_s_raw,
            'flow_speed_capped_for_hold_m_s': min(args.max_flow_hold_speed_mps, flow_speed_m_s) if math.isfinite(flow_speed_m_s) else '',
            'flow_speed_cap_m_s': args.max_flow_speed_mps,
            'flow_propagated_dx_m':flow_prop_dx_m if state.startswith('FLOW_PROPAGATED') else '',
            'flow_propagated_dy_m':flow_prop_dy_m if state.startswith('FLOW_PROPAGATED') else '',
            'flow_propagated_step_m':math.hypot(flow_prop_dx_m,flow_prop_dy_m) if state.startswith('FLOW_PROPAGATED') else '',
            'valid_estimate':1 if valid_estimate and output_step is not None else 0,
        }
        if valid_estimate and output_step is not None:
            if state.startswith('FLOW_PROPAGATED'):
                row=flow_propagated_row_from_step(output_step,sample_idx,video_time_s,frame_path,state,truth_rows,flow_prop_dx_m,flow_prop_dy_m,extra)
            else:
                row=output_row_from_step(output_step,state,0,float(best.total_increment_cost),truth_rows)
                row.update(extra)
            last_output_step=output_step
        else:
            row=make_no_estimate_row(sample_idx,video_time_s,frame_path,state,truth_rows,extra)
        output_rows.append(row)

        latency_ms=(time.perf_counter()-t0)*1000.0
        latencies.append(latency_ms)
        debug_f.write(json.dumps({
            'sample_index':sample_idx,
            'video_time_s':video_time_s,
            'search_mode':search_mode,
            'accepted_region_id':accepted_region_id,
            'bad_count':bad_count,
            'best_region_id':row_region_id(best.reference_row),
            'top_voted_region_id':top_region_id,
            'best_dataset':best.reference_row.get('dataset_id'),
            'best_frame':best.reference_row.get('frame_count'),
            'best_good':best_good,
            'best_geometry_verified':best.geometry_verified,
            'best_inliers':best.homography_inliers,
            'jump_m':jump_m,
            'contradiction':contradiction,
            'flow_dx_px':float(flow['dx_px']),
            'flow_dy_px':float(flow['dy_px']),
            'flow_mag_px':float(flow['mag_px']),
            'flow_quality':float(flow['quality']),
            'flow_points':int(flow['points']),
            'flow_m_per_px':active_mpp,
            'flow_speed_m_s':flow_speed_m_s,
            'flow_speed_capped_for_hold_m_s': min(args.max_flow_hold_speed_mps, flow_speed_m_s) if math.isfinite(flow_speed_m_s) else '',
            'flow_speed_cap_m_s': args.max_flow_speed_mps,
            'dynamic_region_ring':dyn_ring,
            'projected_center_offset_frac': getattr(best, 'projected_center_offset_frac', math.nan),
            'landmark_pose_mismatch_warning': int(math.isfinite(getattr(best, 'projected_center_offset_frac', math.nan)) and getattr(best, 'projected_center_offset_frac', math.nan) > args.projected_center_offset_warning_frac),
            'predicted_region':predicted_region,
            'candidate_count':len(candidates),
            'valid_estimate':int(row.get('valid_estimate',1)),
            'latency_ms':latency_ms,
        })+'\n')
        sample_idx += 1
        pbar.update(1)
    cap.release(); pbar.close(); debug_f.close()
    write_csv(args.output_csv, output_rows)
    summary=summarize(output_rows, latencies)
    no_est=sum(1 for r in output_rows if str(r.get('valid_estimate','1')) == '0')
    fresh_states={'ACQUIRE_ACCEPTED_REGION','LOCKED_REGION','LOCKED_REGION_TRANSITION_ACCEPTED'}
    flow_prop_rows=[r for r in output_rows if str(r.get('state','')).startswith('FLOW_PROPAGATED')]
    static_hold_rows=[r for r in output_rows if 'HOLD' in str(r.get('state','')) and not str(r.get('state','')).startswith('FLOW_PROPAGATED')]
    fresh_rows=[r for r in output_rows if str(r.get('state','')) in fresh_states]
    def subset_metrics(prefix: str, rows_subset: list[dict[str,Any]]) -> dict[str,Any]:
        vals=np.array([float(r['position_error_m']) for r in rows_subset if r.get('position_error_m','') not in ('',None)], dtype=float)
        out={prefix+'_frames':len(rows_subset), prefix+'_evaluated_frames':int(len(vals))}
        if len(vals):
            out.update({
                prefix+'_mean_error_m':float(vals.mean()),
                prefix+'_median_error_m':float(np.median(vals)),
                prefix+'_p90_error_m':float(np.percentile(vals,90)),
                prefix+'_p95_error_m':float(np.percentile(vals,95)),
                prefix+'_pct_under_100m':float((vals < 100).mean()*100.0),
            })
        return out
    summary.update({
        'search_mode_counts':dict(mode_counts),
        'no_estimate_frames':no_est,
        'valid_estimate_frames':len(output_rows)-no_est,
        'fresh_geometry_frames':len(fresh_rows),
        'flow_propagated_frames':len(flow_prop_rows),
        'static_hold_frames':len(static_hold_rows),
    })
    summary.update(subset_metrics('fresh_only', fresh_rows))
    summary.update(subset_metrics('flow_propagated_only', flow_prop_rows))
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(f'wrote: {args.output_csv}')
    print(f'wrote: {args.summary_json}')
    print(f'wrote: {args.debug_jsonl}')

if __name__ == '__main__':
    main()
