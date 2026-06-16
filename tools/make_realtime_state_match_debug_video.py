"""Create per-frame realtime localization debug video and explanations CSV.

The video shows the query frame next to the final chosen reference frame and
red/green LightGlue match lines. It also overlays why the pipeline output that
state: fresh match, static hold, flow-propagated hold, or no-estimate.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lightglue import LightGlue, SuperPoint  # noqa: E402
from lightglue.utils import load_image  # noqa: E402
from anyloc_dino_retrieval import choose_device  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def safe_float(v: Any, default: float = math.nan) -> float:
    try:
        if v in (None, ""):
            return default
        return float(v)
    except Exception:
        return default


def resize_longest_bgr(path: Path, max_size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not read image: {path}")
    h, w = img.shape[:2]
    scale = float(max_size) / max(h, w) if max(h, w) > max_size else 1.0
    if scale != 1.0:
        img = cv2.resize(img, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_AREA)
    return img


def pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    pad = np.zeros((height - img.shape[0], img.shape[1], 3), dtype=np.uint8)
    return np.vstack([img, pad])


def explain_state(row: dict[str, str]) -> str:
    state = str(row.get("state", ""))
    if state == "LOCKED_REGION":
        return "fresh local match accepted in locked region"
    if state == "LOCKED_REGION_TRANSITION_ACCEPTED":
        return "fresh transition to neighboring region accepted"
    if state == "ACQUIRE_ACCEPTED_REGION":
        return "global/reacquire region confirmed and accepted"
    if state.startswith("FLOW_PROPAGATED"):
        return "no fresh trusted match; propagated last accepted estimate using optical flow"
    if state in ("HOLD_BAD_LOCAL", "LOCKED_REGION_TRANSITION_HOLD"):
        return "local match failed/transition unconfirmed; reused previous accepted estimate"
    if state.startswith("NO_ESTIMATE"):
        return "no trusted estimate; KML should show a gap"
    return "unclassified pipeline state"


def match_lightglue(query_path: Path, ref_path: Path, extractor: SuperPoint, matcher: LightGlue, device: torch.device, resize: int, ransac: float) -> dict[str, Any]:
    out = {"matches": 0, "inliers": 0, "ratio": 0.0, "pts0": np.empty((0,2)), "pts1": np.empty((0,2)), "mask": np.zeros((0,), dtype=bool)}
    if not query_path.exists() or not ref_path.exists():
        return out
    try:
        image0 = load_image(query_path, resize=resize).to(device)
        image1 = load_image(ref_path, resize=resize).to(device)
        with torch.no_grad():
            feats0 = extractor.extract(image0)
            feats1 = extractor.extract(image1)
            matches01 = matcher({"image0": feats0, "image1": feats1})
        matches = matches01["matches"][0].detach().cpu().numpy()
        kpts0 = feats0["keypoints"][0].detach().cpu().numpy()
        kpts1 = feats1["keypoints"][0].detach().cpu().numpy()
        if len(matches) == 0:
            return out
        pts0 = kpts0[matches[:,0]].astype(np.float32)
        pts1 = kpts1[matches[:,1]].astype(np.float32)
        mask = np.zeros((len(matches),), dtype=bool)
        if len(matches) >= 4:
            H, inmask = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac)
            if inmask is not None:
                mask = inmask.ravel().astype(bool)
        out.update({"matches": int(len(matches)), "inliers": int(mask.sum()), "ratio": float(mask.mean()) if len(mask) else 0.0, "pts0": pts0, "pts1": pts1, "mask": mask})
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def draw_panel(row: dict[str,str], q_img: np.ndarray, r_img: np.ndarray | None, lg: dict[str,Any], max_lines: int) -> np.ndarray:
    if r_img is None:
        r_img = np.zeros_like(q_img)
    h = max(q_img.shape[0], r_img.shape[0])
    q = pad_to_height(q_img, h)
    r = pad_to_height(r_img, h)
    panel = np.hstack([q, r])
    xoff = q.shape[1]
    pts0 = lg.get("pts0", np.empty((0,2)))
    pts1 = lg.get("pts1", np.empty((0,2)))
    mask = lg.get("mask", np.zeros((0,), dtype=bool))
    n = len(pts0)
    if n:
        # Prefer drawing inliers first; cap total lines to keep video readable.
        order = list(np.where(mask)[0]) + list(np.where(~mask)[0])
        for idx in order[:max_lines]:
            p0 = tuple(np.round(pts0[idx]).astype(int))
            p1 = tuple(np.round(pts1[idx]).astype(int) + np.array([xoff, 0]))
            color = (0, 220, 0) if bool(mask[idx]) else (0, 0, 255)
            cv2.line(panel, p0, p1, color, 1, cv2.LINE_AA)
            cv2.circle(panel, p0, 2, color, -1)
            cv2.circle(panel, p1, 2, color, -1)
    overlay = panel.copy()
    cv2.rectangle(overlay, (0,0), (panel.shape[1], 155), (0,0,0), -1)
    panel = cv2.addWeighted(overlay, 0.62, panel, 0.38, 0)
    lines = [
        f"sample={row.get('query_sample_index')} t={row.get('query_video_time_s')} state={row.get('state')}",
        f"why: {explain_state(row)}",
        f"ref={row.get('reference_dataset')} frame={row.get('reference_frame_count')} search={row.get('search_mode')} bad={row.get('bad_count')} ring={row.get('dynamic_region_ring')}",
        f"err={row.get('position_error_m')} drone_err={row.get('drone_position_error_m')} dino={row.get('dino_similarity')} geom={row.get('geometry_verified')}",
        f"csv_inliers={row.get('homography_inliers')} fresh_LG_inliers={lg.get('inliers',0)}/{lg.get('matches',0)} ratio={lg.get('ratio',0):.3f}",
        f"flow_speed={row.get('flow_speed_m_s')} flow_step={row.get('flow_propagated_step_m')} valid={row.get('valid_estimate','1')}",
    ]
    y=22
    for text in lines:
        cv2.putText(panel, text[:180], (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        y += 22
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output-video", type=Path, required=True)
    ap.add_argument("--explanations-csv", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--image-resize", type=int, default=768)
    ap.add_argument("--max-keypoints", type=int, default=1024)
    ap.add_argument("--max-lines", type=int, default=120)
    ap.add_argument("--ransac-reproj-threshold", type=float, default=5.0)
    args = ap.parse_args()

    rows = read_csv(args.predictions)
    selected=[]
    for r in rows:
        idx = int(float(r.get("query_sample_index", -1)))
        if idx < args.start: continue
        if args.end >= 0 and idx > args.end: continue
        if (idx - args.start) % max(1,args.every) != 0: continue
        selected.append(r)
    if not selected:
        raise RuntimeError("no rows selected")

    device = choose_device()
    print(f"device: {device}")
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    if args.frames_dir:
        args.frames_dir.mkdir(parents=True, exist_ok=True)

    writer=None
    explanation_rows=[]
    for r in tqdm(selected, desc="state match debug"):
        qpath = Path(r.get("query_frame_path", ""))
        rpath = Path(r.get("reference_frame_path", ""))
        if not qpath.exists():
            q_img = np.zeros((432,768,3), dtype=np.uint8)
            cv2.putText(q_img, f"missing query: {qpath}", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            q_img = resize_longest_bgr(qpath, args.image_resize)
        ref_img=None
        lg={"matches":0,"inliers":0,"ratio":0.0,"pts0":np.empty((0,2)),"pts1":np.empty((0,2)),"mask":np.zeros((0,),dtype=bool)}
        if rpath.exists() and str(r.get("valid_estimate","1")) != "0":
            ref_img = resize_longest_bgr(rpath, args.image_resize)
            lg = match_lightglue(qpath, rpath, extractor, matcher, device, args.image_resize, args.ransac_reproj_threshold)
        panel = draw_panel(r, q_img, ref_img, lg, args.max_lines)
        if writer is None:
            h,w=panel.shape[:2]
            writer=cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w,h))
        writer.write(panel)
        if args.frames_dir:
            idx=int(float(r.get("query_sample_index",0)))
            cv2.imwrite(str(args.frames_dir / f"debug_{idx:06d}_{r.get('state','state')}.jpg"), panel)
        explanation_rows.append({
            "query_sample_index": r.get("query_sample_index"),
            "video_time_s": r.get("query_video_time_s"),
            "state": r.get("state"),
            "why": explain_state(r),
            "reference_dataset": r.get("reference_dataset"),
            "reference_frame_count": r.get("reference_frame_count"),
            "reference_frame_path": r.get("reference_frame_path"),
            "search_mode": r.get("search_mode"),
            "bad_count": r.get("bad_count"),
            "dynamic_region_ring": r.get("dynamic_region_ring"),
            "position_error_m": r.get("position_error_m"),
            "drone_position_error_m": r.get("drone_position_error_m"),
            "dino_similarity": r.get("dino_similarity"),
            "csv_geometry_verified": r.get("geometry_verified"),
            "csv_homography_inliers": r.get("homography_inliers"),
            "fresh_lightglue_matches": lg.get("matches",0),
            "fresh_lightglue_inliers": lg.get("inliers",0),
            "fresh_lightglue_inlier_ratio": lg.get("ratio",0.0),
            "flow_speed_m_s": r.get("flow_speed_m_s"),
            "flow_propagated_step_m": r.get("flow_propagated_step_m"),
            "valid_estimate": r.get("valid_estimate","1"),
        })
    if writer is not None:
        writer.release()
    write_csv(args.explanations_csv, explanation_rows)
    print(f"wrote: {args.output_video}")
    print(f"wrote: {args.explanations_csv}")
    if args.frames_dir:
        print(f"wrote frames: {args.frames_dir}")


if __name__ == "__main__":
    main()
