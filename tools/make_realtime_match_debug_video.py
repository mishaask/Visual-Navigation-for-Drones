"""Create a realtime match debug video with LightGlue connecting lines.

For each realtime prediction row, this script re-runs SuperPoint+LightGlue on the
saved query frame and the selected reference frame, then draws keypoint match
lines between the two images.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image
from tqdm import tqdm


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_bgr(img: torch.Tensor) -> np.ndarray:
    # LightGlue load_image returns C,H,W float RGB in [0,1].
    arr = img.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def draw_text_box(img: np.ndarray, lines: list[str]) -> None:
    x, y = 10, 24
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output-video", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--image-resize", type=int, default=768)
    ap.add_argument("--max-keypoints", type=int, default=1024)
    ap.add_argument("--max-lines", type=int, default=80)
    args = ap.parse_args()

    rows = read_csv(args.predictions)
    rows = [r for i, r in enumerate(rows) if i % max(args.every, 1) == 0]
    device = choose_device()
    print(f"device: {device}")
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    for row in tqdm(rows, desc="debug video"):
        q_path = Path(row["query_frame_path"])
        r_path = Path(row["reference_frame_path"])
        if not q_path.exists() or not r_path.exists():
            continue

        img0_t = load_image(q_path, resize=args.image_resize).to(device)
        img1_t = load_image(r_path, resize=args.image_resize).to(device)
        with torch.no_grad():
            feats0 = extractor.extract(img0_t)
            feats1 = extractor.extract(img1_t)
            matches01 = matcher({"image0": feats0, "image1": feats1})
        matches = matches01["matches"][0].detach().cpu().numpy()
        scores = matches01.get("scores")
        if scores is not None:
            scores_np = scores[0].detach().cpu().numpy()
            order = np.argsort(scores_np)[::-1]
            matches = matches[order]
        matches = matches[: args.max_lines]

        k0 = feats0["keypoints"][0].detach().cpu().numpy()
        k1 = feats1["keypoints"][0].detach().cpu().numpy()
        img0 = tensor_to_bgr(img0_t)
        img1 = tensor_to_bgr(img1_t)
        h = max(img0.shape[0], img1.shape[0])
        w = img0.shape[1] + img1.shape[1]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: img0.shape[0], : img0.shape[1]] = img0
        canvas[: img1.shape[0], img0.shape[1] : img0.shape[1] + img1.shape[1]] = img1
        offset_x = img0.shape[1]

        for m in matches:
            p0 = k0[int(m[0])]
            p1 = k1[int(m[1])]
            x0, y0 = int(p0[0]), int(p0[1])
            x1, y1 = int(p1[0]) + offset_x, int(p1[1])
            cv2.line(canvas, (x0, y0), (x1, y1), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (x0, y0), 2, (0, 255, 0), -1)
            cv2.circle(canvas, (x1, y1), 2, (0, 0, 255), -1)

        lines = [
            f"query {row.get('query_sample_index')} t={row.get('query_video_time_s')}s",
            f"ref {row.get('reference_dataset')} frame {row.get('reference_frame_count')} rank {row.get('rank')}",
            f"err={row.get('position_error_m','')}m  inliers={row.get('lg_inlier_count','')} ratio={row.get('lg_inlier_ratio','')}",
        ]
        draw_text_box(canvas, lines)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(args.output_video), fourcc, args.fps, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)

    if writer is not None:
        writer.release()
    print(f"wrote: {args.output_video}")


if __name__ == "__main__":
    main()
