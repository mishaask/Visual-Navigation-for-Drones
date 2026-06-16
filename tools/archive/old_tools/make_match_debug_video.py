import argparse
import csv
from pathlib import Path

import cv2


def pick_prefix(fieldnames):
    if "motion_viterbi_reference_frame_path" in fieldnames:
        return "motion_viterbi"
    if "temporal_reference_frame_path" in fieldnames:
        return "temporal"
    raise RuntimeError(
        "Could not find supported reference frame path column. "
        "Expected motion_viterbi_reference_frame_path or temporal_reference_frame_path."
    )


def safe_float(row, key, default=None):
    try:
        value = row.get(key, "")
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def resize_keep_aspect(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    canvas = cv2.copyMakeBorder(
        resized,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return canvas


def put_text(img, text, x, y, scale=0.65, color=(255, 255, 255), thickness=2):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def basename(path_text):
    if not path_text:
        return ""
    return Path(path_text.replace("\\", "/")).name


def load_image(path_text):
    path = Path(path_text)
    img = cv2.imread(str(path))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--panel-height", type=int, default=540)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--query-label", default="QUERY")
    args = parser.parse_args()

    fieldnames, rows = read_rows(args.results)
    prefix = pick_prefix(fieldnames)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    out_w = args.panel_width * 2
    out_h = args.panel_height + 120

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (out_w, out_h))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {args.output}")

    written = 0
    missing = 0

    for idx, row in enumerate(rows):
        if args.every > 1 and idx % args.every != 0:
            continue

        if args.limit and written >= args.limit:
            break

        query_path = row.get("query_frame_path", "")
        ref_path = row.get(f"{prefix}_reference_frame_path", "")

        query_img = load_image(query_path)
        ref_img = load_image(ref_path)

        if query_img is None or ref_img is None:
            missing += 1
            continue

        query_panel = resize_keep_aspect(query_img, args.panel_width, args.panel_height)
        ref_panel = resize_keep_aspect(ref_img, args.panel_width, args.panel_height)

        frame = cv2.hconcat([query_panel, ref_panel])
        frame = cv2.copyMakeBorder(
            frame,
            top=0,
            bottom=120,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=(20, 20, 20),
        )

        q_frame = row.get("query_frame_count", "")
        ref_dataset = row.get(f"{prefix}_reference_dataset", "")
        ref_frame = row.get(f"{prefix}_reference_frame_count", "")
        rank = row.get(f"{prefix}_rank", "")

        sim = safe_float(row, f"{prefix}_dino_similarity")
        matches = row.get("lg_match_count", "")
        inliers = row.get("lg_inlier_count", "")
        ratio = safe_float(row, "lg_inlier_ratio")
        err = safe_float(row, f"{prefix}_position_error_m")

        sim_text = "NA" if sim is None else f"{sim:.3f}"
        ratio_text = "NA" if ratio is None else f"{ratio:.3f}"
        metric_text = f"rank={rank}  sim={sim_text}"

        put_text(frame, f"QUERY: {args.query_label}", 20, 34)
        put_text(frame, "MATCHED REFERENCE", args.panel_width + 20, 34)

        put_text(
            frame,
            f"query_frame={q_frame}  query_file={basename(query_path)}",
            20,
            args.panel_height + 35,
            scale=0.6,
        )

        put_text(
            frame,
            f"ref_dataset={ref_dataset}  ref_frame={ref_frame}  ref_file={basename(ref_path)}",
            20,
            args.panel_height + 65,
            scale=0.6,
        )

        put_text(
            frame,
            f"{metric_text}  matches={matches}  inliers={inliers}  ratio={ratio_text}",
            20,
            args.panel_height + 95,
            scale=0.6,
        )

        # Only show known error for real SRT evaluation runs.
        # No-GNSS dummy errors are around 4,800,000m, so hide those.
        if err is not None and err < 10000:
            color = (80, 255, 80) if err < 50 else (0, 200, 255) if err < 150 else (0, 0, 255)
            put_text(
                frame,
                f"known-error={err:.1f} m",
                args.panel_width + 20,
                args.panel_height + 95,
                scale=0.6,
                color=color,
            )

        writer.write(frame)
        written += 1

    writer.release()

    print(f"results: {args.results}")
    print(f"mode: {prefix}")
    print(f"written_frames: {written}")
    print(f"missing_frames: {missing}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()