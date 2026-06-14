import argparse
import csv
from pathlib import Path
from PIL import Image

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def center_crop_resize(image: Image.Image, crop_ratio: float) -> Image.Image:
    w, h = image.size

    crop_ratio = max(0.05, min(1.0, crop_ratio))

    crop_w = int(round(w * crop_ratio))
    crop_h = int(round(h * crop_ratio))

    crop_w = max(32, min(w, crop_w))
    crop_h = max(32, min(h, crop_h))

    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((w, h), RESAMPLE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-frames-dir", required=True, type=Path)

    parser.add_argument("--target-alt-m", type=float, required=True)
    parser.add_argument("--min-crop-ratio", type=float, default=0.18)
    parser.add_argument("--max-crop-ratio", type=float, default=0.60)
    parser.add_argument("--jpeg-quality", type=int, default=92)

    args = parser.parse_args()

    args.output_frames_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    with args.input_manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No rows found in {args.input_manifest}")

    input_fields = list(rows[0].keys())
    extra_fields = [
        "scale_target_alt_m",
        "scale_reference_alt_m",
        "scale_crop_ratio",
        "scale_source_frame_path",
    ]

    fieldnames = input_fields + [x for x in extra_fields if x not in input_fields]

    written = 0

    with args.output_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            src_path = Path(row["frame_path"])

            if not src_path.exists():
                print(f"WARNING missing frame: {src_path}")
                continue

            ref_alt = safe_float(row.get("rel_alt_m"), default=0.0)

            if ref_alt <= 1.0:
                crop_ratio = args.max_crop_ratio
            else:
                crop_ratio = args.target_alt_m / ref_alt

            crop_ratio = max(args.min_crop_ratio, min(args.max_crop_ratio, crop_ratio))

            out_name = f"{src_path.stem}_scale{int(args.target_alt_m)}m_r{crop_ratio:.3f}.jpg"
            out_path = args.output_frames_dir / out_name

            with Image.open(src_path) as img:
                img = img.convert("RGB")
                scaled = center_crop_resize(img, crop_ratio)
                scaled.save(out_path, quality=args.jpeg_quality)

            out_row = dict(row)
            out_row["frame_path"] = str(out_path)
            out_row["scale_target_alt_m"] = f"{args.target_alt_m:.3f}"
            out_row["scale_reference_alt_m"] = f"{ref_alt:.3f}"
            out_row["scale_crop_ratio"] = f"{crop_ratio:.6f}"
            out_row["scale_source_frame_path"] = str(src_path)

            writer.writerow(out_row)
            written += 1

    print(f"input_rows: {len(rows)}")
    print(f"written_rows: {written}")
    print(f"wrote_manifest: {args.output_manifest}")
    print(f"wrote_frames_dir: {args.output_frames_dir}")


if __name__ == "__main__":
    main()