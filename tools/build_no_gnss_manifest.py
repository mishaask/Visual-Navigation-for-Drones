import argparse
import csv
from pathlib import Path

FIELDS = [
    "frame_count",
    "frame_path",
    "timestamp",
    "start_seconds",
    "drone_latitude",
    "drone_longitude",
    "rel_alt_m",
    "heading_deg",
    "heading_source",
    "camera_angle_deg",
    "camera_angle_source",
    "ground_latitude",
    "ground_longitude",
]

def frame_number_from_name(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0

def fmt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3600000
    total_ms %= 3600000
    m = total_ms // 60000
    total_ms %= 60000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()

    frames = sorted(args.frames_dir.glob("*.jpg"))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for frame_path in frames:
            frame_index = frame_number_from_name(frame_path)
            seconds = (frame_index - 1) / args.fps

            writer.writerow({
                "frame_count": frame_index,
                "frame_path": str(frame_path),
                "timestamp": fmt_timestamp(seconds),
                "start_seconds": f"{seconds:.3f}",
                "drone_latitude": "0.0",
                "drone_longitude": "0.0",
                "rel_alt_m": "0.0",
                "heading_deg": "0.0",
                "heading_source": "none",
                "camera_angle_deg": "0.0",
                "camera_angle_source": "none",
                "ground_latitude": "0.0",
                "ground_longitude": "0.0",
            })

    print(f"manifest_rows: {len(frames)}")
    print(f"wrote: {args.output_csv}")

if __name__ == "__main__":
    main()