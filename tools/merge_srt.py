import argparse
import re
import subprocess
from pathlib import Path
from datetime import timedelta

TIME_RE = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s+-->\s+(\d\d):(\d\d):(\d\d),(\d\d\d)")

def parse_time(h, m, s, ms):
    return timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms))

def fmt_time(td):
    total_ms = int(round(td.total_seconds() * 1000))
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def video_duration(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1",
        str(path)
    ], text=True).strip()
    return timedelta(seconds=float(out))

def shift_srt_text(text, offset, start_index):
    blocks = re.split(r"\n\s*\n", text.strip(), flags=re.MULTILINE)
    output = []
    idx = start_index

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            continue

        time_line_i = None
        for i, line in enumerate(lines):
            if "-->" in line:
                time_line_i = i
                break

        if time_line_i is None:
            continue

        match = TIME_RE.search(lines[time_line_i])
        if not match:
            continue

        start = parse_time(*match.groups()[0:4]) + offset
        end = parse_time(*match.groups()[4:8]) + offset

        lines[0] = str(idx)
        lines[time_line_i] = f"{fmt_time(start)} --> {fmt_time(end)}"
        output.append("\n".join(lines))
        idx += 1

    return output, idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="video.mp4:srt.srt pairs")
    args = ap.parse_args()

    all_blocks = []
    offset = timedelta(0)
    next_index = 1

    for pair in args.pairs:
        video_s, srt_s = pair.split(":", 1)
        video = Path(video_s)
        srt = Path(srt_s)

        text = srt.read_text(encoding="utf-8", errors="ignore")
        blocks, next_index = shift_srt_text(text, offset, next_index)
        all_blocks.extend(blocks)

        offset += video_duration(video)

    Path(args.out).write_text("\n\n".join(all_blocks) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()