from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EARTH_RADIUS_M = 6_378_137.0

TIME_RE = re.compile(r"(?P<h>\d\d):(?P<m>\d\d):(?P<s>\d\d),(?P<ms>\d{3})")
FRAME_RE = re.compile(r"FrameCnt:\s*(\d+)")
FLOAT_RE = r"([-+]?\d+(?:\.\d+)?)"
YAW_PATTERNS = [
    r"\byaw\s*[:=]\s*" + FLOAT_RE,
    r"\bflight_yaw\s*[:=]\s*" + FLOAT_RE,
    r"\bflight_yaw_degree\s*[:=]\s*" + FLOAT_RE,
    r"\bgimbal_yaw\s*[:=]\s*" + FLOAT_RE,
    r"\bgimbal_yaw_degree\s*[:=]\s*" + FLOAT_RE,
    r"\bcamera_yaw\s*[:=]\s*" + FLOAT_RE,
    r"\bheading\s*[:=]\s*" + FLOAT_RE,
    r"\bcompass\s*[:=]\s*" + FLOAT_RE,
]


@dataclass
class Sample:
    index: int
    frame: int
    start_s: float
    end_s: float
    lat: float | None
    lon: float | None
    rel_alt: float | None
    raw: str


def _time_to_s(text: str) -> float:
    m = TIME_RE.search(text)
    if not m:
        raise ValueError(f"bad SRT time: {text!r}")
    return int(m.group('h')) * 3600 + int(m.group('m')) * 60 + int(m.group('s')) + int(m.group('ms')) / 1000.0


def _float_key(text: str, key: str) -> float | None:
    m = re.search(rf"\b{re.escape(key)}\s*:\s*{FLOAT_RE}", text)
    return float(m.group(1)) if m else None


def parse_srt(path: Path) -> list[Sample]:
    text = path.read_text(encoding='utf-8', errors='replace')
    blocks = re.split(r"\n\s*\n", text.strip())
    out: list[Sample] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2 or '-->' not in lines[1]:
            continue
        try:
            idx = int(lines[0])
        except ValueError:
            idx = len(out) + 1
        try:
            a, b = [x.strip() for x in lines[1].split('-->', 1)]
            start_s, end_s = _time_to_s(a), _time_to_s(b)
        except Exception:
            continue
        raw = '\n'.join(lines[2:])
        fm = FRAME_RE.search(raw)
        frame = int(fm.group(1)) if fm else idx
        out.append(Sample(
            index=idx,
            frame=frame,
            start_s=start_s,
            end_s=end_s,
            lat=_float_key(raw, 'latitude'),
            lon=_float_key(raw, 'longitude'),
            rel_alt=_float_key(raw, 'rel_alt'),
            raw=raw,
        ))
    return out


def valid(samples: list[Sample]) -> list[Sample]:
    return [s for s in samples if s.lat is not None and s.lon is not None and abs(s.lat) > 1e-9 and abs(s.lon) > 1e-9]


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def destination(lat: float, lon: float, brg_deg: float, dist_m: float) -> tuple[float, float]:
    ang = dist_m / EARTH_RADIUS_M
    brg = math.radians(brg_deg)
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    sp2 = math.sin(p1) * math.cos(ang) + math.cos(p1) * math.sin(ang) * math.cos(brg)
    p2 = math.asin(max(-1.0, min(1.0, sp2)))
    y = math.sin(brg) * math.sin(ang) * math.cos(p1)
    x = math.cos(ang) - math.sin(p1) * math.sin(p2)
    l2 = l1 + math.atan2(y, x)
    return math.degrees(p2), (math.degrees(l2) + 540.0) % 360.0 - 180.0


def angular_mean(vals: list[float]) -> float:
    if not vals:
        return math.nan
    sx = sum(math.sin(math.radians(v)) for v in vals)
    cx = sum(math.cos(math.radians(v)) for v in vals)
    return (math.degrees(math.atan2(sx, cx)) + 360.0) % 360.0


def nearest_by_time(samples: list[Sample], t: float) -> Sample:
    return min(samples, key=lambda s: abs(s.start_s - t))


def find_heading(samples: list[Sample], idx: int, min_step_m: float = 0.5) -> tuple[float, float]:
    cur = samples[idx]
    # Prefer a symmetric local derivative around the sample.
    for radius in [1, 2, 5, 10, 20, 30]:
        a = samples[max(0, idx - radius)]
        b = samples[min(len(samples) - 1, idx + radius)]
        if a.lat is None or a.lon is None or b.lat is None or b.lon is None:
            continue
        d = haversine_m(a.lat, a.lon, b.lat, b.lon)
        dt = max(1e-6, b.start_s - a.start_s)
        if d >= min_step_m:
            return bearing_deg(a.lat, a.lon, b.lat, b.lon), d / dt
    return math.nan, 0.0


def ground_distance(alt_m: float, angle_deg: float, convention: str) -> float:
    alt = abs(float(alt_m))
    a = math.radians(angle_deg)
    if convention == 'from-horizon':
        return alt / max(1e-6, math.tan(a))
    if convention == 'from-nadir':
        return alt * math.tan(a)
    raise ValueError(f'bad convention: {convention}')


def inspect_yaw_fields(samples: list[Sample]) -> dict[str, Any]:
    found: dict[str, int] = {}
    examples: dict[str, str] = {}
    for s in samples[: min(len(samples), 5000)]:
        low = s.raw.lower()
        for pat in YAW_PATTERNS:
            m = re.search(pat, low)
            if m:
                key = pat.split('\\b')[1].split('\\s')[0] if '\\b' in pat else pat
                found[key] = found.get(key, 0) + 1
                examples.setdefault(key, m.group(0))
    return {'fields_found': found, 'examples': examples, 'has_direct_yaw': bool(found)}


def write_kml(path: Path, rows: list[dict[str, Any]], methods: list[str], angles: list[float]) -> None:
    colors = ['ff0000ff', 'ff00ff00', 'ffff0000', 'ff00ffff', 'ffff00ff', 'ffffff00', 'ff8888ff', 'ff88ff88']
    def coords(points):
        return ' '.join(f"{p['look_lon']},{p['look_lat']},0" for p in points if p.get('look_lat') not in ('', None))
    def drone_coords():
        seen=[]
        last=None
        for r in rows:
            k=r['sample_index']
            if k == last: continue
            last=k
            seen.append(f"{r['drone_lon']},{r['drone_lat']},0")
        return ' '.join(seen)
    out=['<?xml version="1.0" encoding="UTF-8"?>','<kml xmlns="http://www.opengis.net/kml/2.2"><Document>','<name>Projection audit</name>']
    out.append('<Style id="drone"><LineStyle><color>ff00ff00</color><width>4</width></LineStyle></Style>')
    out.append('<Placemark><name>Drone GPS path</name><styleUrl>#drone</styleUrl><LineString><tessellate>1</tessellate><coordinates>'+drone_coords()+'</coordinates></LineString></Placemark>')
    ci=0
    for method in methods:
        for angle in angles:
            pts=[r for r in rows if r['heading_method']==method and float(r['angle_deg'])==float(angle)]
            style=f's{ci}'
            color=colors[ci%len(colors)]
            out.append(f'<Style id="{style}"><LineStyle><color>{color}</color><width>2</width></LineStyle></Style>')
            out.append(f'<Placemark><name>{method} angle={angle:g}</name><styleUrl>#{style}</styleUrl><LineString><tessellate>1</tessellate><coordinates>{coords(pts)}</coordinates></LineString></Placemark>')
            ci+=1
    out.append('</Document></kml>')
    path.write_text('\n'.join(out), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='Audit look-at projection assumptions from DJI SRT telemetry.')
    ap.add_argument('--srt', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--sample-fps', type=float, default=1.0)
    ap.add_argument('--heading-smooth-window', type=int, default=7)
    ap.add_argument('--speed-threshold-mps', type=float, default=1.5)
    ap.add_argument('--angles-deg', nargs='+', type=float, default=[35,45,60,75])
    ap.add_argument('--convention', choices=['from-horizon','from-nadir'], default='from-horizon')
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_all = parse_srt(args.srt)
    samples = valid(samples_all)
    if not samples:
        raise RuntimeError('no valid GPS samples in SRT')
    field_report = inspect_yaw_fields(samples_all)

    duration = samples[-1].start_s
    sample_times = [i / args.sample_fps for i in range(int(duration * args.sample_fps) + 1)]
    nearest = [nearest_by_time(samples, t) for t in sample_times]
    idxs = [samples.index(s) for s in nearest]
    raw_headings=[]; speeds=[]
    for idx in idxs:
        h, sp = find_heading(samples, idx)
        raw_headings.append(h); speeds.append(sp)
    # Smoothed circular heading.
    smooth=[]
    half=max(0,args.heading_smooth_window//2)
    for i in range(len(raw_headings)):
        vals=[h for h in raw_headings[max(0,i-half):min(len(raw_headings),i+half+1)] if math.isfinite(h)]
        smooth.append(angular_mean(vals))
    # Speed-gated heading: update only when movement is reliable.
    speed_gate=[]; last=next((h for h in smooth if math.isfinite(h)), 0.0)
    for h, sp in zip(smooth, speeds):
        if math.isfinite(h) and sp >= args.speed_threshold_mps:
            last=h
        speed_gate.append(last)

    methods = ['raw_course', f'smoothed_w{args.heading_smooth_window}', f'speed_gated_{args.speed_threshold_mps:g}mps']
    heading_by_method = {
        methods[0]: raw_headings,
        methods[1]: smooth,
        methods[2]: speed_gate,
    }

    out_rows=[]
    for sample_i, (t, s) in enumerate(zip(sample_times, nearest)):
        if s.lat is None or s.lon is None: continue
        alt = s.rel_alt if s.rel_alt is not None else 0.0
        for method in methods:
            h = heading_by_method[method][sample_i]
            if not math.isfinite(h):
                continue
            for angle in args.angles_deg:
                gd = ground_distance(alt, angle, args.convention)
                ll = destination(s.lat, s.lon, h, gd)
                out_rows.append({
                    'sample_index': sample_i,
                    'time_s': round(t,3),
                    'srt_frame': s.frame,
                    'drone_lat': s.lat,
                    'drone_lon': s.lon,
                    'rel_alt_m': alt,
                    'speed_mps': speeds[sample_i],
                    'heading_method': method,
                    'heading_deg': h,
                    'angle_deg': angle,
                    'angle_convention': args.convention,
                    'ground_distance_m': gd,
                    'look_lat': ll[0],
                    'look_lon': ll[1],
                })

    csv_path=args.output_dir/'projection_assumptions.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    kml_path=args.output_dir/'projection_assumptions.kml'
    write_kml(kml_path, out_rows, methods, args.angles_deg)
    (args.output_dir/'srt_field_report.json').write_text(json.dumps({
        'srt': str(args.srt),
        'rows_total': len(samples_all),
        'valid_gps_rows': len(samples),
        'direct_yaw_or_gimbal_yaw_fields_found': field_report['fields_found'],
        'examples': field_report['examples'],
        'has_direct_camera_look_direction': field_report['has_direct_yaw'],
        'fallback': 'course-over-ground from GPS deltas',
        'warning': 'course-over-ground is drone movement direction, not guaranteed camera/gimbal look direction',
    }, indent=2), encoding='utf-8')

    # Estimate projection uncertainty relative to raw_course at 45 deg when available.
    base = {(r['sample_index']):(float(r['look_lat']), float(r['look_lon'])) for r in out_rows if r['heading_method']=='raw_course' and abs(float(r['angle_deg'])-45.0)<1e-6}
    uncert={}
    for method in methods:
        for angle in args.angles_deg:
            vals=[]
            for r in out_rows:
                if r['heading_method']==method and abs(float(r['angle_deg'])-angle)<1e-6 and r['sample_index'] in base:
                    b=base[r['sample_index']]
                    vals.append(haversine_m(b[0],b[1],float(r['look_lat']),float(r['look_lon'])))
            if vals:
                vals=sorted(vals)
                uncert[f'{method}_angle_{angle:g}_vs_raw45_m']={
                    'median': vals[len(vals)//2],
                    'p90': vals[int(0.9*(len(vals)-1))],
                    'max': vals[-1],
                }
    summary={
        'sample_count': len(sample_times),
        'assumption_rows': len(out_rows),
        'methods': methods,
        'angles_deg': args.angles_deg,
        'angle_convention': args.convention,
        'speed_threshold_mps': args.speed_threshold_mps,
        'heading_smooth_window': args.heading_smooth_window,
        'direct_yaw_or_gimbal_yaw_fields_found': field_report['fields_found'],
        'has_direct_camera_look_direction': field_report['has_direct_yaw'],
        'projection_uncertainty_vs_raw_course_45deg': uncert,
        'outputs': {'csv': str(csv_path), 'kml': str(kml_path)},
    }
    (args.output_dir/'projection_assumptions_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'wrote: {csv_path}')
    print(f'wrote: {kml_path}')

if __name__ == '__main__':
    main()
