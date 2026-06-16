from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

EARTH_RADIUS_M = 6_378_137.0


def parse_manifest_arg(value: str) -> tuple[str, Path]:
    if '=' not in value:
        p = Path(value)
        return p.stem, p
    ds, path = value.split('=', 1)
    return ds, Path(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def safe_float(v: Any, default: float = math.nan) -> float:
    try:
        if v in (None, '', 'nan'):
            return default
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def local_xy_from_latlon(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    return (
        math.radians(lon - origin_lon) * math.cos(math.radians(origin_lat)) * EARTH_RADIUS_M,
        math.radians(lat - origin_lat) * EARTH_RADIUS_M,
    )


def region_id_for(lat: float, lon: float, origin: tuple[float, float], grid_m: float) -> tuple[str, int, int, float, float]:
    x, y = local_xy_from_latlon(lat, lon, origin[0], origin[1])
    gx = int(math.floor(x / grid_m))
    gy = int(math.floor(y / grid_m))
    return f"r_{gx}_{gy}", gx, gy, x, y


def segment_key(row: dict[str, str], span: int) -> str:
    try:
        frame = int(float(row.get('frame_count', row.get('frame', 0))))
    except Exception:
        frame = 0
    return f"{row.get('dataset_id','')}:{frame // max(span,1)}"


def main() -> None:
    ap = argparse.ArgumentParser(description='Build spatial region IDs for realtime anchor localization.')
    ap.add_argument('--reference-manifest', action='append', required=True)
    ap.add_argument('--output-csv', type=Path, required=True)
    ap.add_argument('--summary-json', type=Path, required=True)
    ap.add_argument('--grid-m', type=float, default=90.0)
    ap.add_argument('--segment-frame-span', type=int, default=3000)
    args = ap.parse_args()

    rows: list[dict[str, str]] = []
    for value in args.reference_manifest:
        ds, path = parse_manifest_arg(value)
        part = read_csv(path)
        for r in part:
            r = dict(r)
            r['dataset_id'] = r.get('dataset_id') or ds
            rows.append(r)
        print(f'reference {ds}: {len(part)} rows from {path}')
    if not rows:
        raise RuntimeError('no reference rows')
    pts = [(safe_float(r.get('ground_latitude')), safe_float(r.get('ground_longitude'))) for r in rows]
    pts = [(a,b) for a,b in pts if math.isfinite(a) and math.isfinite(b)]
    if not pts:
        raise RuntimeError('reference manifests do not contain ground_latitude/ground_longitude')
    origin = (sum(a for a,_ in pts)/len(pts), sum(b for _,b in pts)/len(pts))

    out=[]
    region_counts: dict[str,int] = {}
    region_datasets: dict[str,set[str]] = {}
    for idx, r in enumerate(rows):
        lat = safe_float(r.get('ground_latitude'))
        lon = safe_float(r.get('ground_longitude'))
        if not math.isfinite(lat) or not math.isfinite(lon):
            continue
        rid, gx, gy, x, y = region_id_for(lat, lon, origin, args.grid_m)
        r = dict(r)
        r['reference_index'] = str(idx)
        r['region_id'] = rid
        r['region_x'] = str(gx)
        r['region_y'] = str(gy)
        r['region_local_x_m'] = f'{x:.3f}'
        r['region_local_y_m'] = f'{y:.3f}'
        r['region_grid_m'] = f'{args.grid_m:.3f}'
        r['segment_key'] = segment_key(r, args.segment_frame_span)
        out.append(r)
        region_counts[rid] = region_counts.get(rid,0) + 1
        region_datasets.setdefault(rid,set()).add(r.get('dataset_id',''))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if out:
        fields=[]
        for r in out:
            for k in r.keys():
                if k not in fields:
                    fields.append(k)
        with args.output_csv.open('w', newline='', encoding='utf-8') as f:
            w=csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(out)

    summary={
        'rows': len(out),
        'regions': len(region_counts),
        'grid_m': args.grid_m,
        'origin_latitude': origin[0],
        'origin_longitude': origin[1],
        'top_regions': sorted(region_counts.items(), key=lambda x:x[1], reverse=True)[:20],
        'multi_dataset_regions': sum(1 for v in region_datasets.values() if len(v)>1),
        'output_csv': str(args.output_csv),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'wrote: {args.output_csv}')

if __name__ == '__main__':
    main()
