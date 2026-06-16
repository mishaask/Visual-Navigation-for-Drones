from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs" / "realtime"
ANYLOC_OUT = ROOT / "outputs" / "anyloc"

REF_FLIGHTS = ["DJI_0006", "DJI_0007", "DJI_0008", "DJI_0009"]


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    printable = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    print("\n>>>", printable, flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def python() -> str:
    return sys.executable


def find_existing(candidates: list[Path], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    msg = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{msg}")


def video_path(stem: str) -> Path:
    return find_existing(
        [
            RAW / f"{stem}.mp4",
            RAW / f"{stem}.MP4",
            RAW / f"{stem}.mov",
            RAW / f"{stem}.MOV",
        ],
        f"video for {stem}",
    )


def srt_path(stem: str) -> Path:
    return find_existing(
        [
            RAW / f"{stem}.SRT",
            RAW / f"{stem}.srt",
        ],
        f"SRT for {stem}",
    )


def frame_dir_for(stem: str) -> Path:
    if stem == "DJI_Test1_100m":
        return PROCESSED / "frames_test1_1fps"
    suffix = stem.split("_")[-1]
    return PROCESSED / f"frames_{suffix}_1fps"


def ensure_frames(stem: str) -> Path:
    v = video_path(stem)
    out_dir = frame_dir_for(stem)

    existing = list(out_dir.glob("*.jpg")) if out_dir.exists() else []
    if existing:
        print(f"frames already exist for {stem}: {out_dir} ({len(existing)} jpg files)")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")
    run(["ffmpeg", "-y", "-i", str(v), "-vf", "fps=1", pattern])
    return out_dir


def ensure_manifest(stem: str, camera_angle_deg: float) -> Path:
    frames = ensure_frames(stem)
    manifest = PROCESSED / f"{stem}_frame_manifest_1fps.csv"

    telemetry = PROCESSED / f"{stem}_telemetry.csv"
    projection = PROCESSED / f"{stem}_ground_projection_{int(camera_angle_deg)}deg.csv"

    if not telemetry.exists():
        run([python(), "src/telemetry_parser.py", str(srt_path(stem)), str(telemetry)])

    if not projection.exists():
        run(
            [
                python(),
                "src/project_ground_point.py",
                str(telemetry),
                str(projection),
                "--camera-angle-deg",
                str(camera_angle_deg),
                "--camera-angle-source",
                "fixed",
                "--heading-source",
                "trajectory",
            ]
        )

    if not manifest.exists():
        run(
            [
                python(),
                "src/build_frame_manifest.py",
                str(frames),
                str(projection),
                str(manifest),
                "--fps",
                "1",
            ]
        )
    else:
        print(f"manifest already exists: {manifest}")

    return manifest


def ensure_all_manifests() -> dict[str, Path]:
    manifests: dict[str, Path] = {}

    for stem in REF_FLIGHTS:
        manifests[stem] = ensure_manifest(stem, 60)

    manifests["DJI_Test1_100m"] = ensure_manifest("DJI_Test1_100m", 45)
    return manifests


def build_regions_good(name: str, refs: list[str], manifests: dict[str, Path]) -> Path:
    """Build reference regions using the old/good setup: grid 90 + segment-frame-span 3000."""
    if name == "all_0006_0009":
        out_csv = PROCESSED / "realtime_reference_regions_v2.csv"
        out_json = PROCESSED / "realtime_reference_regions_v2_summary.json"
    else:
        out_csv = PROCESSED / f"reference_regions_{name}.csv"
        out_json = PROCESSED / f"reference_regions_{name}_summary.json"

    cmd = [
        python(),
        "tools/build_reference_regions_v2.py",
        "--output-csv",
        str(out_csv),
        "--summary-json",
        str(out_json),
        "--grid-m",
        "90",
        "--segment-frame-span",
        "3000",
    ]

    for ref in refs:
        dataset_id = ref.replace("DJI_", "")
        cmd.extend(["--reference-manifest", f"{dataset_id}={manifests[ref]}"])

    run(cmd)
    return out_csv


def old_good_v7_args() -> list[str]:
    """The old/good V7 parameter set from run_realtime_test1_region_anchor_v7_spread_consistency_2fps.bat."""
    return [
        "--sample-fps", "2",
        "--region-grid-m", "90",
        "--global-topk", "100",
        "--candidate-pool-limit", "90",
        "--locked-region-ring", "1",
        "--recovery-region-ring", "3",
        "--recovery-after", "5",
        "--fail-limit", "10",
        "--acquire-window", "5",
        "--acquire-min-region-votes", "3",
        "--acquire-min-geometry-votes", "2",
        "--transition-confirm-window", "3",
        "--transition-min-votes", "2",
        "--acquire-vote-topn", "45",
        "--acquire-verified-vote-bonus", "1.0",
        "--acquire-geometry-vote-bonus", "3.0",
        "--lg-topk", "12",
        "--lightglue-every", "2",
        "--lg-min-inliers", "10",
        "--lg-min-ratio", "0.08",
        "--require-geometry-verified",
        "--use-geometry-scoring",
        "--geometry-verified-bonus", "2.2",
        "--geometry-failed-penalty", "3.0",
        "--geometry-min-inliers", "8",
        "--geometry-min-inlier-ratio", "0.08",
        "--max-homography-reproj-rmse", "10.0",
        "--min-query-inlier-width-frac", "0.14",
        "--min-query-inlier-height-frac", "0.08",
        "--min-reference-inlier-width-frac", "0.14",
        "--min-reference-inlier-height-frac", "0.08",
        "--min-altitude-spread-scale", "0.25",
        "--require-spread-balance",
        "--spread-balance-tolerance", "2.5",
        "--same-altitude-ratio-threshold", "1.35",
        "--dino-weight", "4.0",
        "--inlier-weight", "1.15",
        "--ratio-weight", "1.0",
        "--unverified-penalty", "2.8",
        "--motion-weight", "7.0",
        "--max-step-m", "45",
        "--strong-jump-m", "180",
        "--flow-contradiction-enabled",
        "--low-flow-px", "12",
        "--camera-hfov-deg", "82",
        "--default-altitude-m", "100",
        "--oblique-scale", "1.35",
        "--low-altitude-m", "45",
        "--low-altitude-ring-bonus", "1",
        "--fast-flow-px", "22",
        "--very-fast-flow-px", "45",
        "--fast-flow-ring-bonus", "1",
        "--very-fast-flow-ring-bonus", "2",
        "--fast-speed-mps", "10",
        "--very-fast-speed-mps", "20",
        "--fast-speed-ring-bonus", "1",
        "--very-fast-speed-ring-bonus", "2",
        "--bad-frame-ring-bonus", "2",
        "--max-dynamic-region-ring", "6",
        "--flow-calib-min-px", "4",
        "--flow-calib-max-jump-m", "180",
        "--max-flow-pred-speed-mps", "35",
        "--no-estimate-during-acquire",
        "--flow-propagate-holds",
        "--flow-propagation-min-quality", "0.55",
        "--flow-propagation-min-points", "40",
        "--flow-propagation-max-step-m", "15",
        "--flow-propagation-max-bad-count", "3",
        "--disable-flow-propagation-in-recovery",
        "--max-flow-hold-speed-mps", "35",
        "--max-flow-speed-mps", "35",
        "--hold-max-bad-count", "3",
        "--projected-center-offset-warning-frac", "0.28",
    ]


def clear_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        print(f"overwriting output folder: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def run_test_good(
    test_name: str,
    query_video: Path,
    truth_manifest: Path | None,
    regions_csv: Path,
    descriptor_cache: Path,
) -> None:
    out_dir = OUTPUTS / test_name
    clear_output_dir(out_dir)
    query_frames = out_dir / "query_frames"

    cmd = [
        python(),
        "src/realtime_region_anchor_v7_spread_consistency_localizer.py",
        "--reference-regions",
        str(regions_csv),
        "--query-video",
        str(query_video),
        "--reference-descriptor-cache",
        str(descriptor_cache),
        "--query-frame-dir",
        str(query_frames),
        "--output-csv",
        str(out_dir / "realtime_predictions.csv"),
        "--summary-json",
        str(out_dir / "raw_realtime_summary.json"),
        "--debug-jsonl",
        str(out_dir / "region_anchor_debug.jsonl"),
    ]
    cmd.extend(old_good_v7_args())

    if truth_manifest is not None:
        cmd.extend(["--truth-manifest", str(truth_manifest)])

    run(cmd)

    if (ROOT / "tools/enhance_realtime_summary.py").exists():
        run(
            [
                python(),
                "tools/enhance_realtime_summary.py",
                "--predictions",
                str(out_dir / "realtime_predictions.csv"),
                "--base-summary",
                str(out_dir / "raw_realtime_summary.json"),
                "--output-json",
                str(out_dir / "summary.json"),
                "--output-md",
                str(out_dir / "summary.md"),
                "--test-name",
                test_name,
            ]
        )

    if (ROOT / "tools/export_final_realtime_kml.py").exists():
        run(
            [
                python(),
                "tools/export_final_realtime_kml.py",
                "--predictions",
                str(out_dir / "realtime_predictions.csv"),
                "--output",
                str(out_dir / "paths.kml"),
                "--name",
                test_name,
            ]
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run all requested videos using the old/good V7 setup: segment-frame-span 3000 + original V7 params."
    )
    ap.add_argument(
        "--only",
        default="",
        help="Optional test name to run only one test, e.g. DJI_Test1_100m or DJI_0006.",
    )
    args = ap.parse_args()

    required = [
        Path("src/realtime_region_anchor_v7_spread_consistency_localizer.py"),
        Path("src/realtime_beam_localizer.py"),
        Path("src/anyloc_dino_retrieval.py"),
        Path("tools/build_reference_regions_v2.py"),
    ]
    missing = [str(p) for p in required if not (ROOT / p).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    print("=" * 72)
    print("Preparing manifests")
    print("=" * 72)
    manifests = ensure_all_manifests()

    print("=" * 72)
    print("Building OLD/GOOD reference-region files")
    print("=" * 72)

    all_regions = build_regions_good("all_0006_0009", REF_FLIGHTS, manifests)
    loo_regions: dict[str, Path] = {}

    for query_ref in REF_FLIGHTS:
        refs = [r for r in REF_FLIGHTS if r != query_ref]
        loo_regions[query_ref] = build_regions_good(f"excluding_{query_ref}", refs, manifests)

    tests: list[tuple[str, Path, Path | None, Path, Path]] = []

    for query_ref in REF_FLIGHTS:
        tests.append(
            (
                query_ref,
                video_path(query_ref),
                manifests[query_ref],
                loo_regions[query_ref],
                ANYLOC_OUT / f"reference_regions_excluding_{query_ref}_oldgood_v7_dino_descriptors.npy",
            )
        )

    tests.append(
        (
            "DJI_Test1_100m",
            video_path("DJI_Test1_100m"),
            manifests["DJI_Test1_100m"],
            all_regions,
            ANYLOC_OUT / "realtime_reference_regions_v2_dino_descriptors.npy",
        )
    )

    tests.append(
        (
            "DJI_0010_0011_merged",
            video_path("DJI_0010_0011_merged"),
            None,
            all_regions,
            ANYLOC_OUT / "realtime_reference_regions_v2_dino_descriptors.npy",
        )
    )

    if args.only:
        tests = [t for t in tests if t[0].lower() == args.only.lower()]
        if not tests:
            valid = ", ".join([
                *REF_FLIGHTS,
                "DJI_Test1_100m",
                "DJI_0010_0011_merged",
            ])
            raise SystemExit(f"Unknown --only value {args.only!r}. Valid: {valid}")

    print("=" * 72)
    print("Running tests with OLD/GOOD V7 params")
    print("=" * 72)

    for test_name, query_video, truth_manifest, regions_csv, descriptor_cache in tests:
        print("\n" + "#" * 72)
        print(f"TEST: {test_name}")
        print(f"REGIONS: {regions_csv}")
        print(f"OUTPUT: {OUTPUTS / test_name}")
        print("#" * 72)
        run_test_good(test_name, query_video, truth_manifest, regions_csv, descriptor_cache)

    print("=" * 72)
    print("Done. Outputs are in:")
    print(OUTPUTS)
    print("=" * 72)


if __name__ == "__main__":
    main()
