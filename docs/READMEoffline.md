# Visual Navigation for Drones

This repository contains an easy-to-build implementation for the assignment's main GNSS-denied optical navigation problem:

> Given a preprocessed reference flight with video and telemetry, estimate in a new flight, without GNSS at inference time, the GPS coordinate of the center point seen by the drone camera.

The retained solution is an AnyLoc-inspired visual place recognition pipeline. It uses frozen DINOv2 image descriptors to retrieve candidate reference frames, SuperPoint + LightGlue to verify the best candidates locally, and a temporal Viterbi motion prior to choose a coherent path.

## Current Best Result

Main benchmark:

- Reference database: DJI Mini 3 Pro `v11`, `v12`, `v13`
- Query/test flight: DJI Mini 3 Pro `v14`
- Sampling: `1 fps`
- Target: projected ground coordinate of the video center

Best retained result:

| Method | Mean error | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: |
| DINOv2 global retrieval | 27.28 m | 20.04 m | 57.63 m | 180.52 m |
| DINOv2 + LightGlue + Motion Viterbi | 18.83 m | 15.21 m | 36.05 m | 72.53 m |
| **+ Path smoothing (w=19)** | **14.16 m** | **13.05 m** | **25.63 m** | **38.94 m** |

Error tolerance breakdown — best result (Viterbi + smoothing w=19):

| Threshold | Frames | % of total | Frequency |
| --- | ---: | ---: | ---: |
| ≤ 5 m | 14 / 115 | 12.2% | ~1 every 8 s |
| ≤ 10 m | 41 / 115 | 35.7% | ~1 every 3 s |
| ≤ 15 m | 68 / 115 | 59.1% | ~1 every 2 s |

Main outputs:

- `outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps_motion_viterbi_top6_acc0_results.csv`
- `outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps_motion_viterbi_top6_acc0_summary.json`
- `outputs/maps/dji_mini3_v14_google_earth_best_motion_viterbi.kml`
- `outputs/debug/dji_mini3_v14_worst_retrieval_debug.html`

## Repository Layout

```text
src/
  telemetry_parser.py              DJI SRT to structured telemetry CSV
  project_ground_point.py          geometric camera-center projection
  build_frame_manifest.py          joins extracted frames with projected coordinates
  anyloc_dino_retrieval.py         DINOv2 feature extraction and aggregation
  frozen_dino_cross_retrieval.py   cross-flight visual retrieval
  temporal_lightglue_rerank.py     LightGlue candidate verification
  motion_viterbi_rerank.py         retained temporal path selection
  confidence_gate_results.py       FIX/NO_FIX confidence evaluation
  smooth_path.py                   Gaussian path smoothing on Viterbi output
  export_google_earth_kml.py       Google Earth visualization
  build_retrieval_debug_page.py    worst-error debug HTML
  analyze_retrieval_failures.py    worst-error CSV extraction
  make_failure_contact_sheet.py    visual contact sheet for failures
  dji_mp4_metadata.py              optional DJI MP4 metadata extraction
  trajectory_report.py             optional GNSS path SVG report
  projection_report.py             optional camera-center projection SVG report
  interpolated_navigation.py       negative result: FIX/interp approach (29.32 m, rejected)

data/raw/
  DJI_v11.SRT, DJI_v12.SRT, DJI_v13.SRT, DJI_v14.SRT

data/processed/
  DJI_v*_telemetry.csv
  DJI_v*_ground_projection_60deg.csv
  DJI_v*_frame_manifest_1fps.csv
  frames_v*_1fps/

docs/
  final_report.md
  literature_review.md

scripts/
  run_best_pipeline.sh
```

## Setup

Create and activate the environment:

```bash
python3 -m venv .venv-anyloc
source .venv-anyloc/bin/activate
pip install --upgrade pip
pip install -r requirements-anyloc.txt
```

The project expects a local DINOv2 checkout and pretrained weights:

```bash
git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
mkdir -p outputs/models/dinov2
```

Place the DINOv2 ViT-S/14 checkpoint at:

```text
outputs/models/dinov2/dinov2_vits14_pretrain.pth
```

LightGlue is installed from `requirements-anyloc.txt`.

## Rebuild The Data

If frames are missing, extract them at 1 fps:

```bash
ffmpeg -i data/raw/DJI_v11.mp4 -vf fps=1 data/processed/frames_v11_1fps/frame_%06d.jpg
ffmpeg -i data/raw/DJI_v12.mp4 -vf fps=1 data/processed/frames_v12_1fps/frame_%06d.jpg
ffmpeg -i data/raw/DJI_v13.mp4 -vf fps=1 data/processed/frames_v13_1fps/frame_%06d.jpg
ffmpeg -i data/raw/DJI_v14.mp4 -vf fps=1 data/processed/frames_v14_1fps/frame_%06d.jpg
```

Parse SRT telemetry:

```bash
python src/telemetry_parser.py data/raw/DJI_v11.SRT data/processed/DJI_v11_telemetry.csv
python src/telemetry_parser.py data/raw/DJI_v12.SRT data/processed/DJI_v12_telemetry.csv
python src/telemetry_parser.py data/raw/DJI_v13.SRT data/processed/DJI_v13_telemetry.csv
python src/telemetry_parser.py data/raw/DJI_v14.SRT data/processed/DJI_v14_telemetry.csv
```

Project the center of the video onto the ground. The Mini 3 Pro flights were documented as 60 degrees at about 119 m, so this benchmark uses a fixed 60 degree camera angle and trajectory-derived heading:

```bash
python src/project_ground_point.py data/processed/DJI_v11_telemetry.csv data/processed/DJI_v11_ground_projection_60deg.csv --camera-angle-deg 60 --camera-angle-source fixed --heading-source trajectory
python src/project_ground_point.py data/processed/DJI_v12_telemetry.csv data/processed/DJI_v12_ground_projection_60deg.csv --camera-angle-deg 60 --camera-angle-source fixed --heading-source trajectory
python src/project_ground_point.py data/processed/DJI_v13_telemetry.csv data/processed/DJI_v13_ground_projection_60deg.csv --camera-angle-deg 60 --camera-angle-source fixed --heading-source trajectory
python src/project_ground_point.py data/processed/DJI_v14_telemetry.csv data/processed/DJI_v14_ground_projection_60deg.csv --camera-angle-deg 60 --camera-angle-source fixed --heading-source trajectory
```

Build frame manifests:

```bash
python src/build_frame_manifest.py data/processed/frames_v11_1fps data/processed/DJI_v11_ground_projection_60deg.csv data/processed/DJI_v11_frame_manifest_1fps.csv --fps 1
python src/build_frame_manifest.py data/processed/frames_v12_1fps data/processed/DJI_v12_ground_projection_60deg.csv data/processed/DJI_v12_frame_manifest_1fps.csv --fps 1
python src/build_frame_manifest.py data/processed/frames_v13_1fps data/processed/DJI_v13_ground_projection_60deg.csv data/processed/DJI_v13_frame_manifest_1fps.csv --fps 1
python src/build_frame_manifest.py data/processed/frames_v14_1fps data/processed/DJI_v14_ground_projection_60deg.csv data/processed/DJI_v14_frame_manifest_1fps.csv --fps 1
```

## Run The Best Pipeline

```bash
./scripts/run_best_pipeline.sh
```

The script recomputes:

1. DINOv2 descriptors and top-k retrieval.
2. LightGlue verification of DINOv2 candidates.
3. Motion-Viterbi selection of a coherent estimated path.
4. Google Earth KML export.
5. Confidence-gated FIX/NO_FIX evaluation.
6. Gaussian path smoothing (w=19, σ=5.4) — reduces mean error from 18.83 m to 14.16 m.
7. Preliminary experiment SVG — three-path comparison (direction 4).

## Confidence-Gated Fixes

The base pipeline outputs one position every second. To answer the question "how often can we be confident that the position is correct?", the confidence-gated evaluation can abstain:

```text
FIX    if visual evidence is strong enough
NO_FIX otherwise
```

Run:

```bash
python src/confidence_gate_results.py \
  outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps_motion_viterbi_top6_acc0_results.csv \
  --query-manifest data/processed/DJI_v14_frame_manifest_1fps.csv \
  --sweep-csv outputs/anyloc/dji_mini3_confidence_gate_sweep.csv \
  --decisions-csv outputs/anyloc/dji_mini3_confidence_gate_best_decisions.csv \
  --summary-json outputs/anyloc/dji_mini3_confidence_gate_best_summary.json \
  --good-error-m 20 \
  --min-coverage 0.30 \
  --max-longest-gap-s 60
```

Current retained policy:

- `motion_viterbi_rank <= 6`
- `lg_inlier_count >= 50`
- `lg_inlier_ratio >= 0.70`
- `DINO similarity >= 0.98`

Result with a 20 m "good fix" threshold:

| Mode | Coverage | Mean accepted error | Good fixes <=20m | Mean time between fixes | Longest gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always output | 100.0% | 18.83 m | 65.2% | 1.00 s | 0.00 s |
| Confidence gated | 30.4% | 13.67 m | 80.0% | 2.00 s | 46.01 s |

## Preliminary Experiment (Direction 4)

The preliminary experiment answers direction 4: given a video and the camera angle, compute the expected ground-center path and compare it with the pipeline estimate and the SRT-captured path.

Run it standalone (after the pipeline has produced results):

```bash
python src/preliminary_experiment_report.py \
  data/processed/DJI_v14_ground_projection_60deg.csv \
  data/processed/DJI_v14_frame_manifest_1fps.csv \
  outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps_motion_viterbi_top6_acc0_results.csv \
  data/processed/DJI_v11_frame_manifest_1fps.csv \
  data/processed/DJI_v12_frame_manifest_1fps.csv \
  data/processed/DJI_v13_frame_manifest_1fps.csv \
  --smoothed-csv outputs/anyloc/dji_mini3_smoothed_results.csv \
  --output outputs/figures/preliminary_experiment_v14.svg
```

Output: `outputs/figures/preliminary_experiment_v14.svg`

The figure overlays three paths in local XY coordinates (metres):

- **Blue dashed** — drone GNSS trajectory from SRT telemetry
- **Green** — ground-truth camera-center, computed geometrically from altitude + 60° camera angle + trajectory heading
- **Red** — estimated camera-center from the DINOv2 + LightGlue + Motion Viterbi + Gaussian smoothing (w=19) pipeline

Grey lines connect each ground-truth point to its estimate; darker = larger error.

## Real-Time Interpretation

The retained pipeline is compatible with a real-time version if preprocessing has already built the reference database. In real time, each incoming frame is embedded with frozen DINOv2, compared with the reference descriptors, reranked with LightGlue on a small top-k set, and passed to the temporal selector. GNSS is only used offline to build/evaluate the reference map; it is not used as an input for the query flight estimate.

## Reports

- `docs/final_report.md` explains the assignment mapping, pipeline, experiments, results, and limitations.
- `docs/literature_review.md` summarizes AnyLoc and the supporting open-source methods used in this project.
