# Visual Navigation for Drones

GNSS-denied visual navigation for drone video.

This project preprocesses GNSS-tagged reference drone flights into a visual/geographic reference map. Then, for a new query flight, it estimates the geographic coordinate of the point seen at the center of the drone camera without using query GNSS during inference.

The repository keeps two stages of the work:

1. **Offline/batch localization** — best accuracy, uses the whole query sequence.
2. **Realtime-style localization** — final realtime pipeline, processes frames causally and can output `NO_ESTIMATE` when uncertain.

---

## 1. Quick Start

This section is the fastest way to reproduce the final realtime pipeline from a clean clone on Windows.

### Step 1 — Open the repository

```bat
cd /d "<Your-Repo-Path-Here>"
```

### Step 2 — Create and activate the Python environment

```bat
py -3.12 -m venv .venv-anyloc
call .venv-anyloc\Scripts\activate
```

After activation, the terminal should show:

```text
(.venv-anyloc)
```

### Step 3 — Install Python dependencies

```bat
python -m pip install --upgrade pip
pip install -r requirements-anyloc.txt
```

This installs the main computer-vision dependencies, including PyTorch, OpenCV, Kornia, scikit-learn, and LightGlue.

### Step 4 — Check CUDA / GPU availability

```bat
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

If CUDA is available, the run will be much faster. If it prints `NO CUDA`, the pipeline can still run, but it will be significantly slower. See the Troubleshooting section below for CUDA notes.

### Step 5 — Check that the final V7 code compiles

```bat
python -m py_compile src\realtime_region_anchor_v7_spread_consistency_localizer.py
python -m py_compile tools\run_v7_all_tests_good.py
python -m py_compile tools\enhance_realtime_summary.py
python -m py_compile tools\export_final_realtime_kml.py
```

### Step 6 — Put the raw input files in `data/raw/`

The required files are listed in the "Required Input Files" section below. The important ones are:

```text
DJI_0006.mp4 / DJI_0006.SRT
DJI_0007.mp4 / DJI_0007.SRT
DJI_0008.mp4 / DJI_0008.SRT
DJI_0009.mp4 / DJI_0009.SRT
DJI_Test1_100m.MP4 / DJI_Test1_100m.SRT
DJI_0010_0011_merged.mp4
```

### Step 7 — Run the full final evaluation

```bat
scripts\run_v7_all_tests_autoprep.bat
```

This is the main final command. It automatically prepares missing processed files, builds reference-region maps, runs every requested test, creates enhanced summaries, and exports KML files.

### Optional — Run only the Test1 realtime pipeline

```bat
scripts\run_final_realtime_test1.bat
```

This runs only the V7 Test1 pipeline instead of all leave-one-out tests. Use this for a faster sanity check after the project has already been set up.

### Output location

After the run, results are written under:

```text
outputs/realtime/<query_video_name>/
```

For example:

```text
outputs/realtime/DJI_Test1_100m/
```

Open:

```text
summary.md
paths.kml
```

to inspect the final result.

---

## 2. External Requirements

### Python packages

Install with:

```bat
pip install -r requirements-anyloc.txt
```

The main required packages are:

```text
torch
torchvision
pillow
numpy
tqdm
natsort
einops
scikit-learn
tyro
opencv-python
kornia
LightGlue
```

### FFmpeg

`ffmpeg` is required for frame extraction if frame folders are missing.

Check:

```bat
ffmpeg -version
ffprobe -version
```

If missing:

```bat
winget install Gyan.FFmpeg
```

### DINOv2

The project expects a local DINOv2 checkout:

```bat
git clone https://github.com/facebookresearch/dinov2.git third_party\dinov2
```

The DINOv2 ViT-S/14 checkpoint must be placed at:

```text
outputs/models/dinov2/dinov2_vits14_pretrain.pth
```

Create the folder if needed:

```bat
mkdir outputs\models\dinov2
```

---

## 3. Required Input Files

Place the raw input files in:

```text
data/raw/
```

Required files:

```text
DJI_0006.mp4
DJI_0006.SRT

DJI_0007.mp4
DJI_0007.SRT

DJI_0008.mp4
DJI_0008.SRT

DJI_0009.mp4
DJI_0009.SRT

DJI_Test1_100m.MP4
DJI_Test1_100m.SRT

DJI_0010_0011_merged.mp4
```

`DJI_0010_0011_merged.mp4` is treated as a no-SRT/no-ground-truth query. It receives estimated paths only.

---

## 4. What the Final Runner Builds Automatically

Run:

```bat
scripts\run_v7_all_tests_autoprep.bat
```

The script calls:

```text
tools/run_v7_all_tests_good.py
```

It automatically creates missing files in `data/processed/`:

```text
DJI_0006_telemetry.csv
DJI_0006_ground_projection_60deg.csv
DJI_0006_frame_manifest_1fps.csv

DJI_0007_telemetry.csv
DJI_0007_ground_projection_60deg.csv
DJI_0007_frame_manifest_1fps.csv

DJI_0008_telemetry.csv
DJI_0008_ground_projection_60deg.csv
DJI_0008_frame_manifest_1fps.csv

DJI_0009_telemetry.csv
DJI_0009_ground_projection_60deg.csv
DJI_0009_frame_manifest_1fps.csv

DJI_Test1_100m_telemetry.csv
DJI_Test1_100m_ground_projection_45deg.csv
DJI_Test1_100m_frame_manifest_1fps.csv
```

It also builds the reference-region files using `--grid-m 90` and `--segment-frame-span 3000`:

```text
realtime_reference_regions_v2.csv
reference_regions_excluding_DJI_0006.csv
reference_regions_excluding_DJI_0007.csv
reference_regions_excluding_DJI_0008.csv
reference_regions_excluding_DJI_0009.csv
```

`realtime_reference_regions_v2.csv` is the final Test1 reference-region file. The older `reference_regions_all_0006_0009.csv` file is not used for the final Test1 result.

If frame folders are missing, the script uses `ffmpeg` to extract them at 1 FPS.

---

## 5. Tests Performed

The final runner tests:

| Query video            | Reference set                               | Query SRT truth? |
| ---                    | ---                                         | ---              |
| `DJI_0006`             | `DJI_0007 + DJI_0008 + DJI_0009`            | Yes              |
| `DJI_0007`             | `DJI_0006 + DJI_0008 + DJI_0009`            | Yes              |
| `DJI_0008`             | `DJI_0006 + DJI_0007 + DJI_0009`            | Yes              |
| `DJI_0009`             | `DJI_0006 + DJI_0007 + DJI_0008`            | Yes              |
| `DJI_Test1_100m`       | `DJI_0006 + DJI_0007 + DJI_0008 + DJI_0009` | Yes              |
| `DJI_0010_0011_merged` | `DJI_0006 + DJI_0007 + DJI_0008 + DJI_0009` | No               |

The leave-one-out tests make sure a reference video is not tested against itself.

---

## 6. Output Folders

Every test writes to:

```text
outputs/realtime/<query_video_name>/
```

Example:

```text
outputs/realtime/DJI_0006/
outputs/realtime/DJI_0007/
outputs/realtime/DJI_0008/
outputs/realtime/DJI_0009/
outputs/realtime/DJI_Test1_100m/
outputs/realtime/DJI_0010_0011_merged/
```

Each folder contains:

```text
realtime_predictions.csv
raw_realtime_summary.json
summary.json
summary.md
paths.kml
region_anchor_debug.jsonl
query_frames/
```

Use `summary.md` for a readable result summary.

Use `paths.kml` for Google Earth visualization.

---

## 7. KML Color Convention

For tests with known SRT truth:

| Path                                              | Color  |
| ---                                               | ---    |
| Calculated SRT drone path                         | Green  |
| Calculated SRT drone look-at / camera-center path | Blue   |
| Estimated drone path                              | Red    |
| Estimated look-at / camera-center path            | Yellow |

For `DJI_0010_0011_merged`, because no query SRT truth is available:

| Path                                   | Color  |
| ---                                    | ---    |
| Estimated drone path                   | Red    |
| Estimated look-at / camera-center path | Yellow |

---

## 8. Summary Metrics

Each `summary.md` and `summary.json` contains:

| Metric        | Meaning                                                                            |
| ---           | ---                                                                                |
| Mean error    | Average error distance in metres.                                                  |
| Median error  | Middle error value.                                                                |
| P90 error     | 90% of evaluated estimates are at or below this error.                             |
| P95 error     | 95% of evaluated estimates are at or below this error.                             |
| Max error     | Worst evaluated valid estimate.                                                    |
| % under 100 m | Percent of evaluated valid estimates with error ≤ 100 m.                           |
| % under 50 m  | Percent of evaluated valid estimates with error ≤ 50 m.                            |
| % under 10 m  | Percent of evaluated valid estimates with error ≤ 10 m.                            |
| % under 5 m   | Percent of evaluated valid estimates with error ≤ 5 m.                             |
| Coverage      | Valid estimates divided by total processed frames.                                 |
| NO_ESTIMATE   | Frames where the system refused to output a coordinate because confidence was low. |

For videos without SRT truth, such as `DJI_0010_0011_merged`, accuracy metrics are marked as unavailable and only estimated paths are exported.

---

## 9. Final Realtime Result

The final retained realtime pipeline is `Region Anchor V7 Spread Consistency` using the old/good reference setup:

```text
data/processed/realtime_reference_regions_v2.csv
```

On `DJI_Test1_100m`, the final realtime result is:

| Metric               | Value    |
| ---                  | ---:     |
| Frames processed     | 739      |
| Valid estimates      | 441      |
| NO_ESTIMATE frames   | 298      |
| Mean look-at error   | 97.78 m  |
| Median look-at error | 74.97 m  |
| P90 look-at error    | 229.18 m |
| P95 look-at error    | 238.34 m |
| % under 100 m        | 70.29%   |
| Coverage             | 59.68%   |

This is the final realtime result. The stronger offline result below is kept only for comparison.

---

## 10. Best Offline Result Kept for Comparison

The strongest Test1 result is still offline/postprocessed, not realtime.

Best offline Test1 result:

| Pipeline                                                 | Mean    | Median  | P90      | Max      |
| ---                                                      | ---:    | ---:    | ---:     | ---:     |
| Top25 + Motion Viterbi + segment calibration + smoothing | 53.54 m | 28.53 m | 105.46 m | 362.88 m |

This result is reported for comparison only. It is not the final realtime pipeline.
---

## 11. Scale / Height Handling

The project handles altitude-induced image scale in two ways.

### V7 spread consistency

V7 checks whether the LightGlue/RANSAC inlier spread is consistent with the query/reference altitude difference. This does not physically change the image; it is a geometric validation rule.

### Optional scale-aware reference manifests

`tools/make_scale_aware_reference_manifest.py` can physically crop and resize reference frames to simulate a target altitude:

```text
crop_ratio = target_altitude / reference_altitude
```

This is optional and not enabled by default in the final all-tests runner. It should be used only as a separate experiment unless it beats the normal V7 run.

---

## 12. Main Files

```text
src/
  anyloc_dino_retrieval.py
  build_frame_manifest.py
  project_ground_point.py
  realtime_beam_localizer.py
  realtime_region_anchor_v7_spread_consistency_localizer.py
  telemetry_parser.py

tools/
  build_reference_regions_v2.py
  enhance_realtime_summary.py
  export_final_realtime_kml.py
  make_scale_aware_reference_manifest.py
  run_v7_all_tests_good.py

scripts/
  run_v7_all_tests_autoprep.bat
  run_realtime_test1_region_anchor_v7_spread_consistency_2fps.bat
  run_realtime_test1_v7_spread_consistency_2fps_debug_around240.bat
```

---


## 13. Troubleshooting

### `ModuleNotFoundError: sklearn` or `ModuleNotFoundError: lightglue`

Activate the environment and install requirements:

```bat
call .venv-anyloc\Scripts\activate
pip install -r requirements-anyloc.txt
```

### CUDA is not being used

Check:

```bat
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

If CUDA is false, install a CUDA-enabled PyTorch build appropriate for your machine.

### First run is slow

The first run builds manifests, reference regions, descriptor caches, and LightGlue weights. Later runs reuse caches.

With cuda enabled long videos such as `DJI_0006` and `DJI_0008` can take around 40–45 minutes at 2 FPS because we run LightGlue verification on many sampled frames.

### Run only Test1

For a faster sanity check, run only the final Test1 pipeline:

```bat
scripts\run_final_realtime_test1.bat
```

