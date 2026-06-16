# Final Report

## Assignment Objective

The assignment asks us to solve an optical navigation problem for drones:

> Given a reference drone flight with video and telemetry, including GNSS, barometric height, and camera angle, preprocess the data so that a new realtime flight can estimate where the drone camera is looking without using GNSS during inference.

The concrete output is the GPS coordinate of the point at the center of the video frame. During evaluation, when the query SRT is available, we compare the estimated coordinate with the SRT-derived camera-center ground coordinate.

The project therefore has two separate requirements:

1. Build a visual/geographic reference map from GNSS-tagged reference flights.
2. Estimate the camera-center coordinate of a new query video without using the query GNSS during inference.

## Data Used

### Final realtime benchmark data

| Role                | Videos                                         | Notes                                                                 |
| ---                 | ---                                            | ---                                                                   |
| Reference flights   | `DJI_0006`, `DJI_0007`, `DJI_0008`, `DJI_0009` | Used as the visual map. Each has video and SRT telemetry.             |
| Leave-one-out tests | `DJI_0006`, `DJI_0007`, `DJI_0008`, `DJI_0009` | Each video is tested while excluded from the reference set.           |
| Test video with SRT | `DJI_Test1_100m`                               | Used for final quantitative realtime evaluation.                      |
| No-SRT query        | `DJI_0010_0011_merged`                         | Used as a no-GNSS/no-ground-truth demonstration. Estimated path only. |

The all-tests runner writes one output folder per tested video:

```text
outputs/realtime/DJI_0006/
outputs/realtime/DJI_0007/
outputs/realtime/DJI_0008/
outputs/realtime/DJI_0009/
outputs/realtime/DJI_Test1_100m/
outputs/realtime/DJI_0010_0011_merged/
```

### Historical offline benchmark data

Earlier experiments used Mini 3 Pro `v11`, `v12`, `v13`, and `v14` flights. These were useful for proving the offline AnyLoc-style pipeline before moving to realtime.

## Final Retained Realtime Pipeline

The final realtime pipeline is:

```text
Region Anchor Spread Consistency
```

The main command is:

```bat
scripts\run_v7_all_tests_autoprep.bat
```

This command:

1. builds missing processed manifests
2. builds reference-region maps
3. runs the realtime localizer on every requested test video
4. writes enhanced summaries
5. exports clean KML files

For the final Test1 result, the runner uses:

```text
data/processed/realtime_reference_regions_v2.csv
```

This file is built with `--grid-m 90` and `--segment-frame-span 3000`. The newer `reference_regions_all_0006_0009.csv` setup was tested but was not retained for the final Test1 result.

## Pipeline Stages

### 1. Parse telemetry

`src/telemetry_parser.py` converts DJI SRT files into structured CSV files containing frame number, time, latitude, longitude, altitude, and available camera metadata.

### 2. Project the video center onto the ground

`src/project_ground_point.py` estimates the geographic coordinate of the center of the camera frame using:

- drone GNSS position from the SRT,
- relative altitude,
- fixed camera angle when direct gimbal angle is unavailable,
- trajectory-derived heading when yaw is unavailable.

For the current data:

- `DJI_0006` to `DJI_0009` use a fixed 60 degree projection.
- `DJI_Test1_100m` uses a fixed 45 degree projection.

### 3. Build frame manifests

`src/build_frame_manifest.py` joins extracted frames with their projected ground coordinates. These manifests are used both as reference-map entries and as evaluation truth when the query SRT is known.

### 4. Build reference regions

`tools/build_reference_regions_v2.py` divides the reference map into spatial regions. Region IDs allow the realtime system to avoid global search after it has acquired a trusted local area.

For leave-one-out tests, the query video is excluded from the region map:

| Query      | Reference region map excludes |
| ---        | ---        |
| `DJI_0006` | `DJI_0006` |
| `DJI_0007` | `DJI_0007` |
| `DJI_0008` | `DJI_0008` |
| `DJI_0009` | `DJI_0009` |

### 5. Retrieve visual candidates with frozen DINOv2

DINOv2 descriptors are computed for reference frames and query frames. The reference descriptors are cached, so they do not need to be recomputed for every run.

The DINOv2 descriptor produces the initial top candidates.

### 6. Verify candidates with LightGlue and homography

For each candidate set, SuperPoint + LightGlue finds local feature matches. RANSAC homography then filters matches into geometric inliers and outliers.

This is important because global image retrieval can confuse repeated structures such as roads, buildings, fields, and parking lots.

### 7. Region-anchor realtime tracking

The realtime state logic uses three ideas:

1. **Acquire**: search globally until a region has enough visual/geometric support.
2. **Lock**: after a good region is accepted, search only nearby regions.
3. **Reacquire**: if confidence drops, output `NO_ESTIMATE` and widen/globalize search again.

This prevents the system from forcing a coordinate when the visual evidence is weak.

### 8. Optical flow as a short-gap cue only

Optical flow is used as a bounded motion prior, not as a final localization method. Earlier versions showed that optical flow can drift badly when the anchor is wrong. V7 only uses it in restricted cases and prefers `NO_ESTIMATE` during uncertain recovery.

### 9. Spread consistency

V7 adds a spread-consistency check based on LightGlue/RANSAC inliers.

The idea is:

- If query and reference altitude are similar, the matched inlier support should cover a comparable image area.
- If the inliers are very spread out in one image but collapsed into a tiny patch in the other, the match may be a repeated-object or wrong-viewpoint match.
- The threshold is altitude-scaled so that different flight heights are handled more fairly.

This reduced accepted-match error on `DJI_Test1_100m`.

## Final Realtime Result

On `DJI_Test1_100m`, the final retained V7 realtime result is:

| Pipeline                            | Evaluated frames | No-estimate frames | Mean    | Median  | P90      | P95      | % under 100 m |
| ---                                 | ---:             | ---:               | ---:    | ---:    | ---:     | ---:     | ---:          |
| Region Anchor V7 Spread Consistency | 441              | 298                | 97.78 m | 74.97 m | 229.18 m | 238.34 m | 70.29%        |

## Output Metrics

Each enhanced summary includes:

| Metric        | Meaning |
| ---           | --- |
| Mean error    | Average distance between estimated look-at point and SRT-derived look-at point. |
| Median error  | Middle error value. |
| P90 error     | 90% of evaluated estimates are at or below this error. |
| P95 error     | 95% of evaluated estimates are at or below this error. |
| Max error     | Worst evaluated valid estimate. |
| % under 100 m | Percentage of valid evaluated estimates at or below 100 m error. |
| % under 50 m  | Percentage of valid evaluated estimates at or below 50 m error. |
| % under 10 m  | Percentage of valid evaluated estimates at or below 10 m error. |
| % under 5 m   | Percentage of valid evaluated estimates at or below 5 m error. |
| Coverage      | Valid estimates divided by processed frames. |
| NO_ESTIMATE   | Frames where the system intentionally abstained. |

The percentages are computed over valid evaluated estimates, not all processed frames.

## Final KML Output

For tests with known query SRT, the KML contains:

| Path                              | Color  |
| ---                               | ---    |
| Calculated SRT drone path         | Green  |
| Calculated SRT drone look-at path | Blue   |
| Estimated drone path              | Red    |
| Estimated look-at path            | Yellow |

For `DJI_0010_0011_merged`, no SRT truth is available, so the KML contains only:

| Path                   | Color |
| ---                    | --- |
| Estimated drone path   | Red |
| Estimated look-at path | Yellow |

The KML avoids numbered point markers and exports clean path lines.

## Best Offline Result Kept for Comparison

The strongest Test1 result is still the offline/postprocessed pipeline, not the realtime pipeline.

Best offline Test1 result:

| Pipeline                                                 | Mean    | Median  | P90      | Max      |
| ---                                                      | ---:    | ---:    | ---:     | ---:     |
| Top25 + Motion Viterbi + segment calibration + smoothing | 53.54 m | 28.53 m | 105.46 m | 362.88 m |

This is more accurate because it uses the whole sequence and postprocessing. It is not the final realtime method, but it is important to report as the best offline result.

## Why We Moved From Offline to Realtime

The offline Viterbi pipeline gives better accuracy because it can see the whole sequence before selecting the final path. But the assignment asks for realtime navigation. A real drone cannot wait until the end of the flight before estimating its location.

The realtime pipeline therefore had to solve additional problems:

- local tracking after acquisition,
- reacquisition after losing confidence,
- deciding when to abstain,
- avoiding stale held coordinates,
- avoiding optical-flow drift,
- handling repeated landmarks and viewpoint ambiguity.

## Rejected Or Archived Attempts

The archive folders contain earlier approaches that were useful but not retained as the final pipeline:

| Attempt                          | Reason archived                                                                 |
| ---                              | ---                                                                             |
| Greedy realtime visual localizer | Too many jumps and wrong local matches.                                         |
| State machine v1                 | Locked onto wrong regions and propagated errors.                                |
| Beam-only localizer              | Useful helpers were retained, but the full pipeline became region-anchor based. |
| V3/V4 flow-fill                  | Improved coverage but could propagate bad flow during recovery.                 |
| V5 safe landmark                 | Added safety diagnostics, but V6/V7 improved correctness.                       |
| V6 gap reacquire                 | Strong baseline, archived after V7 improved accepted-estimate accuracy.         |

## Limitations

The largest limitation is viewpoint ambiguity. A reference frame can show the same object as the query frame but from a different side or distance. That can be visually meaningful but geographically pose-wrong.

The second limitation is the ground-truth projection itself. If gimbal yaw or exact camera angle is missing, the SRT-derived look-at coordinate is approximate. Errors in the projected ground truth affect reported localization error.

The third limitation is runtime. DINOv2 and LightGlue are expensive. Reference descriptors are cached, but LightGlue verification is still the main bottleneck.

Finally, V7 intentionally lowers coverage. This is a safety choice: the system refuses to publish a coordinate when the visual/geometric evidence is weak.

## Final Deliverables

| File                                                            | Purpose                                             |
| ---                                                             | ---                                                 |
| `README.md`                                                     | Reproduction guide and final pipeline instructions. |
| `docs/final_report.md`                                          | This final report.                                  |
| `docs/literature_review.md`                                     | Literature review and method justification.         |
| `scripts/run_v7_all_tests_autoprep.bat`                         | Main final evaluation command.                      |
| `tools/run_v7_all_tests_good.py`                                | auto-prep and all-tests runner.                     |
| `tools/enhance_realtime_summary.py`                             | Adds extra metrics and metric notes.                |
| `tools/export_final_realtime_kml.py`                            | Exports clean KML paths with requested colors.      |
| `src/realtime_region_anchor_v7_spread_consistency_localizer.py` | Final realtime localizer.                           |

## Conclusion

The project began with an offline AnyLoc-style visual place recognition pipeline and then evolved into a causal realtime system. The offline pipeline remains the most accurate, with a best Test1 result of 53.54 m mean error after Viterbi, segment calibration, and smoothing. The final realtime pipeline is V7 spread consistency, which improves the accepted-estimate mean error on `DJI_Test1_100m` to 97.78 m while explicitly outputting `NO_ESTIMATE` when confidence is low.


