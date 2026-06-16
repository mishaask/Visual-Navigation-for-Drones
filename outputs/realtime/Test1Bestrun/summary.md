# Realtime V7 Summary — DJI_Test1_100m

## Overall

- Frames processed: `739`
- Valid estimates: `441`
- No-estimate frames: `298`
- Coverage: `59.68%`

## Look-at / camera-center error

| Metric | Value |
| --- | ---: |
| Mean error | 97.78 m |
| Median error | 74.97 m |
| P90 error | 229.18 m |
| P95 error | 238.34 m |
| Max error | 396.61 m |
| % under 100 m | 70.29% |
| % under 50 m | 25.40% |
| % under 10 m | 1.13% |
| % under 5 m | 0.00% |

## Drone-position error

| Metric | Value |
| --- | ---: |
| Mean drone error | 121.05 m |
| Median drone error | 89.91 m |
| P90 drone error | 300.67 m |
| P95 drone error | 308.75 m |
| Max drone error | 424.34 m |
| Drone % under 100 m | 55.33% |
| Drone % under 50 m | 34.69% |
| Drone % under 10 m | 2.04% |
| Drone % under 5 m | 0.45% |

## Metric notes

- Valid estimate frames are frames where the realtime localizer output an estimated coordinate. NO_ESTIMATE frames are intentionally skipped because the system was uncertain.
- Coverage is valid_estimate_frames divided by frames_processed.
- Mean error is the arithmetic average distance between the estimated look-at point and the SRT-derived ground-truth look-at point, in metres.
- Median error is the middle error value; half of evaluated estimates are below it and half are above it.
- P90 error means 90 percent of evaluated estimates have error less than or equal to this value.
- P95 error means 95 percent of evaluated estimates have error less than or equal to this value.
- % under 100m / 50m / 10m / 5m is the percentage of evaluated valid estimates whose error is less than or equal to that threshold. It is not divided by all frames, only by frames that have a valid estimate and ground truth.
- Drone-position error compares estimated drone GPS position to SRT drone GPS position. Look-at/camera-center error compares estimated camera-center ground coordinate to the SRT-derived camera-center ground coordinate.
- When no SRT/ground truth is available, accuracy metrics are N/A and only estimated paths are exported.
