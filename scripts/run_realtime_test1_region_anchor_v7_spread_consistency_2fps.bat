@echo off
setlocal
cd /d "%~dp0\.."

REM ============================================================
REM OLD / GOOD final V7 Test1 runner.
REM Uses:
REM   data\processed\realtime_reference_regions_v2.csv
REM and the original V7 parameter set.
REM ============================================================

if exist ".venv-anyloc\Scripts\activate.bat" (
    call ".venv-anyloc\Scripts\activate.bat"
) else (
    echo ERROR: .venv-anyloc was not found.
    exit /b 1
)

set RUN_DIR=outputs\realtime\Test1_region_anchor_v7_spread_consistency_2fps
mkdir "%RUN_DIR%" 2>nul
mkdir "%RUN_DIR%\query_frames_test1_2fps" 2>nul
mkdir outputs\anyloc 2>nul

REM Always rebuild the old/good reference-region file so we do not accidentally use
REM the bad all-tests reference_regions_all_0006_0009.csv setup.
call scripts\run_build_reference_regions_v2_test1.bat
if errorlevel 1 (
    echo ERROR: Reference-region build failed.
    exit /b 1
)

python src\realtime_region_anchor_v7_spread_consistency_localizer.py ^
  --reference-regions data\processed\realtime_reference_regions_v2.csv ^
  --query-video data\raw\DJI_Test1_100m.MP4 ^
  --truth-manifest data\processed\DJI_Test1_100m_frame_manifest_1fps.csv ^
  --reference-descriptor-cache outputs\anyloc\realtime_reference_regions_v2_dino_descriptors.npy ^
  --query-frame-dir "%RUN_DIR%\query_frames_test1_2fps" ^
  --output-csv "%RUN_DIR%\DJI_Test1_100m_realtime_predictions.csv" ^
  --summary-json "%RUN_DIR%\DJI_Test1_100m_realtime_summary.json" ^
  --debug-jsonl "%RUN_DIR%\DJI_Test1_100m_region_anchor_debug.jsonl" ^
  --sample-fps 2 ^
  --region-grid-m 90 ^
  --global-topk 100 ^
  --candidate-pool-limit 90 ^
  --locked-region-ring 1 ^
  --recovery-region-ring 3 ^
  --recovery-after 5 ^
  --fail-limit 10 ^
  --acquire-window 5 ^
  --acquire-min-region-votes 3 ^
  --acquire-min-geometry-votes 2 ^
  --transition-confirm-window 3 ^
  --transition-min-votes 2 ^
  --acquire-vote-topn 45 ^
  --acquire-verified-vote-bonus 1.0 ^
  --acquire-geometry-vote-bonus 3.0 ^
  --lg-topk 12 ^
  --lightglue-every 2 ^
  --lg-min-inliers 10 ^
  --lg-min-ratio 0.08 ^
  --require-geometry-verified ^
  --use-geometry-scoring ^
  --geometry-verified-bonus 2.2 ^
  --geometry-failed-penalty 3.0 ^
  --geometry-min-inliers 8 ^
  --geometry-min-inlier-ratio 0.08 ^
  --max-homography-reproj-rmse 10.0 ^
  --min-query-inlier-width-frac 0.14 ^
  --min-query-inlier-height-frac 0.08 ^
  --min-reference-inlier-width-frac 0.14 ^
  --min-reference-inlier-height-frac 0.08 ^
  --min-altitude-spread-scale 0.25 ^
  --require-spread-balance ^
  --spread-balance-tolerance 2.5 ^
  --same-altitude-ratio-threshold 1.35 ^
  --dino-weight 4.0 ^
  --inlier-weight 1.15 ^
  --ratio-weight 1.0 ^
  --unverified-penalty 2.8 ^
  --motion-weight 7.0 ^
  --max-step-m 45 ^
  --strong-jump-m 180 ^
  --flow-contradiction-enabled ^
  --low-flow-px 12 ^
  --camera-hfov-deg 82 ^
  --default-altitude-m 100 ^
  --oblique-scale 1.35 ^
  --low-altitude-m 45 ^
  --low-altitude-ring-bonus 1 ^
  --fast-flow-px 22 ^
  --very-fast-flow-px 45 ^
  --fast-flow-ring-bonus 1 ^
  --very-fast-flow-ring-bonus 2 ^
  --fast-speed-mps 10 ^
  --very-fast-speed-mps 20 ^
  --fast-speed-ring-bonus 1 ^
  --very-fast-speed-ring-bonus 2 ^
  --bad-frame-ring-bonus 2 ^
  --max-dynamic-region-ring 6 ^
  --flow-calib-min-px 4 ^
  --flow-calib-max-jump-m 180 ^
  --max-flow-pred-speed-mps 35 ^
  --no-estimate-during-acquire ^
  --flow-propagate-holds ^
  --flow-propagation-min-quality 0.55 ^
  --flow-propagation-min-points 40 ^
  --flow-propagation-max-step-m 15 ^
  --flow-propagation-max-bad-count 3 ^
  --disable-flow-propagation-in-recovery ^
  --max-flow-hold-speed-mps 35 ^
  --max-flow-speed-mps 35 ^
  --hold-max-bad-count 3 ^
  --projected-center-offset-warning-frac 0.28

if errorlevel 1 (
    echo ERROR: V7 realtime localizer failed.
    exit /b 1
)

REM Old KML exporter, kept for compatibility with the original run.
python tools\export_realtime_kml.py ^
  --predictions "%RUN_DIR%\DJI_Test1_100m_realtime_predictions.csv" ^
  --truth-manifest data\processed\DJI_Test1_100m_frame_manifest_1fps.csv ^
  --reference-manifest 0006=data\processed\DJI_0006_frame_manifest_1fps.csv ^
  --reference-manifest 0007=data\processed\DJI_0007_frame_manifest_1fps.csv ^
  --reference-manifest 0008=data\processed\DJI_0008_frame_manifest_1fps.csv ^
  --reference-manifest 0009=data\processed\DJI_0009_frame_manifest_1fps.csv ^
  --output-kml "%RUN_DIR%\DJI_Test1_100m_realtime_paths.kml" ^
  --line-every 1 ^
  --point-every 30

REM New enhanced summary and clean final KML. These do not change localization results.
if exist tools\enhance_realtime_summary.py (
  python tools\enhance_realtime_summary.py ^
    --predictions "%RUN_DIR%\DJI_Test1_100m_realtime_predictions.csv" ^
    --base-summary "%RUN_DIR%\DJI_Test1_100m_realtime_summary.json" ^
    --output-json "%RUN_DIR%\summary.json" ^
    --output-md "%RUN_DIR%\summary.md" ^
    --test-name DJI_Test1_100m
)

if exist tools\export_final_realtime_kml.py (
  python tools\export_final_realtime_kml.py ^
    --predictions "%RUN_DIR%\DJI_Test1_100m_realtime_predictions.csv" ^
    --output "%RUN_DIR%\paths.kml" ^
    --name DJI_Test1_100m
)

echo.
echo ============================================================
echo Finished OLD/GOOD V7 Test1 setup.
echo Output folder:
echo %RUN_DIR%
echo ============================================================
echo.

endlocal
