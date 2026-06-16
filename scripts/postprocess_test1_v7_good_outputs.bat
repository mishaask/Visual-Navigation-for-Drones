@echo off
setlocal
cd /d "%~dp0\.."

REM Postprocess old/good V7 Test1 output after the localizer has already finished.

if exist ".venv-anyloc\Scripts\activate.bat" (
    call ".venv-anyloc\Scripts\activate.bat"
)

set RUN_DIR=outputs\realtime\Test1_region_anchor_v7_spread_consistency_2fps
set PRED=%RUN_DIR%\DJI_Test1_100m_realtime_predictions.csv
set RAW_SUMMARY=%RUN_DIR%\DJI_Test1_100m_realtime_summary.json

if not exist "%PRED%" (
    echo ERROR: Missing predictions:
    echo %PRED%
    exit /b 1
)

if exist tools\enhance_realtime_summary.py (
    python tools\enhance_realtime_summary.py ^
      --predictions "%PRED%" ^
      --base-summary "%RAW_SUMMARY%" ^
      --output-json "%RUN_DIR%\summary.json" ^
      --output-md "%RUN_DIR%\summary.md" ^
      --test-name DJI_Test1_100m
)

if exist tools\export_final_realtime_kml.py (
    python tools\export_final_realtime_kml.py ^
      --predictions "%PRED%" ^
      --output "%RUN_DIR%\paths.kml" ^
      --name DJI_Test1_100m
)

echo.
echo Done.
echo Summary:
echo %RUN_DIR%\summary.md
echo KML:
echo %RUN_DIR%\paths.kml

endlocal
