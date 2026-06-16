@echo off
setlocal
cd /d "%~dp0\.."

REM ============================================================
REM Build the OLD / GOOD Test1 realtime reference regions.
REM This intentionally writes:
REM   data\processed\realtime_reference_regions_v2.csv
REM using --segment-frame-span 3000.
REM Do NOT replace this with reference_regions_all_0006_0009.csv for final Test1.
REM ============================================================

mkdir data\processed 2>nul

set M6=data\processed\DJI_0006_frame_manifest_1fps.csv
set M7=data\processed\DJI_0007_frame_manifest_1fps.csv
set M8=data\processed\DJI_0008_frame_manifest_1fps.csv
set M9=data\processed\DJI_0009_frame_manifest_1fps.csv

if not exist "%M6%" (
  echo ERROR: Missing %M6%
  exit /b 1
)
if not exist "%M7%" (
  echo ERROR: Missing %M7%
  exit /b 1
)
if not exist "%M8%" (
  echo ERROR: Missing %M8%
  exit /b 1
)
if not exist "%M9%" (
  echo ERROR: Missing %M9%
  exit /b 1
)

echo.
echo ============================================================
echo Building OLD/GOOD Test1 reference regions
echo Output: data\processed\realtime_reference_regions_v2.csv
echo ============================================================
echo.

python tools\build_reference_regions_v2.py ^
  --reference-manifest 0006=%M6% ^
  --reference-manifest 0007=%M7% ^
  --reference-manifest 0008=%M8% ^
  --reference-manifest 0009=%M9% ^
  --output-csv data\processed\realtime_reference_regions_v2.csv ^
  --summary-json data\processed\realtime_reference_regions_v2_summary.json ^
  --grid-m 90 ^
  --segment-frame-span 3000

if errorlevel 1 (
  echo ERROR: Failed to build realtime_reference_regions_v2.csv
  exit /b 1
)

echo.
echo Done building old/good reference regions.
echo.

endlocal
