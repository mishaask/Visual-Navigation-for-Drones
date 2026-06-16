@echo off
setlocal

REM ============================================================
REM Run all requested videos using the OLD/GOOD V7 setup.
REM Outputs are written per video:
REM   outputs\realtime\DJI_0006\
REM   outputs\realtime\DJI_0007\
REM   outputs\realtime\DJI_0008\
REM   outputs\realtime\DJI_0009\
REM   outputs\realtime\DJI_Test1_100m\
REM   outputs\realtime\DJI_0010_0011_merged\
REM ============================================================

cd /d "%~dp0\.."

if exist ".venv-anyloc\Scripts\activate.bat" (
    call ".venv-anyloc\Scripts\activate.bat"
) else (
    echo ERROR: .venv-anyloc was not found.
    exit /b 1
)

python tools\run_v7_all_tests_good.py

endlocal
