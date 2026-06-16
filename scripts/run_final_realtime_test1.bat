@echo off
setlocal

REM ============================================================
REM Final Test1 only, using OLD/GOOD V7 setup,
REM but writing to the per-video folder:
REM   outputs\realtime\DJI_Test1_100m\
REM ============================================================

cd /d "%~dp0\.."

if exist ".venv-anyloc\Scripts\activate.bat" (
    call ".venv-anyloc\Scripts\activate.bat"
) else (
    echo ERROR: .venv-anyloc was not found.
    exit /b 1
)

python tools\run_v7_all_tests_good.py --only DJI_Test1_100m

endlocal
