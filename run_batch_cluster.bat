@echo off
setlocal enabledelayedexpansion
REM Batch script to run clustering with multiple parameter combinations
REM Classes (k): 1, 2, 3, 4, 5, 10
REM Max iterations: 5, 10, 70, 100

set PYTHON_EXE=C:\Users\1\miniforge3\envs\specviewenv\python.exe
set SCRIPT_PATH=C:/Users/1/PycharmProjects/CoreSpecViewer/batch_cluster.py
set INPUT_FILE=data/134Mcrop/2025-10-31_09-01-32_white_circ_savgol_cr.npy
set OUTPUT_DIR=data/134Mcrop/outputs

REM Loop through k values
for %%k in (1 2 3 4 5 10) do (
    REM Loop through max-iter values
    for %%i in (5 10 70 100) do (
        set OUTPUT_FILE=!OUTPUT_DIR!\cropped_clusters_k%%k_iter%%i.png
        echo Running: k=%%k, max-iter=%%i
        "%PYTHON_EXE%" "%SCRIPT_PATH%" "%INPUT_FILE%" -k %%k --max-iter %%i -o "!OUTPUT_FILE!"
        echo.
    )
)

echo All clustering runs completed!
pause

