@echo off
setlocal

REM One-click targeted Genesis ablation runner.
REM This suite sweeps:
REM   - stage2_ap_margin around the 0.1 baseline
REM   - fair-length AB / ABB / ABBB / AAB schedules
REM   - Stage2-B reconstruction strength lambda_rec_B

cd /d "%~dp0"

set "PYTHON_EXE=D:\ProgramData\anaconda3\envs\fukan\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python environment not found: %PYTHON_EXE%
  echo Please edit run_genesis_targeted_ablation.bat and set PYTHON_EXE to your training Python.
  pause
  exit /b 1
)

echo [Genesis Targeted Ablation] Python: %PYTHON_EXE%
echo [Genesis Targeted Ablation] Running targeted suite...
echo.

"%PYTHON_EXE%" scripts\run_genesis_ablation.py --suite targeted --rank_by score_proto_v2.MCC_score

echo.
echo [Genesis Targeted Ablation] Finished.
pause
