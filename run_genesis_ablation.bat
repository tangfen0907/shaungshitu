@echo off
setlocal

REM One-click Genesis ablation runner.
REM It uses the fukan conda environment found on this machine.

cd /d "%~dp0"

set "PYTHON_EXE=D:\ProgramData\anaconda3\envs\fukan\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python environment not found: %PYTHON_EXE%
  echo Please edit run_genesis_ablation.bat and set PYTHON_EXE to your training Python.
  pause
  exit /b 1
)

echo [Genesis Ablation] Python: %PYTHON_EXE%
echo [Genesis Ablation] Running core ablation suite...
echo.

"%PYTHON_EXE%" scripts\run_genesis_ablation.py --suite core

echo.
echo [Genesis Ablation] Finished.
pause
