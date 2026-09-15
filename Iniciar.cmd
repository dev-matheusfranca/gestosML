@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Execute Instalar.ps1 antes de abrir o GestureLab.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m gesturelab
if errorlevel 1 pause
