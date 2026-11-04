@echo off
setlocal
title Fotogenie Launcher
cd /d "%~dp0"
set FOTOGENIE_ALLOW_NSFW_CHECKPOINT=true

:: Ensure Python 3.11 takes precedence in PATH and strip conflicting versions
set PATH=%~dp0venv\Scripts;%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\DLLs;%PATH%

:: Clean out rogue DLLs from venv\Scripts if inadvertently created
if exist "%~dp0venv\Scripts\libffi-8.dll" del /f /q "%~dp0venv\Scripts\libffi-8.dll"
if exist "%~dp0venv\Scripts\python3.dll" del /f /q "%~dp0venv\Scripts\python3.dll"
if exist "%~dp0venv\Scripts\python311.dll" del /f /q "%~dp0venv\Scripts\python311.dll"
if exist "%~dp0venv\Scripts\vcruntime140.dll" del /f /q "%~dp0venv\Scripts\vcruntime140.dll"
if exist "%~dp0venv\Scripts\vcruntime140_1.dll" del /f /q "%~dp0venv\Scripts\vcruntime140_1.dll"

if not exist "logs" mkdir "logs"

echo [1/2] Starting ComfyUI Engine on RTX 3060 (Port 8188)...
start "Fotogenie ComfyUI Engine" "%~dp0venv\Scripts\python.exe" "%~dp0ComfyUI\main.py" --listen 127.0.0.1 --port 8188 --lowvram --reserve-vram 1.0 --disable-smart-memory

echo Waiting for the ComfyUI engine...
powershell -NoProfile -Command "$deadline=(Get-Date).AddSeconds(150); while ((Get-Date) -lt $deadline) { try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8188/system_stats -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; Start-Sleep -Seconds 2 }; exit 1"
if errorlevel 1 (
  echo ComfyUI did not become ready. Check the ComfyUI window for errors.
  pause
  exit /b 1
)

echo [2/2] Starting Fotogenie Web Studio (Port 8000)...
start "Fotogenie Web UI" "%~dp0venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

echo Waiting for the Fotogenie API...
powershell -NoProfile -Command "$deadline=(Get-Date).AddSeconds(60); while ((Get-Date) -lt $deadline) { try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/status -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 1"
if errorlevel 1 (
  echo The Fotogenie API did not become ready. Check the Web UI window for errors.
  pause
  exit /b 1
)

start "" http://127.0.0.1:8000
endlocal
