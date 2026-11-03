@echo off
title Fotogenie Launcher
cd /d D:\Fotogenie

echo [1/2] Starting ComfyUI Engine on RTX 3060 (Port 8188)...
start "Fotogenie ComfyUI Engine" "D:\Fotogenie\venv\Scripts\python.exe" "D:\Fotogenie\ComfyUI\main.py" --listen 127.0.0.1 --port 8188 --lowvram --reserve-vram 1.0 --disable-smart-memory

echo [2/2] Starting Fotogenie Web Studio (Port 8000)...
start "Fotogenie Web UI" "D:\Fotogenie\venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

timeout /t 4 >nul
start http://localhost:8000
