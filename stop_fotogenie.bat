@echo off
title Stop Fotogenie
echo Stopping Fotogenie services on port 8000 and 8188...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000, 8188 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"
echo Fotogenie services stopped.
ping 127.0.0.1 -n 3 >nul

