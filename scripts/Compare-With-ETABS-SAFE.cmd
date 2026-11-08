@echo off
rem Double-click: Structural Studio vs ETABS / SAFE on free-form slabs, ramps, skew walls,
rem columns and loads anywhere.  ETABS 21 / SAFE must be installed and licensed.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Compare-With-ETABS-SAFE.ps1"
pause
