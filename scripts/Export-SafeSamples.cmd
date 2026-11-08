@echo off
rem Re-converts only the SAFE sample models to tables (tests\verification\data\safe).
rem The converter checks the numbers against the SAFE object API (SAFE 22 drops trailing
rem zeros of whole numbers in GetTableForEditingArray) and reports the route it used.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Export-CsiSamples.ps1" -Only safe
pause
