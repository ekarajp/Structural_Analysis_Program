@echo off
rem Drag ETABS .edb / SAP2000 .sdb / SAFE .fdb files onto this file: the installed CSI program
rem writes them as text (tables + .e2k) for Structural Studio.  Output: %LOCALAPPDATA%\StructuralStudio\import
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\structstudio\io\csi_convert.ps1" %*
pause
