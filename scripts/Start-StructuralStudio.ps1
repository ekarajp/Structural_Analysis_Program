# Starts Structural Studio (Python desktop app).  On the first run it installs the
# package and its UI libraries (PySide6, VTK, openpyxl) into the project's Python
# at E:\REA ETABS\tools\python (internet needed once, about 300 MB).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File "E:\REA ETABS\studio\scripts\Start-StructuralStudio.ps1" [model]
param([string]$Model = "")
# 'Continue': in Windows PowerShell 5.1 anything a program writes to stderr (a Python traceback,
# pip warnings) becomes a terminating error under 'Stop' - exit codes are checked instead.
$ErrorActionPreference = 'Continue'
$Studio = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)   # ...\studio
$Repo   = Split-Path -Parent $Studio                                              # E:\REA ETABS

# 1. find Python: the project's venv first, then the launcher / PATH
$Py = Join-Path $Repo 'tools\python\Scripts\python.exe'
if (-not (Test-Path $Py)) {
    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) { $Py = (& py -3 -c "import sys; print(sys.executable)").Trim() }
}
if (-not (Test-Path $Py)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $Py = $cmd.Source }
}
if (-not $Py -or -not (Test-Path $Py)) {
    Write-Host 'Python 3.11+ was not found. Install it from https://www.python.org and run this again.' -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}
Write-Host "Python: $Py"

# 2. first run (or after an update added a library): install Structural Studio + UI libraries
function Test-Ready {
    $null = & $Py -c "import PySide6, vtk, numpy, scipy, openpyxl, structstudio.app.mainwindow" 2>&1
    return ($LASTEXITCODE -eq 0)
}
if (-not (Test-Ready)) {
    $logDir = Join-Path $env:LOCALAPPDATA 'StructuralStudio'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $log = Join-Path $logDir 'install.log'
    Write-Host 'Installing Structural Studio and its UI libraries (PySide6, VTK) - first run only,' -ForegroundColor Cyan
    Write-Host 'a few minutes, internet needed ...' -ForegroundColor Cyan
    & $Py -m pip install --upgrade pip 2>&1 | Tee-Object -FilePath $log
    & $Py -m pip install -e "$Studio[app]" 2>&1 | Tee-Object -FilePath $log -Append
    if (-not (Test-Ready)) {
        Write-Host ''
        Write-Host 'Structural Studio could not be started. The reason is below; the install log is' -ForegroundColor Red
        Write-Host "$log" -ForegroundColor Red
        & $Py -c "import PySide6, vtk, numpy, scipy, openpyxl, structstudio.app.mainwindow" 2>&1 | Out-Host
        Read-Host 'Press Enter to close'
        exit 1
    }
}

# 3. start the app without a console window
$Pyw = Join-Path (Split-Path $Py) 'pythonw.exe'
if (-not (Test-Path $Pyw)) { $Pyw = $Py }
$appArgs = @('-m', 'structstudio.app')
if ($Model) { $appArgs += ('"' + $Model + '"') }
Start-Process -FilePath $Pyw -ArgumentList $appArgs -WorkingDirectory $Studio
