# One click: builds the free-form / inclined comparison models in Structural Studio,
# the same models in ETABS and SAFE (official API), runs everything and writes
#   F:\REA ETABS\studio\tests\verification\data\compare\report.txt
# (opened in Notepad at the end).  ETABS / SAFE open on screen while they work.
$ErrorActionPreference = 'Continue'
$Studio = Split-Path -Parent $PSScriptRoot
$Repo   = Split-Path -Parent $Studio
$Out    = Join-Path $Studio 'tests\verification\data\compare'

$Py = Join-Path $Repo 'tools\python\Scripts\python.exe'
if (-not (Test-Path $Py)) { $c = Get-Command py -ErrorAction SilentlyContinue; if ($c) { $Py = (& py -3 -c "import sys; print(sys.executable)").Trim() } }
if (-not (Test-Path $Py)) { $c = Get-Command python -ErrorAction SilentlyContinue; if ($c) { $Py = $c.Source } }
if (-not $Py -or -not (Test-Path $Py)) { Write-Host 'Python was not found - run Start-StructuralStudio once first.' -ForegroundColor Red; exit 1 }
$env:PYTHONPATH = $Studio
$env:PYTHONIOENCODING = 'utf-8'

New-Item -ItemType Directory -Force -Path $Out | Out-Null
Get-ChildItem -Path $Out -Recurse -File -Include *.csv, *.log, *.png, report.txt -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host '1/3  Structural Studio: building and analysing the models ...' -ForegroundColor Cyan
& $Py (Join-Path $PSScriptRoot 'csi_compare.py') build $Out
if ($LASTEXITCODE -ne 0) { Write-Host 'Structural Studio part failed (see above).' -ForegroundColor Red; exit 1 }

Write-Host '2/3  ETABS / SAFE: the same models through the API (the programs open on screen) ...' -ForegroundColor Cyan
& (Join-Path $PSScriptRoot 'Run-CsiCompare.ps1') -Folder $Out

Write-Host '3/3  Comparing ...' -ForegroundColor Cyan
& $Py (Join-Path $PSScriptRoot 'csi_compare.py') compare $Out | Select-Object -Last 12
$report = Join-Path $Out 'report.txt'
if (Test-Path $report) {
    Write-Host "Report: $report" -ForegroundColor Green
    Start-Process notepad.exe $report
}
Write-Host "Send me $Out (report.txt, and the etabs / safe folders with the .log files) if anything failed."
