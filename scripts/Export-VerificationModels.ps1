# Exports the ETABS verification examples to text (.e2k) using the official ETABS API.
# Needs an installed, licensed ETABS 21.  Nothing outside F:\REA ETABS is written.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File "F:\REA ETABS\studio\scripts\Export-VerificationModels.ps1"
#
$ErrorActionPreference = 'Stop'
$EtabsDir = 'C:\Program Files\Computers and Structures\ETABS 21'
$Source   = 'F:\REA ETABS\reference\etabs21\Verification\Analysis Examples'
$Dest     = 'F:\REA ETABS\studio\tests\verification\data\etabs'
$Log      = Join-Path $Dest 'export.log'

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
"$(Get-Date -Format s) start" | Set-Content -Path $Log

Add-Type -Path (Join-Path $EtabsDir 'ETABSv1.dll')
$helper = New-Object ETABSv1.Helper
$app = $helper.CreateObject((Join-Path $EtabsDir 'ETABS.exe'))
if ($null -eq $app) { throw 'Could not start an ETABS instance through the API' }
$rc = $app.ApplicationStart()
if ($rc -ne 0) { throw "ApplicationStart returned $rc" }
$sap = $app.SapModel

try {
    foreach ($f in Get-ChildItem -Path $Source -Filter 'Example *.edb' | Sort-Object Name) {
        $copy = Join-Path $Dest $f.Name
        Copy-Item -LiteralPath $f.FullName -Destination $copy -Force
        $rc = $sap.File.OpenFile($copy)
        if ($rc -ne 0) { "$($f.Name): OpenFile rc=$rc" | Add-Content $Log; continue }
        # saving with an .e2k extension writes the text model file
        $e2k = Join-Path $Dest ($f.BaseName.Replace(' ', '') + '.e2k')
        $rc = $sap.File.Save($e2k)
        "$($f.Name): Save e2k rc=$rc -> $e2k" | Add-Content $Log
        # also save the .EDB copy so ETABS writes the .$et companion text file
        $rc2 = $sap.File.Save($copy)
        "$($f.Name): Save edb rc=$rc2" | Add-Content $Log
    }
}
finally {
    $app.ApplicationExit($false) | Out-Null
    "$(Get-Date -Format s) done" | Add-Content $Log
}
Get-Content $Log
