# Opens a model text file (.e2k) - or an .edb - in the installed ETABS through its API, runs it
# and writes beside it, in -OutDir:  <name>.edb, <name>.etabs.e2k (the model as ETABS wrote it
# back), <name>.<table>.csv (Story Drifts, Story Forces, Joint Displacements, Base Reactions,
# Modal Periods ... in kN-m) and run_etabs.log.  The original file is not changed (a copy is
# opened).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File "F:\REA ETABS\studio\scripts\Run-EtabsModel.ps1" <file>
#
param([Parameter(Mandatory = $true)][string]$File, [string]$OutDir = "")
$ErrorActionPreference = 'Continue'
$Studio = Split-Path -Parent $PSScriptRoot
. (Join-Path $Studio 'structstudio\io\csi_api.ps1')
$src = (Resolve-Path -LiteralPath $File).Path
$name = [System.IO.Path]::GetFileNameWithoutExtension($src)
if (-not $OutDir) { $OutDir = Join-Path ([System.IO.Path]::GetDirectoryName($src)) "$name.etabs" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Log = Join-Path $OutDir 'run_etabs.log'
"$(Get-Date -Format s) start $src" | Set-Content -Path $Log
function Say([string]$text) { Write-Host $text; Add-Content -Path $Log -Value $text }
function Rc([string]$what, $rc) { Say ("{0,-44} rc={1}" -f $what, $rc); return $rc }
function Csv-Line { param([object[]]$cells) ($cells | ForEach-Object { '"' + ([string]$_).Replace('"', '""') + '"' }) -join ',' }
$info = $Programs['.edb']
$found = Find-Program $info
if ($null -eq $found) { Say 'ETABS is not installed'; exit 1 }
Say "Using $($found.Label)"
$Hidden = $false
$app = Start-CsiProgram $found $info
$ns = [System.IO.Path]::GetFileNameWithoutExtension($found.Dll)
function E([string]$enum, [int]$v) { return [Enum]::ToObject(("$ns.$enum" -as [type]), $v) }
$exit = 0
try {
    $m = $app.SapModel
    $copy = Join-Path $OutDir ([System.IO.Path]::GetFileName($src))
    Copy-Item -LiteralPath $src -Destination $copy -Force
    $rc = Rc "OpenFile $([System.IO.Path]::GetFileName($copy))" $m.File.OpenFile($copy)
    if ($rc -ne 0) { throw "ETABS could not open the file (rc=$rc)" }
    try {
        $n = 0; [string[]]$tn = @(); $null = $m.Tower.GetNameList([ref]$n, [ref]$tn); Say "Towers: $($tn -join ', ')"
        foreach ($t in $tn) {
            $null = $m.Tower.SetActiveTower($t)
            $ns2 = 0; [string[]]$sn = @(); [double[]]$se = @(); [double[]]$sh = @(); [bool[]]$im = @(); [string[]]$si = @(); [bool[]]$sa = @(); [double[]]$sp = @()
            $null = $m.Story.GetStories([ref]$ns2, [ref]$sn, [ref]$se, [ref]$sh, [ref]$im, [ref]$si, [ref]$sa, [ref]$sp)
            Say ("  stories of {0}: {1}" -f $t, (($sn | ForEach-Object -Begin { $k = 0 } -Process { "$_@$($se[$k])"; $k++ }) -join ', '))
        }
    } catch { Say "towers: $($_.Exception.Message)" }
    $edb = Join-Path $OutDir "$name.edb"
    Rc 'File.Save edb' $m.File.Save($edb) | Out-Null
    $e2k = Join-Path $OutDir "$name.etabs.e2k"
    Rc 'File.Save e2k (written back)' $m.File.Save($e2k) | Out-Null
    $et = Join-Path $OutDir ("$name.etabs" + '.$et')
    if (-not (Test-Path -LiteralPath $e2k) -and (Test-Path -LiteralPath $et)) { Copy-Item -LiteralPath $et -Destination $e2k -Force }
    $null = $m.File.Save($edb)
    $null = $m.SetPresentUnits((E 'eUnits' 6))   # tables in kN-m
    Rc 'RunAnalysis' $m.Analyze.RunAnalysis() | Out-Null
    $db = $m.DatabaseTables
    [string[]]$cases = @('EQX', 'Dead', 'Modal')
    try { $null = $db.SetLoadCasesSelectedForDisplay([ref]$cases) } catch { Say "SetLoadCasesSelectedForDisplay: $($_.Exception.Message)" }
    foreach ($key in @('Story Drifts', 'Story Forces', 'Joint Displacements', 'Base Reactions',
                       'Modal Periods And Frequencies', 'Diaphragm Center Of Mass Displacements', 'Story Definitions',
                       'Tower and Base Story Definitions', 'Tower Definitions')) {
        try {
            $ver = 0; [string[]]$req = @(); [string[]]$inc = @(); $nrec = 0; [string[]]$data = @()
            $rc = $db.GetTableForDisplayArray($key, [ref]$req, '', [ref]$ver, [ref]$inc, [ref]$nrec, [ref]$data)
            if ($rc -ne 0 -or $nrec -le 0) { Say "table '$key': rc=$rc rows=$nrec"; continue }
            $lines = @(Csv-Line $inc)
            for ($r = 0; $r -lt $nrec; $r++) { $lines += Csv-Line ($data[($r * $inc.Count)..(($r + 1) * $inc.Count - 1)]) }
            $file = Join-Path $OutDir ("$name." + ($key -replace '[^A-Za-z0-9]+', '_') + ".csv")
            [IO.File]::WriteAllLines($file, [string[]]$lines)
            Say "table '$key': $nrec rows -> $file"
        } catch { Say "table '$key': $($_.Exception.Message)" }
    }
} catch {
    Say "FAIL $($_.Exception.Message)"
    $exit = 1
} finally {
    try { $app.ApplicationExit($false) | Out-Null } catch { }
}
Say "$(Get-Date -Format s) done"
exit $exit
