# Builds a small two-tower model in the installed ETABS through its API, runs it and writes:
#   TwoTowers.edb / TwoTowers.e2k          - the model and ETABS' own text model (the tower format)
#   TwoTowers.ETABS.tables.txt             - every model table (csi_convert.ps1)
#   TwoTowers.<table>.csv                  - Story Drifts, Story Forces, Joint Displacements, Base
#                                            Reactions, Modal Periods (ETABS results, kN-m)
#   make_tower_sample.log                  - what every API call returned
# Tower T1: 3 x 3 grid at x 0..12, stories 4 / 3 / 3 (z = 4, 7, 10), diaphragm D1.
# Tower T2: 2 x 2 grid at x 30..36, stories 3.5 / 3.5 (z = 3.5, 7), diaphragm D2 - the 7 m
# level is in both towers.  Columns 40x40, beams 30x50, 150 mm shell slabs, fixed bases,
# lateral loads EQX at the corner joints, a modal case.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File "F:\REA ETABS\studio\scripts\Make-TowerSample.ps1"
#
# Nothing outside the output folder (default F:\REA ETABS\samples\towers) is written.
param([string]$OutDir = "F:\REA ETABS\samples\towers")
$ErrorActionPreference = 'Continue'
$Studio = Split-Path -Parent $PSScriptRoot
. (Join-Path $Studio 'structstudio\io\csi_api.ps1')
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Log = Join-Path $OutDir 'make_tower_sample.log'
"$(Get-Date -Format s) start" | Set-Content -Path $Log
function Say([string]$text) { Write-Host $text; Add-Content -Path $Log -Value $text }
function Rc([string]$what, $rc) { Say ("{0,-44} rc={1}" -f $what, $rc); return $rc }
function Csv-Line { param([object[]]$cells) ($cells | ForEach-Object { '"' + ([string]$_).Replace('"', '""') + '"' }) -join ',' }

$info = $Programs['.edb']
$found = Find-Program $info
if ($null -eq $found) { Say 'ETABS is not installed'; exit 1 }
Say "Using $($found.Label) ($($found.Exe))"
$Hidden = $false
$app = Start-CsiProgram $found $info
$ns = [System.IO.Path]::GetFileNameWithoutExtension($found.Dll)
function E([string]$enum, [int]$v) { return [Enum]::ToObject(("$ns.$enum" -as [type]), $v) }
$exit = 0
try {
    $m = $app.SapModel
    try { $v = ''; $b = 0.0; $null = $m.GetVersion([ref]$v, [ref]$b); Say "ETABS version $v" } catch { }
    Rc 'InitializeNewModel(kN_m_C)' $m.InitializeNewModel((E 'eUnits' 6)) | Out-Null
    Rc 'NewGridOnly(3 stories, 3 m, bottom 4 m, 4x4 lines @ 4 m)' $m.File.NewGridOnly(3, 3.0, 4.0, 4, 4, 4.0, 4.0) | Out-Null
    # ---- properties
    Rc 'SetMaterial CONC' $m.PropMaterial.SetMaterial('CONC', (E 'eMatType' 2), -1, '', '') | Out-Null
    Rc 'SetMPIsotropic CONC' $m.PropMaterial.SetMPIsotropic('CONC', 25e6, 0.2, 1e-5, 0) | Out-Null
    Rc 'SetWeightAndMass CONC' $m.PropMaterial.SetWeightAndMass('CONC', 1, 24.0, 0) | Out-Null
    Rc 'SetRectangle C40' $m.PropFrame.SetRectangle('C40', 'CONC', 0.4, 0.4, -1, '', '') | Out-Null
    Rc 'SetRectangle B30x50' $m.PropFrame.SetRectangle('B30x50', 'CONC', 0.5, 0.3, -1, '', '') | Out-Null
    Rc 'SetSlab S15 (ShellThin)' $m.PropArea.SetSlab('S15', (E 'eSlabType' 0), (E 'eShellType' 1), 'CONC', 0.15, -1, '', '') | Out-Null
    Rc 'SetDiaphragm D1' $m.Diaphragm.SetDiaphragm('D1', $false) | Out-Null
    Rc 'SetDiaphragm D2' $m.Diaphragm.SetDiaphragm('D2', $false) | Out-Null
    Rc 'LoadPatterns.Add EQX' $m.LoadPatterns.Add('EQX', (E 'eLoadPatternType' 8), 0, $true) | Out-Null
    # ---- towers
    Rc 'AllowMultipleTowers(true)' $m.Tower.AllowMultipleTowers($true, '', $true) | Out-Null
    Rc 'AddNewTower T2 (2 stories, 3.5, 3.5)' $m.Tower.AddNewTower('T2', 2, 3.5, 3.5) | Out-Null
    $n = 0; [string[]]$tn = @()
    try { $null = $m.Tower.GetNameList([ref]$n, [ref]$tn); Say "Towers: $($tn -join ', ')" } catch { Say "GetNameList: $($_.Exception.Message)" }

    function Build-Tower([string]$tower, [double[]]$xs, [double[]]$ys, [double[]]$zs, [string]$dia, [double]$fx) {
        Rc "SetActiveTower $tower" $m.Tower.SetActiveTower($tower) | Out-Null
        try {
            $ns2 = 0; [string[]]$sn = @(); [double[]]$se = @(); [double[]]$sh = @(); [bool[]]$im = @(); [string[]]$si = @(); [bool[]]$sa = @(); [double[]]$sp = @()
            $null = $m.Story.GetStories([ref]$ns2, [ref]$sn, [ref]$se, [ref]$sh, [ref]$im, [ref]$si, [ref]$sa, [ref]$sp)
            Say ("  stories of {0}: {1}" -f $tower, (($sn | ForEach-Object -Begin { $k = 0 } -Process { "$_@$($se[$k])"; $k++ }) -join ', '))
        } catch { Say "  GetStories: $($_.Exception.Message)" }
        $bad = 0
        for ($k = 1; $k -lt $zs.Count; $k++) {
            $z0 = $zs[$k - 1]; $z1 = $zs[$k]
            foreach ($x in $xs) { foreach ($y in $ys) {
                $name = ''
                $rc = $m.FrameObj.AddByCoord($x, $y, $z0, $x, $y, $z1, [ref]$name, 'C40', '', 'Global')
                if ($rc -ne 0) { $bad++ }
                if ($k -eq 1) {
                    $p1 = ''; $p2 = ''; $null = $m.FrameObj.GetPoints($name, [ref]$p1, [ref]$p2)
                    [bool[]]$fix = @($true, $true, $true, $true, $true, $true)
                    $null = $m.PointObj.SetRestraint($p1, [ref]$fix, (E 'eItemType' 0))
                }
            } }
            foreach ($y in $ys) { for ($i = 1; $i -lt $xs.Count; $i++) {
                $name = ''; if ($m.FrameObj.AddByCoord($xs[$i - 1], $y, $z1, $xs[$i], $y, $z1, [ref]$name, 'B30x50', '', 'Global') -ne 0) { $bad++ } } }
            foreach ($x in $xs) { for ($j = 1; $j -lt $ys.Count; $j++) {
                $name = ''; if ($m.FrameObj.AddByCoord($x, $ys[$j - 1], $z1, $x, $ys[$j], $z1, [ref]$name, 'B30x50', '', 'Global') -ne 0) { $bad++ } } }
            for ($i = 1; $i -lt $xs.Count; $i++) { for ($j = 1; $j -lt $ys.Count; $j++) {
                [double[]]$X = @($xs[$i - 1], $xs[$i], $xs[$i], $xs[$i - 1])
                [double[]]$Y = @($ys[$j - 1], $ys[$j - 1], $ys[$j], $ys[$j])
                [double[]]$Z = @($z1, $z1, $z1, $z1)
                $name = ''
                if ($m.AreaObj.AddByCoord(4, [ref]$X, [ref]$Y, [ref]$Z, [ref]$name, 'S15', '', 'Global') -ne 0) { $bad++; continue }
                $null = $m.AreaObj.SetDiaphragm($name, $dia, (E 'eItemType' 0))
            } }
            # lateral load at the corner joint (last x, first y) of this level
            $pn = ''
            $null = $m.PointObj.AddCartesian($xs[-1], $ys[0], $z1, [ref]$pn, '', 'Global', $false, 0)  # (merges with the joint there)
            [double[]]$f = @(($fx * $z1 / $zs[-1]), (0.25 * $fx * $z1 / $zs[-1]), 0, 0, 0, 0)
            $null = $m.PointObj.SetLoadForce($pn, 'EQX', [ref]$f, $false, 'Global', (E 'eItemType' 0))
        }
        Say "  $tower built ($bad failed object calls)"
    }
    Build-Tower 'T1' @(0, 6, 12) @(0, 6, 12) @(0, 4, 7, 10) 'D1' 30.0
    Build-Tower 'T2' @(30, 36) @(0, 6) @(0, 3.5, 7) 'D2' 20.0
    $null = $m.Tower.SetActiveTower('T1')
    # ---- save: the model and its text file
    $edb = Join-Path $OutDir 'TwoTowers.edb'
    Rc 'File.Save edb' $m.File.Save($edb) | Out-Null
    $e2k = Join-Path $OutDir 'TwoTowers.e2k'
    Rc 'File.Save e2k' $m.File.Save($e2k) | Out-Null
    $et = Join-Path $OutDir 'TwoTowers.$et'
    if (-not (Test-Path -LiteralPath $e2k) -and (Test-Path -LiteralPath $et)) { Copy-Item -LiteralPath $et -Destination $e2k -Force }
    Say ("e2k written: " + (Test-Path -LiteralPath $e2k))
    $null = $m.File.Save($edb)  # (back to the .edb for the analysis)
    # ---- analysis and results tables
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
            $file = Join-Path $OutDir ("TwoTowers." + ($key -replace '[^A-Za-z0-9]+', '_') + ".csv")
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
# every model table, as the import reads it
$conv = Join-Path $Studio 'structstudio\io\csi_convert.ps1'
$edb = Join-Path $OutDir 'TwoTowers.edb'
if (Test-Path -LiteralPath $edb) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $conv -OutDir (Join-Path $OutDir 'converted') $edb 2>&1 |
        ForEach-Object { Say "  convert: $_" }
}
Say "$(Get-Date -Format s) done"
exit $exit
