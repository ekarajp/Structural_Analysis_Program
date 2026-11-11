# Runs the converted SAP2000 / SAFE sample models (tests\verification\data\sap2000|safe\*.sdb|*.FDB)
# in the installed program and dumps the analysis results to CSV (kip-in), like
# Export-VerificationResults.ps1 does for ETABS, so Structural Studio can be checked against
# SAP2000 and SAFE themselves.  Nothing outside F:\REA ETABS is written.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File "F:\REA ETABS\studio\scripts\Export-CsiResults.ps1"
#
$ErrorActionPreference = 'Continue'
$Studio = Split-Path -Parent $PSScriptRoot
$Data   = Join-Path $Studio 'tests\verification\data'
. (Join-Path $Studio 'structstudio\io\csi_api.ps1')
$Log = Join-Path $Data 'csi_results.log'
"$(Get-Date -Format s) start" | Set-Content -Path $Log
function Say([string]$text) { Write-Host $text; Add-Content -Path $Log -Value $text }
function Csv-Line { param([object[]]$cells) ($cells | ForEach-Object { '"' + ([string]$_).Replace('"', '""') + '"' }) -join ',' }

function Get-Names($ns, $obj) {
    $n = 0; [string[]]$names = @()
    try { $null = $obj.GetNameList([ref]$n, [ref]$names); return ,$names } catch { }
    $t = ("$ns.eLoadCaseType" -as [type])
    if ($t) { try { $null = $obj.GetNameList([ref]$n, [ref]$names, [Enum]::ToObject($t, 0)); return ,$names } catch { } }
    return ,$names
}

function Write-Csv($dest, $name, $table, [scriptblock]$body) {
    try {
        $lines = & $body
        [IO.File]::WriteAllLines((Join-Path $dest "$name.$table.csv"), [string[]]$lines)
        Say "  $name : $table rows=$($lines.Count - 1)"
    } catch { Say "  $name : $table failed: $($_.Exception.Message)" }
}

function Export-Results($sap, $ns, $dest, $name) {
    $res = $sap.Results
    $grp = [Enum]::ToObject(("$ns.eItemTypeElm" -as [type]), 2)   # GroupElm
    $cases = Get-Names $ns $sap.LoadCases
    $combos = Get-Names $ns $sap.RespCombo
    $null = $res.Setup.DeselectAllCasesAndCombosForOutput()
    foreach ($c in $cases) { $null = $res.Setup.SetCaseSelectedForOutput($c, $true) }
    foreach ($c in $combos) { $null = $res.Setup.SetComboSelectedForOutput($c, $true) }
    Write-Csv $dest $name 'joint_displacements' {
        $num = 0; [string[]]$obj = @(); [string[]]$elm = @(); [string[]]$lc = @(); [string[]]$st = @(); [double[]]$sn = @()
        [double[]]$u1 = @(); [double[]]$u2 = @(); [double[]]$u3 = @(); [double[]]$r1 = @(); [double[]]$r2 = @(); [double[]]$r3 = @()
        $null = $res.JointDispl('All', $grp, [ref]$num, [ref]$obj, [ref]$elm, [ref]$lc, [ref]$st, [ref]$sn, [ref]$u1, [ref]$u2, [ref]$u3, [ref]$r1, [ref]$r2, [ref]$r3)
        Csv-Line @('UniqueName', 'Element', 'Case', 'StepType', 'StepNum', 'Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz')
        for ($i = 0; $i -lt $num; $i++) { Csv-Line @($obj[$i], $elm[$i], $lc[$i], $st[$i], $sn[$i], $u1[$i], $u2[$i], $u3[$i], $r1[$i], $r2[$i], $r3[$i]) }
    }
    Write-Csv $dest $name 'joint_reactions' {
        $num = 0; [string[]]$obj = @(); [string[]]$elm = @(); [string[]]$lc = @(); [string[]]$st = @(); [double[]]$sn = @()
        [double[]]$f1 = @(); [double[]]$f2 = @(); [double[]]$f3 = @(); [double[]]$m1 = @(); [double[]]$m2 = @(); [double[]]$m3 = @()
        $null = $res.JointReact('All', $grp, [ref]$num, [ref]$obj, [ref]$elm, [ref]$lc, [ref]$st, [ref]$sn, [ref]$f1, [ref]$f2, [ref]$f3, [ref]$m1, [ref]$m2, [ref]$m3)
        Csv-Line @('UniqueName', 'Case', 'StepType', 'StepNum', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3')
        for ($i = 0; $i -lt $num; $i++) { Csv-Line @($obj[$i], $lc[$i], $st[$i], $sn[$i], $f1[$i], $f2[$i], $f3[$i], $m1[$i], $m2[$i], $m3[$i]) }
    }
    Write-Csv $dest $name 'modal_periods' {
        $num = 0; [string[]]$lc = @(); [string[]]$st = @(); [double[]]$sn = @()
        [double[]]$period = @(); [double[]]$freq = @(); [double[]]$circ = @(); [double[]]$eig = @()
        $null = $res.ModalPeriod([ref]$num, [ref]$lc, [ref]$st, [ref]$sn, [ref]$period, [ref]$freq, [ref]$circ, [ref]$eig)
        Csv-Line @('Case', 'StepType', 'Mode', 'Period', 'Frequency', 'CircFreq', 'Eigenvalue')
        for ($i = 0; $i -lt $num; $i++) { Csv-Line @($lc[$i], $st[$i], $sn[$i], $period[$i], $freq[$i], $circ[$i], $eig[$i]) }
    }
    Write-Csv $dest $name 'base_reactions' {
        $num = 0; [string[]]$lc = @(); [string[]]$st = @(); [double[]]$sn = @()
        [double[]]$fx = @(); [double[]]$fy = @(); [double[]]$fz = @(); [double[]]$mx = @(); [double[]]$my = @(); [double[]]$mz = @()
        $gx = 0.0; $gy = 0.0; $gz = 0.0
        $null = $res.BaseReact([ref]$num, [ref]$lc, [ref]$st, [ref]$sn, [ref]$fx, [ref]$fy, [ref]$fz, [ref]$mx, [ref]$my, [ref]$mz, [ref]$gx, [ref]$gy, [ref]$gz)
        Csv-Line @('Case', 'StepType', 'StepNum', 'FX', 'FY', 'FZ', 'MX', 'MY', 'MZ')
        for ($i = 0; $i -lt $num; $i++) { Csv-Line @($lc[$i], $st[$i], $sn[$i], $fx[$i], $fy[$i], $fz[$i], $mx[$i], $my[$i], $mz[$i]) }
    }
    Write-Csv $dest $name 'frame_forces' {
        $num = 0; [string[]]$obj = @(); [double[]]$osta = @(); [string[]]$elm = @(); [double[]]$esta = @(); [string[]]$lc = @(); [string[]]$st = @(); [double[]]$sn = @()
        [double[]]$p = @(); [double[]]$v2 = @(); [double[]]$v3 = @(); [double[]]$t = @(); [double[]]$m2 = @(); [double[]]$m3 = @()
        $null = $res.FrameForce('All', $grp, [ref]$num, [ref]$obj, [ref]$osta, [ref]$elm, [ref]$esta, [ref]$lc, [ref]$st, [ref]$sn, [ref]$p, [ref]$v2, [ref]$v3, [ref]$t, [ref]$m2, [ref]$m3)
        Csv-Line @('UniqueName', 'ObjSta', 'Element', 'ElmSta', 'Case', 'StepType', 'StepNum', 'P', 'V2', 'V3', 'T', 'M2', 'M3')
        for ($i = 0; $i -lt $num; $i++) { Csv-Line @($obj[$i], $osta[$i], $elm[$i], $esta[$i], $lc[$i], $st[$i], $sn[$i], $p[$i], $v2[$i], $v3[$i], $t[$i], $m2[$i], $m3[$i]) }
    }
}

foreach ($p in @(@{ Key = '.sdb'; Folder = 'sap2000' }, @{ Key = '.fdb'; Folder = 'safe' })) {
    $folder = Join-Path $Data $p.Folder
    $models = @(Get-ChildItem -Path $folder -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -ieq $p.Key } | Sort-Object Name)
    if ($models.Count -eq 0) { Say "$($p.Folder): no models (run Export-CsiSamples.cmd first)"; continue }
    $info = $Programs[$p.Key]
    $found = Find-Program $info
    if ($null -eq $found) { Say "$($p.Folder): $($info.Name) not installed"; continue }
    $dest = Join-Path $folder 'results'
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Say "$($p.Folder): $($models.Count) models, starting $($found.Label) ..."
    try { $app = Start-CsiProgram $found $info } catch { Say "$($p.Folder): FAIL $($_.Exception.Message)"; continue }
    $ns = [System.IO.Path]::GetFileNameWithoutExtension($found.Dll)
    $kipIn = [Enum]::ToObject(("$ns.eUnits" -as [type]), 3)
    try {
        foreach ($f in $models) {
            $name = $f.BaseName.Replace(' ', '')
            Say "Running $($f.Name) ..."
            $sap = $app.SapModel
            $rc = $sap.File.OpenFile($f.FullName)
            if ($rc -ne 0) { Say "  OpenFile rc=$rc"; continue }
            $null = $sap.SetPresentUnits($kipIn)
            $rc = $sap.Analyze.RunAnalysis()
            if ($rc -ne 0) { Say "  RunAnalysis rc=$rc"; continue }
            try { Export-Results $sap $ns $dest $name } catch { Say "  export failed: $($_.Exception.Message)" }
        }
    } finally {
        try { $app.ApplicationExit($false) | Out-Null } catch { }
    }
}
Say "$(Get-Date -Format s) done"
