# Runs the ETABS verification examples in ETABS 21 through the official API and
# dumps the analysis results (joint displacements, modal periods, base reactions,
# frame end forces) to CSV files so StructStudio can be checked against the
# program itself, not only against the (older) verification manual.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File "F:\REA ETABS\studio\scripts\Export-VerificationResults.ps1"
#
# Needs an installed, licensed ETABS 21.  Nothing outside F:\REA ETABS is written.
# Output: F:\REA ETABS\studio\tests\verification\data\etabs\results\<Example>.<table>.csv
$ErrorActionPreference = 'Stop'
$EtabsDir = 'C:\Program Files\Computers and Structures\ETABS 21'
$Source   = 'F:\REA ETABS\reference\etabs21\Verification\Analysis Examples'
$Work     = 'F:\REA ETABS\studio\tests\verification\data\etabs\run'
$Dest     = 'F:\REA ETABS\studio\tests\verification\data\etabs\results'
$Log      = Join-Path $Dest 'results.log'

New-Item -ItemType Directory -Force -Path $Work | Out-Null
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
"$(Get-Date -Format s) start" | Set-Content -Path $Log

Add-Type -Path (Join-Path $EtabsDir 'ETABSv1.dll')
$helper = New-Object ETABSv1.Helper
$app = $helper.CreateObject((Join-Path $EtabsDir 'ETABS.exe'))
if ($null -eq $app) { throw 'Could not start an ETABS instance through the API' }
$rc = $app.ApplicationStart()
if ($rc -ne 0) { throw "ApplicationStart returned $rc" }
$sap = $app.SapModel

function Csv-Line { param([object[]]$cells) ($cells | ForEach-Object { '"' + ([string]$_).Replace('"', '""') + '"' }) -join ',' }

function Get-CaseNames {
    # cLoadCases.GetNameList(ref int, ref string[], eLoadCaseType CaseType = 0): Windows PowerShell
    # cannot omit the optional argument, so pass 0 (= all types) explicitly; fall back to reflection.
    $n = 0; $names = [string[]]@()
    $all = [Enum]::ToObject([ETABSv1.eLoadCaseType], 0)
    try {
        $null = $sap.LoadCases.GetNameList([ref]$n, [ref]$names, $all)
    }
    catch {
        $m = [ETABSv1.cLoadCases].GetMethod('GetNameList')
        $argv = [object[]]@([int]0, [string[]]@(), $all)
        $null = $m.Invoke($sap.LoadCases, $argv)
        $names = [string[]]$argv[1]
    }
    return ,$names
}

function Get-ComboNames {
    $n = 0; $names = [string[]]@()
    $null = $sap.RespCombo.GetNameList([ref]$n, [ref]$names)
    return ,$names
}

function Write-Table {
    param([string]$Name, [string]$Table, [scriptblock]$Body)
    try {
        $lines = & $Body
        [IO.File]::WriteAllLines((Join-Path $Dest "$Name.$Table.csv"), [string[]]$lines)
        "$Name : $Table rows=$($lines.Count - 1)" | Add-Content $Log
    }
    catch { "$Name : $Table failed: $($_.Exception.Message)" | Add-Content $Log }
}

function Export-Model {
    param([string]$Name)
    $res = $sap.Results
    $cases = Get-CaseNames
    $combos = Get-ComboNames
    "$Name : cases=$($cases -join ',') combos=$($combos -join ',')" | Add-Content $Log
    $null = $res.Setup.DeselectAllCasesAndCombosForOutput()
    foreach ($c in $cases) { $null = $res.Setup.SetCaseSelectedForOutput($c, $true) }
    foreach ($c in $combos) { $null = $res.Setup.SetComboSelectedForOutput($c, $true) }

    Write-Table $Name 'joint_displacements' {
        $num = 0; $obj = [string[]]@(); $elm = [string[]]@(); $lc = [string[]]@(); $st = [string[]]@(); $sn = [double[]]@()
        $u1 = [double[]]@(); $u2 = [double[]]@(); $u3 = [double[]]@(); $r1 = [double[]]@(); $r2 = [double[]]@(); $r3 = [double[]]@()
        $null = $res.JointDispl('All', [ETABSv1.eItemTypeElm]::GroupElm, [ref]$num, [ref]$obj, [ref]$elm, [ref]$lc, [ref]$st, [ref]$sn, [ref]$u1, [ref]$u2, [ref]$u3, [ref]$r1, [ref]$r2, [ref]$r3)
        Csv-Line @('UniqueName', 'Element', 'Story', 'Label', 'Case', 'StepType', 'StepNum', 'Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz')
        for ($i = 0; $i -lt $num; $i++) {
            $story = ''; $label = ''
            $null = $sap.PointObj.GetLabelFromName($obj[$i], [ref]$label, [ref]$story)
            Csv-Line @($obj[$i], $elm[$i], $story, $label, $lc[$i], $st[$i], $sn[$i], $u1[$i], $u2[$i], $u3[$i], $r1[$i], $r2[$i], $r3[$i])
        }
    }

    Write-Table $Name 'joint_reactions' {
        $num = 0; $obj = [string[]]@(); $elm = [string[]]@(); $lc = [string[]]@(); $st = [string[]]@(); $sn = [double[]]@()
        $f1 = [double[]]@(); $f2 = [double[]]@(); $f3 = [double[]]@(); $m1 = [double[]]@(); $m2 = [double[]]@(); $m3 = [double[]]@()
        $null = $res.JointReact('All', [ETABSv1.eItemTypeElm]::GroupElm, [ref]$num, [ref]$obj, [ref]$elm, [ref]$lc, [ref]$st, [ref]$sn, [ref]$f1, [ref]$f2, [ref]$f3, [ref]$m1, [ref]$m2, [ref]$m3)
        Csv-Line @('UniqueName', 'Story', 'Label', 'Case', 'StepType', 'StepNum', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3')
        for ($i = 0; $i -lt $num; $i++) {
            $story = ''; $label = ''
            $null = $sap.PointObj.GetLabelFromName($obj[$i], [ref]$label, [ref]$story)
            Csv-Line @($obj[$i], $story, $label, $lc[$i], $st[$i], $sn[$i], $f1[$i], $f2[$i], $f3[$i], $m1[$i], $m2[$i], $m3[$i])
        }
    }

    Write-Table $Name 'modal_periods' {
        $num = 0; $lc = [string[]]@(); $st = [string[]]@(); $sn = [double[]]@()
        $period = [double[]]@(); $freq = [double[]]@(); $circ = [double[]]@(); $eig = [double[]]@()
        $null = $res.ModalPeriod([ref]$num, [ref]$lc, [ref]$st, [ref]$sn, [ref]$period, [ref]$freq, [ref]$circ, [ref]$eig)
        Csv-Line @('Case', 'StepType', 'Mode', 'Period', 'Frequency', 'CircFreq', 'Eigenvalue')
        for ($i = 0; $i -lt $num; $i++) { Csv-Line @($lc[$i], $st[$i], $sn[$i], $period[$i], $freq[$i], $circ[$i], $eig[$i]) }
    }

    Write-Table $Name 'modal_mass_ratios' {
        $num = 0; $lc = [string[]]@(); $st = [string[]]@(); $sn = [double[]]@(); $period = [double[]]@()
        $ux = [double[]]@(); $uy = [double[]]@(); $uz = [double[]]@(); $sux = [double[]]@(); $suy = [double[]]@(); $suz = [double[]]@()
        $rx = [double[]]@(); $ry = [double[]]@(); $rz = [double[]]@(); $srx = [double[]]@(); $sry = [double[]]@(); $srz = [double[]]@()
        $null = $res.ModalParticipatingMassRatios([ref]$num, [ref]$lc, [ref]$st, [ref]$sn, [ref]$period, [ref]$ux, [ref]$uy, [ref]$uz, [ref]$sux, [ref]$suy, [ref]$suz, [ref]$rx, [ref]$ry, [ref]$rz, [ref]$srx, [ref]$sry, [ref]$srz)
        Csv-Line @('Case', 'StepType', 'Mode', 'Period', 'UX', 'UY', 'UZ', 'SumUX', 'SumUY', 'SumUZ', 'RX', 'RY', 'RZ', 'SumRX', 'SumRY', 'SumRZ')
        for ($i = 0; $i -lt $num; $i++) { Csv-Line @($lc[$i], $st[$i], $sn[$i], $period[$i], $ux[$i], $uy[$i], $uz[$i], $sux[$i], $suy[$i], $suz[$i], $rx[$i], $ry[$i], $rz[$i], $srx[$i], $sry[$i], $srz[$i]) }
    }

    Write-Table $Name 'base_reactions' {
        $num = 0; $lc = [string[]]@(); $st = [string[]]@(); $sn = [double[]]@()
        $fx = [double[]]@(); $fy = [double[]]@(); $fz = [double[]]@(); $mx = [double[]]@(); $my = [double[]]@(); $mz = [double[]]@()
        $gx = 0.0; $gy = 0.0; $gz = 0.0
        $null = $res.BaseReact([ref]$num, [ref]$lc, [ref]$st, [ref]$sn, [ref]$fx, [ref]$fy, [ref]$fz, [ref]$mx, [ref]$my, [ref]$mz, [ref]$gx, [ref]$gy, [ref]$gz)
        Csv-Line @('Case', 'StepType', 'StepNum', 'FX', 'FY', 'FZ', 'MX', 'MY', 'MZ', 'gx', 'gy', 'gz')
        for ($i = 0; $i -lt $num; $i++) { Csv-Line @($lc[$i], $st[$i], $sn[$i], $fx[$i], $fy[$i], $fz[$i], $mx[$i], $my[$i], $mz[$i], $gx, $gy, $gz) }
    }

    Write-Table $Name 'frame_forces' {
        $num = 0; $obj = [string[]]@(); $osta = [double[]]@(); $elm = [string[]]@(); $esta = [double[]]@(); $lc = [string[]]@(); $st = [string[]]@(); $sn = [double[]]@()
        $p = [double[]]@(); $v2 = [double[]]@(); $v3 = [double[]]@(); $t = [double[]]@(); $m2 = [double[]]@(); $m3 = [double[]]@()
        $null = $res.FrameForce('All', [ETABSv1.eItemTypeElm]::GroupElm, [ref]$num, [ref]$obj, [ref]$osta, [ref]$elm, [ref]$esta, [ref]$lc, [ref]$st, [ref]$sn, [ref]$p, [ref]$v2, [ref]$v3, [ref]$t, [ref]$m2, [ref]$m3)
        Csv-Line @('UniqueName', 'Story', 'Label', 'ObjSta', 'Element', 'ElmSta', 'Case', 'StepType', 'StepNum', 'P', 'V2', 'V3', 'T', 'M2', 'M3')
        for ($i = 0; $i -lt $num; $i++) {
            $story = ''; $label = ''
            $null = $sap.FrameObj.GetLabelFromName($obj[$i], [ref]$label, [ref]$story)
            Csv-Line @($obj[$i], $story, $label, $osta[$i], $elm[$i], $esta[$i], $lc[$i], $st[$i], $sn[$i], $p[$i], $v2[$i], $v3[$i], $t[$i], $m2[$i], $m3[$i])
        }
    }

    Write-Table $Name 'story_drifts' {
        $num = 0; $story = [string[]]@(); $lc = [string[]]@(); $st = [string[]]@(); $sn = [double[]]@(); $dir = [string[]]@()
        $drift = [double[]]@(); $label = [string[]]@(); $x = [double[]]@(); $y = [double[]]@(); $z = [double[]]@()
        $null = $res.StoryDrifts([ref]$num, [ref]$story, [ref]$lc, [ref]$st, [ref]$sn, [ref]$dir, [ref]$drift, [ref]$label, [ref]$x, [ref]$y, [ref]$z)
        Csv-Line @('Story', 'Case', 'StepType', 'StepNum', 'Direction', 'Drift', 'Label', 'X', 'Y', 'Z')
        for ($i = 0; $i -lt $num; $i++) { Csv-Line @($story[$i], $lc[$i], $st[$i], $sn[$i], $dir[$i], $drift[$i], $label[$i], $x[$i], $y[$i], $z[$i]) }
    }

    # Database tables (as shown in ETABS Display > Show Tables); export what exists
    try {
        $dc = [string[]]$cases; $dk = [string[]]$combos
        $null = $sap.DatabaseTables.SetLoadCasesSelectedForDisplay([ref]$dc)
        if ($dk.Count) { $null = $sap.DatabaseTables.SetLoadCombinationsSelectedForDisplay([ref]$dk) }
    }
    catch { "$Name : table case selection failed: $($_.Exception.Message)" | Add-Content $Log }
    foreach ($table in @('Diaphragm Center Of Mass Displacements', 'Centers Of Mass And Rigidity', 'Story Forces',
                         'Pier Forces', 'Modal Participating Mass Ratios')) {
        Write-Table $Name ('table_' + ($table -replace '[^A-Za-z0-9]+', '_').ToLower()) {
            $ver = 0; $request = [string[]]@(); $fields = [string[]]@(); $nrec = 0; $data = [string[]]@()
            $rc = $sap.DatabaseTables.GetTableForDisplayArray($table, [ref]$request, 'All', [ref]$ver, [ref]$fields, [ref]$nrec, [ref]$data)
            if ($rc -ne 0) { throw "GetTableForDisplayArray rc=$rc" }
            $nf = $fields.Count
            Csv-Line $fields
            for ($r = 0; $r -lt $nrec; $r++) { Csv-Line $data[($r * $nf)..($r * $nf + $nf - 1)] }
        }
    }
}

try {
    foreach ($f in Get-ChildItem -Path $Source -Filter 'Example *.edb' | Sort-Object Name) {
        $name = $f.BaseName.Replace(' ', '')
        $copy = Join-Path $Work $f.Name
        Copy-Item -LiteralPath $f.FullName -Destination $copy -Force
        $rc = $sap.File.OpenFile($copy)
        if ($rc -ne 0) { "$name : OpenFile rc=$rc" | Add-Content $Log; continue }
        $null = $sap.SetPresentUnits([ETABSv1.eUnits]::kip_in_F)
        $rc = $sap.Analyze.RunAnalysis()
        "$name : RunAnalysis rc=$rc" | Add-Content $Log
        if ($rc -ne 0) { continue }
        try { Export-Model -Name $name }
        catch { "$name : export failed: $($_.Exception.Message)" | Add-Content $Log }
        $null = $sap.SetModelIsLocked($false)
    }
}
finally {
    $app.ApplicationExit($false) | Out-Null
    "$(Get-Date -Format s) done" | Add-Content $Log
}
Get-Content $Log
