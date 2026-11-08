# Builds the comparison models (<Folder>\*.spec.json, written by csi_compare.py build) in
# the installed ETABS and SAFE through their official API, runs them and exports the
# joint displacements / reactions to <Folder>\etabs|safe\<model>.*.csv (kN, m).
# Every API call is logged in <Folder>\etabs|safe\<model>.calls.log before it is made (so a call that
# hangs is the last line) and with its return code (not <model>.log: ETABS writes its own <model>.LOG).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Run-CsiCompare.ps1 -Folder <folder>
#
# Each model runs in its own PowerShell process with a fresh ETABS / SAFE (no state left
# over from the model before) and a time limit: a model that is still busy after
# -TimeoutMinutes is stopped, a screenshot of the screen goes to <model>.hang.png (to see
# any dialog the program was waiting on) and the next model / program goes on.
param([Parameter(Mandatory = $true)][string]$Folder, [string[]]$Only = @(), [string]$Model = '',
      [int]$TimeoutMinutes = 10)
$ErrorActionPreference = 'Continue'
$Studio = Split-Path -Parent $PSScriptRoot
. (Join-Path $Studio 'structstudio\io\csi_api.ps1')
$Hidden = $false

function Csv-Line { param([object[]]$cells) ($cells | ForEach-Object { '"' + ([string]$_).Replace('"', '""') + '"' }) -join ',' }

$script:LogFile = $null
function Say([string]$text) { Write-Host $text; if ($script:LogFile) { Add-Content -Path $script:LogFile -Value $text } }
$script:Failed = 0
function Rc([string]$what, $rc) {
    # CSI API calls return 0 on success
    if ($rc -is [array]) { $rc = $rc[0] }
    if ($null -ne $rc -and "$rc" -ne '0') { $script:Failed++; Say "  FAIL rc=$rc : $what" }
    return $rc
}
function Try-Call([string]$what, [scriptblock]$block) {
    if ($script:LogFile) { Add-Content -Path $script:LogFile -Value "  > $what" }
    try { return (Rc $what (& $block)) } catch { $script:Failed++; Say "  ERROR $what : $($_.Exception.Message)"; return -1 }
}
function En([string]$ns, [string]$type, [string]$name) {
    $t = ("$ns.$type" -as [type])
    if ($null -eq $t) { throw "enum $ns.$type not found" }
    return [Enum]::Parse($t, $name)
}

function Build-Model($sap, [string]$ns, $spec, [string]$program) {
    $obj = En $ns 'eItemType' 'Objects'
    $null = Try-Call 'InitializeNewModel kN_m_C' { $sap.InitializeNewModel((En $ns 'eUnits' 'kN_m_C')) }
    $script:Dz = 0.0
    if ($program -eq 'SAFE') {
        # SAFE: one floor at the datum Z = 0, the stories below / above are heights
        # only - shift the whole model so that the slab is at Z = 0
        $script:Dz = [double]$spec.slab_z
        $zs = @($spec.points | ForEach-Object { [double]$_.z })
        $below = $script:Dz - ($zs | Measure-Object -Minimum).Minimum
        $above = ($zs | Measure-Object -Maximum).Maximum - $script:Dz
        if ($below -le 0) { $below = 3.0 }
        if ($above -le 0) { $above = 3.0 }
        $rc = Try-Call "NewGridOnly SAFE floor (below $below / above $above), model shifted by -$($script:Dz)" { $sap.File.NewGridOnly(1, $above, $below, 2, 2, 10.0, 10.0) }
        if ($rc -ne 0) { $null = Try-Call 'NewBlank' { $sap.File.NewBlank() } }
        Build-Objects $sap $ns $spec $program $obj
        return
    }
    # stories: the levels of the model (Base + storeys)
    $z = @($spec.stories)
    $n = $z.Count - 1
    $h1 = [double]$z[1] - [double]$z[0]
    $ht = $h1
    if ($n -ge 2) { $ht = [double]$z[2] - [double]$z[1] }
    $uniform = $true
    for ($k = 2; $k -le $n; $k++) { if ([math]::Abs(([double]$z[$k] - [double]$z[$k - 1]) - $ht) -gt 1e-6) { $uniform = $false } }
    $rc = Try-Call "NewGridOnly $n storeys ($h1 / $ht)" { $sap.File.NewGridOnly($n, $ht, $h1, 2, 2, 10.0, 10.0) }
    if ($rc -ne 0) { $null = Try-Call 'NewBlank' { $sap.File.NewBlank() } }
    if (-not $uniform) {
        [string[]]$names = @(); [double[]]$hs = @(); [bool[]]$master = @(); [string[]]$sim = @(); [bool[]]$spl = @(); [double[]]$sph = @(); [int[]]$col = @()
        for ($k = 1; $k -le $n; $k++) { $names += "Story$k"; $hs += ([double]$z[$k] - [double]$z[$k - 1]); $master += $true; $sim += ''; $spl += $false; $sph += 0.0; $col += 0 }
        $null = Try-Call 'SetStories_2' { $sap.Story.SetStories_2([double]$z[0], $n, [ref]$names, [ref]$hs, [ref]$master, [ref]$sim, [ref]$spl, [ref]$sph, [ref]$col) }
    }
    Build-Objects $sap $ns $spec $program $obj
}

function Build-Objects($sap, [string]$ns, $spec, [string]$program, $obj) {
    foreach ($m in $spec.materials) {
        $null = Try-Call "material $($m.name)" { $sap.PropMaterial.SetMaterial($m.name, (En $ns 'eMatType' 'Concrete'), -1, '', '') }
        $null = Try-Call "material $($m.name) E nu" { $sap.PropMaterial.SetMPIsotropic($m.name, [double]$m.E, [double]$m.nu, 1e-5, 0.0) }
        $null = Try-Call "material $($m.name) weight 0" { $sap.PropMaterial.SetWeightAndMass($m.name, 1, 0.0, 0.0) }
    }
    foreach ($s in $spec.frame_sections) {
        $null = Try-Call "frame section $($s.name)" { $sap.PropFrame.SetRectangle($s.name, $s.material, [double]$s.t3, [double]$s.t2, -1, '', '') }
    }
    foreach ($s in $spec.shell_sections) {
        $shell = En $ns 'eShellType' $s.shell
        if ($s.kind -eq 'Wall') {
            $null = Try-Call "wall section $($s.name)" { $sap.PropArea.SetWall($s.name, (En $ns 'eWallPropType' 'Specified'), $shell, $s.material, [double]$s.t, -1, '', '') }
        } else {
            $null = Try-Call "slab section $($s.name)" { $sap.PropArea.SetSlab($s.name, (En $ns 'eSlabType' 'Slab'), $shell, $s.material, [double]$s.t, -1, '', '') }
        }
        if ($s.modifiers) {
            [double[]]$mods = @($s.modifiers | ForEach-Object { [double]$_ })
            $null = Try-Call "stiffness modifiers $($s.name) ($($mods -join ' '))" { $sap.PropArea.SetModifiers($s.name, [ref]$mods) }
        }
    }
    $named = @{}
    foreach ($p in $spec.points) {
        $nm = ''
        $null = Try-Call "point $($p.name)" { $sap.PointObj.AddCartesian([double]$p.x, [double]$p.y, ([double]$p.z - $script:Dz), [ref]$nm, $p.name, 'Global', $false, 0) }
        if ($nm -ne $p.name) { Say "  note: point $($p.name) was named '$nm' by $program" }
        $named[$p.name] = $nm
        if ($null -ne $p.restraint) {
            [bool[]]$r = @($p.restraint | ForEach-Object { [bool]$_ })
            $null = Try-Call "restraint $($p.name)" { $sap.PointObj.SetRestraint($nm, [ref]$r, $obj) }
        }
    }
    foreach ($f in $spec.frames) {
        $nm = ''
        $null = Try-Call "frame $($f.name)" { $sap.FrameObj.AddByPoint($named[$f.i], $named[$f.j], [ref]$nm, $f.section, $f.name) }
        if ($f.section -ne 'None') {
            [double[]]$o1 = @(0.0, 0.0, 0.0); [double[]]$o2 = @(0.0, 0.0, 0.0)
            $null = Try-Call "insertion point $($f.name)" { $sap.FrameObj.SetInsertionPoint($nm, [int]$f.cardinal, $false, $true, [ref]$o1, [ref]$o2, 'Local', $obj) }
            if ([double]$f.angle -ne 0) { $null = Try-Call "local axes $($f.name)" { $sap.FrameObj.SetLocalAxes($nm, [double]$f.angle, $obj) } }
        }
    }
    $firstSlab = ($spec.shell_sections | Select-Object -First 1).name
    foreach ($a in $spec.areas) {
        [string[]]$pts = @($a.points | ForEach-Object { $named[$_] })
        $nm = ''
        $prop = $a.section
        if ($a.opening) { $prop = $firstSlab }
        $null = Try-Call "area $($a.name) ($($a.type))" { $sap.AreaObj.AddByPoint($pts.Count, [ref]$pts, [ref]$nm, $prop, $a.name) }
        if ($a.opening) { $null = Try-Call "opening $($a.name)" { $sap.AreaObj.SetOpening($nm, $true, $obj) } }
    }
    # a new ETABS / SAFE model already has Dead and Live: add the pattern only if missing
    $pn = 0; [string[]]$pats = @()
    $null = $sap.LoadPatterns.GetNameList([ref]$pn, [ref]$pats)
    if (@($pats) -notcontains $spec.case) {
        $null = Try-Call "load pattern $($spec.case)" { $sap.LoadPatterns.Add($spec.case, (En $ns 'eLoadPatternType' 'Live'), 0.0, $true) }
    }
    foreach ($l in $spec.point_loads) {
        [double[]]$v = @($l.F | ForEach-Object { [double]$_ })
        $null = Try-Call "point load $($l.point)" { $sap.PointObj.SetLoadForce($named[$l.point], $l.pattern, [ref]$v, $false, 'Global', $obj) }
    }
    foreach ($l in $spec.frame_loads) {
        $null = Try-Call "line load $($l.frame)" { $sap.FrameObj.SetLoadDistributed($l.frame, $l.pattern, 1, 10, [double]$l.a, [double]$l.b, [double]$l.wa, [double]$l.wb, 'Global', $true, $true, $obj) }
    }
    foreach ($l in $spec.area_loads) {
        $null = Try-Call "area load $($l.area)" { $sap.AreaObj.SetLoadUniform($l.area, $l.pattern, [double]$l.q, 10, $true, 'Global', $obj) }
    }
    # what this program offers for meshing (logged so the mesh options can be matched later)
    foreach ($o in @(@{ N = 'AreaObj'; V = $sap.AreaObj }, @{ N = 'Analyze'; V = $sap.Analyze })) {
        try {
            $ms = $o.V.GetType().GetMethods() | Where-Object { $_.Name -match 'Mesh' } | ForEach-Object { $_.Name } | Select-Object -Unique
            if ($ms) { Say "  $($o.N) mesh methods: $($ms -join ', ')" }
        } catch { }
    }
}

function Export-Results($sap, [string]$ns, [string]$dest, [string]$name, $spec) {
    $res = $sap.Results
    $grp = En $ns 'eItemTypeElm' 'GroupElm'
    $null = $res.Setup.DeselectAllCasesAndCombosForOutput()
    $null = Try-Call 'select case' { $res.Setup.SetCaseSelectedForOutput($spec.case, $true) }
    $num = 0; [string[]]$o = @(); [string[]]$e = @(); [string[]]$lc = @(); [string[]]$st = @(); [double[]]$sn = @()
    [double[]]$u1 = @(); [double[]]$u2 = @(); [double[]]$u3 = @(); [double[]]$r1 = @(); [double[]]$r2 = @(); [double[]]$r3 = @()
    $null = Try-Call 'JointDispl' { $res.JointDispl('All', $grp, [ref]$num, [ref]$o, [ref]$e, [ref]$lc, [ref]$st, [ref]$sn, [ref]$u1, [ref]$u2, [ref]$u3, [ref]$r1, [ref]$r2, [ref]$r3) }
    $lines = @(Csv-Line @('UniqueName', 'Element', 'Case', 'StepType', 'StepNum', 'Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz'))
    for ($i = 0; $i -lt $num; $i++) { $lines += Csv-Line @($o[$i], $e[$i], $lc[$i], $st[$i], $sn[$i], $u1[$i], $u2[$i], $u3[$i], $r1[$i], $r2[$i], $r3[$i]) }
    [IO.File]::WriteAllLines((Join-Path $dest "$name.joint_displacements.csv"), [string[]]$lines)
    Say "  joint displacements: $num rows"
    $num = 0; [string[]]$o = @(); [string[]]$e = @(); [string[]]$lc = @(); [string[]]$st = @(); [double[]]$sn = @()
    [double[]]$f1 = @(); [double[]]$f2 = @(); [double[]]$f3 = @(); [double[]]$m1 = @(); [double[]]$m2 = @(); [double[]]$m3 = @()
    $null = Try-Call 'JointReact' { $res.JointReact('All', $grp, [ref]$num, [ref]$o, [ref]$e, [ref]$lc, [ref]$st, [ref]$sn, [ref]$f1, [ref]$f2, [ref]$f3, [ref]$m1, [ref]$m2, [ref]$m3) }
    $lines = @(Csv-Line @('UniqueName', 'Case', 'StepType', 'StepNum', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3'))
    for ($i = 0; $i -lt $num; $i++) { $lines += Csv-Line @($o[$i], $lc[$i], $st[$i], $sn[$i], $f1[$i], $f2[$i], $f3[$i], $m1[$i], $m2[$i], $m3[$i]) }
    [IO.File]::WriteAllLines((Join-Path $dest "$name.joint_reactions.csv"), [string[]]$lines)
    Say "  joint reactions: $num rows"
    $num = 0; [string[]]$lc = @(); [string[]]$st = @(); [double[]]$sn = @()
    [double[]]$fx = @(); [double[]]$fy = @(); [double[]]$fz = @(); [double[]]$mx = @(); [double[]]$my = @(); [double[]]$mz = @()
    $gx = 0.0; $gy = 0.0; $gz = 0.0
    $null = Try-Call 'BaseReact' { $res.BaseReact([ref]$num, [ref]$lc, [ref]$st, [ref]$sn, [ref]$fx, [ref]$fy, [ref]$fz, [ref]$mx, [ref]$my, [ref]$mz, [ref]$gx, [ref]$gy, [ref]$gz) }
    $lines = @(Csv-Line @('Case', 'StepType', 'StepNum', 'FX', 'FY', 'FZ', 'MX', 'MY', 'MZ'))
    for ($i = 0; $i -lt $num; $i++) { $lines += Csv-Line @($lc[$i], $st[$i], $sn[$i], $fx[$i], $fy[$i], $fz[$i], $mx[$i], $my[$i], $mz[$i]) }
    [IO.File]::WriteAllLines((Join-Path $dest "$name.base_reactions.csv"), [string[]]$lines)
    # where the program put our joints (a check that the geometry arrived as sent)
    $lines = @(Csv-Line @('UniqueName', 'X', 'Y', 'Z'))
    foreach ($p in $spec.points) {
        $x = 0.0; $y = 0.0; $zz = 0.0
        try { $null = $sap.PointObj.GetCoordCartesian($p.name, [ref]$x, [ref]$y, [ref]$zz, 'Global'); $lines += Csv-Line @($p.name, $x, $y, $zz) } catch { }
    }
    [IO.File]::WriteAllLines((Join-Path $dest "$name.points.csv"), [string[]]$lines)
}

function Save-Screen([string]$path) {
    try {
        Add-Type -AssemblyName System.Windows.Forms, System.Drawing
        $b = [System.Windows.Forms.SystemInformation]::VirtualScreen
        $bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
        $g = [System.Drawing.Graphics]::FromImage($bmp)
        $g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
        $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
        $g.Dispose(); $bmp.Dispose()
    } catch { }
}

$specs = @(Get-ChildItem -Path $Folder -Filter '*.spec.json' | Sort-Object Name)
if ($Model) { $specs = @($specs | Where-Object { $_.BaseName -eq "$Model.spec" }) }
if ($specs.Count -eq 0) { Write-Host "No *.spec.json in $Folder (run csi_compare.py build first)" -ForegroundColor Red; exit 1 }
$keys = @{ 'ETABS' = '.edb'; 'SAFE' = '.fdb' }
$ext = @{ 'ETABS' = '.EDB'; 'SAFE' = '.FDB' }
foreach ($program in @('ETABS', 'SAFE')) {
    if ($Only.Count -gt 0 -and $Only -notcontains $program) { continue }
    $todo = @($specs | Where-Object { (Get-Content $_.FullName -Raw | ConvertFrom-Json).programs -contains $program })
    if ($todo.Count -eq 0) { continue }
    $info = $Programs[$keys[$program]]
    $found = Find-Program $info
    if ($null -eq $found) { Write-Host "$program is not installed - skipped" -ForegroundColor Yellow; continue }
    $dest = Join-Path $Folder $program.ToLower()
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    if (-not $Model) {
        # one process (and one fresh program) per model, with a time limit
        $exe = [System.IO.Path]::GetFileNameWithoutExtension($found.Exe)
        foreach ($f in $todo) {
            $name = (Get-Content $f.FullName -Raw | ConvertFrom-Json).name
            $before = @(Get-Process -Name $exe -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
            $argv = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-Folder', "`"$Folder`"",
                      '-Only', $program, '-Model', $name)
            $child = Start-Process -FilePath 'powershell.exe' -ArgumentList $argv -NoNewWindow -PassThru
            if (-not $child.WaitForExit($TimeoutMinutes * 60000)) {
                $log = Join-Path $dest "$name.calls.log"
                $last = ''
                if (Test-Path $log) { $last = (Get-Content $log | Select-Object -Last 1) }
                $png = Join-Path $dest "$name.hang.png"
                Save-Screen $png
                $titles = @(Get-Process -Name $exe -ErrorAction SilentlyContinue | Where-Object { $before -notcontains $_.Id } |
                            ForEach-Object { $_.MainWindowTitle } | Where-Object { $_ })
                $msg = "TIMEOUT: $name in $program still busy after $TimeoutMinutes min; last call:$last; windows: $($titles -join ' | '); screenshot $png"
                Write-Host "  $msg" -ForegroundColor Red
                Add-Content -Path $log -Value $msg
                # the whole tree: the PowerShell child and the program it started
                if (Get-Command taskkill.exe -ErrorAction SilentlyContinue) { & taskkill.exe /PID $child.Id /T /F 2>&1 | Out-Null }
                try { Stop-Process -Id $child.Id -Force -ErrorAction SilentlyContinue } catch { }
                Get-Process -Name $exe -ErrorAction SilentlyContinue | Where-Object { $before -notcontains $_.Id } |
                    ForEach-Object { try { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue } catch { } }
            }
        }
        continue
    }
    Write-Host "Starting $($found.Label) for $($todo.Count) model(s) ..." -ForegroundColor Cyan
    try { $app = Start-CsiProgram $found $info } catch { Write-Host "$program could not start: $($_.Exception.Message)" -ForegroundColor Red; continue }
    $ns = [System.IO.Path]::GetFileNameWithoutExtension($found.Dll)
    try {
        foreach ($f in $todo) {
            $spec = Get-Content $f.FullName -Raw | ConvertFrom-Json
            $name = $spec.name
            $script:LogFile = Join-Path $dest "$name.calls.log"
            $script:Failed = 0
            "$(Get-Date -Format s) $program $($found.Label) $name" | Set-Content -Path $script:LogFile
            Say "$name in $program ..."
            try {
            $sap = $app.SapModel
            Build-Model $sap $ns $spec $program
            $file = Join-Path $dest ($name + $ext[$program])
            $null = Try-Call "save $file" { $sap.File.Save($file) }
            Say '  running the analysis ...'
            try { Say "  area objects in $program : $($sap.AreaObj.Count())" } catch { }
            $rc = Try-Call 'RunAnalysis' { $sap.Analyze.RunAnalysis() }
            $alog = Join-Path $dest ($name + '.LOG')
            if (Test-Path $alog) {
                $shells = Select-String -Path $alog -Pattern 'NUMBER OF SHELL ELEMENTS\s*=\s*(\d+)' | Select-Object -First 1
                if ($shells) { Say "  $program shell elements: $($shells.Matches[0].Groups[1].Value)" }
                else { Say "  WARNING: $program made no shell elements (see $alog)" }
                if (Select-String -Path $alog -Pattern 'UNSTABLE' -Quiet) { Say "  WARNING: $program reports the structure unstable (see $alog)" }
            }
            if ($rc -eq 0) {
                $null = Try-Call 'units kN m' { $sap.SetPresentUnits((En $ns 'eUnits' 'kN_m_C')) }
                Export-Results $sap $ns $dest $name $spec
            }
            Say "  $name done, $($script:Failed) API call(s) failed (details above / in $($script:LogFile))"
            } catch { Say "  $name stopped: $($_.Exception.Message)" }
        }
    } finally {
        try { $app.ApplicationExit($false) | Out-Null } catch { }
        $script:LogFile = $null
    }
}
