# Shared helpers for driving the installed ETABS / SAP2000 / SAFE / CSiBridge through
# their official APIs (dot-sourced by csi_convert.ps1 and Export-CsiResults.ps1).
$Programs = @{
    '.edb' = @{ Name = 'ETABS';     Dirs = 'ETABS*';     Exe = 'ETABS.exe';     Dlls = @('ETABSv1.dll', 'ETABS2016.dll', 'ETABS2015.dll', 'ETABS2013.dll'); ProgId = 'CSI.ETABS.API.ETABSObject' }
    '.sdb' = @{ Name = 'SAP2000';   Dirs = 'SAP2000*';   Exe = 'SAP2000.exe';   Dlls = @('SAP2000v1.dll', 'SAP2000v19.dll', 'SAP2000v18.dll', 'SAP2000v17.dll', 'SAP2000v16.dll', 'SAP2000v15.dll'); ProgId = 'CSI.SAP2000.API.SapObject' }
    '.fdb' = @{ Name = 'SAFE';      Dirs = 'SAFE*';      Exe = 'SAFE.exe';      Dlls = @('SAFEv1.dll'); ProgId = 'CSI.SAFE.API.ETABSObject' }
    '.bdb' = @{ Name = 'CSiBridge'; Dirs = 'CSiBridge*'; Exe = 'CSiBridge.exe'; Dlls = @('CSiBridge1.dll', 'CSiBridgev1.dll'); ProgId = 'CSI.CSiBridge.API.SapObject' }
}

function Get-VersionNumber([string]$dirName) {
    $m = [regex]::Match($dirName, '(\d+(\.\d+)?)')
    if (-not $m.Success) { return 0.0 }
    $v = [double]$m.Groups[1].Value
    if ($v -ge 2000) { $v = $v - 2000 }   # ETABS 2016 -> 16, SAFE 2014 -> 14
    return $v
}

function Find-Program($info) {
    $roots = @($env:ProgramFiles, $env:ProgramW6432, ${env:ProgramFiles(x86)}) | Where-Object { $_ } | Select-Object -Unique
    $cands = @()
    foreach ($r in $roots) {
        $csi = Join-Path $r 'Computers and Structures'
        if (Test-Path $csi) { $cands += Get-ChildItem -Path $csi -Directory -Filter $info.Dirs }
    }
    foreach ($d in ($cands | Sort-Object { Get-VersionNumber $_.Name } -Descending)) {
        $exe = Join-Path $d.FullName $info.Exe
        if (-not (Test-Path $exe)) { continue }
        foreach ($dll in $info.Dlls) {
            $p = Join-Path $d.FullName $dll
            if (Test-Path $p) { return @{ Dir = $d.FullName; Exe = $exe; Dll = $p; Label = $d.Name } }
        }
        return @{ Dir = $d.FullName; Exe = $exe; Dll = $null; Label = $d.Name }
    }
    return $null
}

function Invoke-ApplicationStart($app, [string]$ns) {
    # ETABS: ApplicationStart().  SAP2000 / CSiBridge: ApplicationStart(Units, Visible, FileName) with
    # optional arguments, which PowerShell cannot omit - pass them explicitly.
    try { return $app.ApplicationStart() } catch { }
    $unitsType = $null
    if ($ns) { $unitsType = ("$ns.eUnits" -as [type]) }
    if ($unitsType) {
        $u = [Enum]::ToObject($unitsType, 6)   # kN_m_C; the file's own units apply once it is opened
        foreach ($call in @({ $app.ApplicationStart($u, $true, '') }, { $app.ApplicationStart($u, $true) },
                            { $app.ApplicationStart($u) })) {
            try { return (& $call) } catch { }
        }
    }
    # last resort: reflection on the API interface, every argument at its default value
    $iface = $null
    if ($ns) { $iface = ("$ns.cOAPI" -as [type]) }
    if ($iface) {
        $m = $iface.GetMethods() | Where-Object { $_.Name -eq 'ApplicationStart' } |
             Sort-Object { $_.GetParameters().Count } | Select-Object -First 1
        $argv = @()
        foreach ($p in $m.GetParameters()) {
            if ($p.HasDefaultValue) { $argv += $p.DefaultValue }
            elseif ($p.ParameterType.IsEnum) { $argv += [Enum]::ToObject($p.ParameterType, 6) }
            elseif ($p.ParameterType -eq [bool]) { $argv += $true }
            else { $argv += '' }
        }
        return $m.Invoke($app, [object[]]$argv)
    }
    throw 'no usable ApplicationStart overload'
}

function Start-CsiProgram($found, $info) {
    $app = $null
    $ns = $null
    $errors = @()
    if ($found.Dll) {
        # same pattern as the ETABS export scripts: Add-Type + New-Object <namespace>.Helper
        # (namespace = DLL name: ETABSv1, SAP2000v1, SAFEv1, ETABS2016 ...)
        try {
            Add-Type -Path $found.Dll
            $ns = [System.IO.Path]::GetFileNameWithoutExtension($found.Dll)
            $helper = New-Object "$ns.Helper"
            $app = $helper.CreateObject($found.Exe)
        } catch { $errors += "Helper: $($_.Exception.Message)" }
    }
    if ($null -eq $app) {
        try {
            $t = [Type]::GetTypeFromProgID($info.ProgId)
            if ($t) { $app = [Activator]::CreateInstance($t) }
        } catch { $errors += "ProgID: $($_.Exception.Message)" }
    }
    if ($null -eq $app) { throw "$($info.Name): could not start the program through its API ($($errors -join '; '))" }
    if (-not $ns -and $found.Dll) { $ns = [System.IO.Path]::GetFileNameWithoutExtension($found.Dll) }
    try {
        $rc = Invoke-ApplicationStart $app $ns
    } catch {
        try { $app.ApplicationExit($false) | Out-Null } catch { }
        throw
    }
    if ($rc -ne 0) {
        try { $app.ApplicationExit($false) | Out-Null } catch { }
        throw "$($info.Name) ApplicationStart returned $rc"
    }
    if ($Hidden) { try { $app.Hide() | Out-Null } catch { } }
    return $app
}

