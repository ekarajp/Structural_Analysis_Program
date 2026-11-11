# Collects sample / verification models shipped with the installed SAP2000 and SAFE
# (and the ETABS verification examples already in F:\REA ETABS\reference) and converts
# them to text with the installed programs, for the Structural Studio import tests.
# Nothing outside F:\REA ETABS is written.  Progress is shown in the window and in
# tests\verification\data\csi_samples.log.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File "F:\REA ETABS\studio\scripts\Export-CsiSamples.ps1" [-Only safe|sap2000|etabs]
#
param([string[]]$Only = @())
$ErrorActionPreference = 'Continue'
$Studio  = Split-Path -Parent $PSScriptRoot
$Data    = Join-Path $Studio 'tests\verification\data'
$Convert = Join-Path $Studio 'structstudio\io\csi_convert.ps1'
$Log     = Join-Path $Data 'csi_samples.log'
$MaxPerProgram = 40
New-Item -ItemType Directory -Force -Path $Data | Out-Null
"$(Get-Date -Format s) start" | Set-Content -Path $Log

function Say([string]$text) { Write-Host $text; Add-Content -Path $Log -Value $text }

function Get-VersionNumber([string]$name) {
    $m = [regex]::Match($name, '(\d+)'); if (-not $m.Success) { return 0 }
    $v = [int]$m.Groups[1].Value; if ($v -ge 2000) { $v -= 2000 }; return $v
}

function Invoke-Convert([string]$outDir, [string[]]$files) {
    # stream the converter output to the window and the log while it runs
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Convert -OutDir $outDir @files 2>&1 |
        ForEach-Object { Say ("  " + $_.ToString()) }
}

$csi = Join-Path $env:ProgramFiles 'Computers and Structures'
foreach ($p in @(@{ Name = 'sap2000'; Dirs = 'SAP2000*'; Ext = '*.sdb' }, @{ Name = 'safe'; Dirs = 'SAFE*'; Ext = '*.fdb' })) {
    if ($Only.Count -gt 0 -and $Only -notcontains $p.Name) { continue }
    try {
        $dirs = @(Get-ChildItem -Path $csi -Directory -Filter $p.Dirs -ErrorAction SilentlyContinue |
                  Sort-Object { Get-VersionNumber $_.Name } -Descending)
        if ($dirs.Count -eq 0) { Say "$($p.Name): not installed"; continue }
        $native = Join-Path $Data "$($p.Name)\native"
        if (Test-Path $native) { Remove-Item -LiteralPath $native -Recurse -Force }
        New-Item -ItemType Directory -Force -Path $native | Out-Null
        # newest version first, one copy of each example (analysis examples before design examples)
        $seen = @{}
        $picked = @()
        foreach ($d in $dirs) {
            $all = @(Get-ChildItem -Path $d.FullName -Recurse -Filter $p.Ext -ErrorAction SilentlyContinue)
            $all = @($all | Sort-Object @{ Expression = { if ($_.FullName -match 'Analysis') { 0 } else { 1 } } }, Name)
            foreach ($f in $all) {
                if ($picked.Count -ge $MaxPerProgram) { break }
                $key = $f.Name.ToLower()
                if ($seen.ContainsKey($key)) { continue }
                $seen[$key] = $true
                $picked += $f
            }
        }
        Say "$($p.Name): $($picked.Count) model files (newest: $($dirs[0].Name))"
        $copies = @()
        foreach ($f in $picked) {
            $dest = Join-Path $native $f.Name
            Copy-Item -LiteralPath $f.FullName -Destination $dest -Force
            $copies += $dest
        }
        if ($copies.Count -gt 0) {
            Say "$($p.Name): converting with the installed program (it opens in its own window) ..."
            Invoke-Convert (Join-Path $Data $p.Name) $copies
        }
    } catch {
        Say "$($p.Name): ERROR $($_.Exception.Message)"
    }
}

# ETABS verification examples -> database tables (to check the tables importer against .e2k)
$etabsSrc = 'F:\REA ETABS\reference\etabs21\Verification\Analysis Examples'
if ((Test-Path $etabsSrc) -and ($Only.Count -eq 0 -or $Only -contains 'etabs')) {
    $edbs = @(Get-ChildItem -Path $etabsSrc -Filter 'Example *.edb' | Select-Object -First 30 | ForEach-Object { $_.FullName })
    if ($edbs.Count -gt 0) {
        Say "etabs: converting $($edbs.Count) verification examples ..."
        Invoke-Convert (Join-Path $Data 'etabs\tables') $edbs
    }
}
Say "$(Get-Date -Format s) done"
