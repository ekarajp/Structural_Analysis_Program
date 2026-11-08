# Converts native ETABS (.edb), SAP2000 (.sdb), SAFE (.fdb) and CSiBridge (.bdb) models
# to text with the CSI program installed on this computer, through its official API:
#
#   * every model-definition database table  ->  <name>.<PROGRAM>.tables.txt   (CSI "TABLE:" text)
#   * ETABS also writes its own text model    ->  <name>.e2k
#
# The newest installed version of the program is used; it opens files of older versions.
# The original file is never modified: it is copied to the output folder and the copy is opened.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File csi_convert.ps1 [-OutDir <folder>] <file> [<file> ...]
#
# Output folder default: %LOCALAPPDATA%\StructuralStudio\import\<name>
# Exit code 0 when every file was converted.
param(
    [string]$OutDir = "",
    [switch]$Hidden,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Files
)
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'csi_api.ps1')

function Quote([string]$v) {
    if ($null -eq $v) { return '""' }
    if ($v -eq '' -or $v -match '\s' -or $v.Contains('"')) { return '"' + $v.Replace('"', "'") + '"' }
    return $v
}

# ---------------------------------------------------------------- table reading routes
# SAFE 22 returns numbers through GetTableForEditingArray with the trailing zeros of whole
# numbers removed (360 -> "36", E = 3000 -> "3"; text fields are intact).  Every route
# that the API offers is therefore checked against values read through the object API
# (point coordinates, material moduli) and the first route that reproduces them is used.
$Routes = @('EditArray', 'DisplayArray', 'EditCsv', 'DisplayCsv')

function Read-CsvRows([string]$path) {
    Add-Type -AssemblyName Microsoft.VisualBasic
    $rows = New-Object System.Collections.Generic.List[object]
    $p = New-Object Microsoft.VisualBasic.FileIO.TextFieldParser($path)
    try {
        $p.TextFieldType = 'Delimited'; $p.SetDelimiters(','); $p.HasFieldsEnclosedInQuotes = $true
        while (-not $p.EndOfData) { $rows.Add([string[]]$p.ReadFields()) }
    } finally { $p.Close() }
    return ,$rows
}

function Get-TableRows($db, [string]$key, [string]$route) {
    # returns @{ Fields = string[]; Rows = list of string[] } or $null
    $ver = 0; [string[]]$fields = @(); $nrec = 0; [string[]]$data = @()
    switch ($route) {
        'EditArray' {
            $rc = $db.GetTableForEditingArray($key, '', [ref]$ver, [ref]$fields, [ref]$nrec, [ref]$data)
        }
        'DisplayArray' {
            [string[]]$req = @(); [string[]]$inc = @()
            $rc = $db.GetTableForDisplayArray($key, [ref]$req, '', [ref]$ver, [ref]$inc, [ref]$nrec, [ref]$data)
            $fields = $inc
        }
        default {
            $tmp = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "ssconv_" + [guid]::NewGuid().ToString('N') + '.csv')
            try {
                if ($route -eq 'EditCsv') {
                    $rc = $db.GetTableForEditingCSVFile($key, '', [ref]$ver, $tmp, ',')
                } else {
                    [string[]]$req = @()
                    $rc = $db.GetTableForDisplayCSVFile($key, [ref]$req, '', [ref]$ver, $tmp, ',')
                }
                if ($rc -ne 0 -or -not (Test-Path -LiteralPath $tmp)) { return $null }
                $all = Read-CsvRows $tmp
            } finally { Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue }
            # header: the widest row near the top; a units row right after it is skipped
            $width = ($all | ForEach-Object { $_.Count } | Measure-Object -Maximum).Maximum
            $h = 0; while ($h -lt $all.Count -and $all[$h].Count -lt $width) { $h++ }
            if ($h -ge $all.Count) { return $null }
            $fields = $all[$h]
            $rows = New-Object System.Collections.Generic.List[object]
            for ($r = $h + 1; $r -lt $all.Count; $r++) {
                if ($all[$r].Count -ne $width) { continue }
                $rows.Add($all[$r])
            }
            return @{ Fields = $fields; Rows = $rows }
        }
    }
    if ($rc -ne 0 -or $nrec -le 0 -or $fields.Count -eq 0) { return $null }
    $nf = $fields.Count
    $rows = New-Object System.Collections.Generic.List[object]
    for ($r = 0; $r -lt $nrec; $r++) {
        $row = New-Object string[] $nf
        for ($f = 0; $f -lt $nf; $f++) { $row[$f] = $data[$r * $nf + $f] }
        $rows.Add($row)
    }
    return @{ Fields = $fields; Rows = $rows }
}

function Get-Col($t, [string[]]$names) {
    foreach ($n in $names) { for ($i = 0; $i -lt $t.Fields.Count; $i++) { if ($t.Fields[$i] -eq $n) { return $i } } }
    return -1
}

function Same-Number([string]$text, [double]$ref) {
    $v = 0.0
    if (-not [double]::TryParse($text, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$v)) { return $false }
    return [math]::Abs($v - $ref) -le (1e-6 * [math]::Max([math]::Abs($ref), 1e-3))
}

$PointTables = @('Point Object Connectivity', 'Joint Coordinates')
$MaterialTables = @('Material Properties - Basic Mechanical Properties', 'Material Properties 02 - Basic Mech Props')

function Get-ObjectCoords($sap, [string]$name) {
    $x = 0.0; $y = 0.0; $z = 0.0
    try { if ($sap.PointObj.GetCoordCartesian($name, [ref]$x, [ref]$y, [ref]$z, 'Global') -eq 0) { return @($x, $y, $z) } } catch { }
    try { if ($sap.PointObj.GetCoordCartesian($name, [ref]$x, [ref]$y, [ref]$z) -eq 0) { return @($x, $y, $z) } } catch { }
    return $null
}

function Get-ObjectModulus($sap, [string]$name) {
    $e = 0.0; $u = 0.0; $a = 0.0; $g = 0.0
    try { if ($sap.PropMaterial.GetMPIsotropic($name, [ref]$e, [ref]$u, [ref]$a, [ref]$g) -eq 0) { return $e } } catch { }
    try { if ($sap.PropMaterial.GetMPIsotropic($name, [ref]$e, [ref]$u, [ref]$a, [ref]$g, 0.0) -eq 0) { return $e } } catch { }
    try { if ($sap.PropMaterial.GetMPUniaxial($name, [ref]$e, [ref]$a) -eq 0) { return $e } } catch { }
    try { if ($sap.PropMaterial.GetMPUniaxial($name, [ref]$e, [ref]$a, 0.0) -eq 0) { return $e } } catch { }
    return $null
}

function Test-Route($sap, $db, [string[]]$keys, [string]$route) {
    # @{ Ok = n; Bad = n; Note = text }: values of the route compared with the object API
    $ok = 0; $bad = 0; $note = ''
    foreach ($key in $PointTables) {
        if ($keys -notcontains $key) { continue }
        $t = $null; try { $t = Get-TableRows $db $key $route } catch { return @{ Ok = 0; Bad = 1; Note = "$route failed: $($_.Exception.Message)" } }
        if ($null -eq $t) { return @{ Ok = 0; Bad = 1; Note = "$route returned no $key" } }
        $cn = Get-Col $t @('UniqueName', 'Joint', 'Name'); $cx = Get-Col $t @('GlobalX', 'X', 'XorR')
        $cy = Get-Col $t @('GlobalY', 'Y'); $cz = Get-Col $t @('GlobalZ', 'Z')
        if ($cn -lt 0 -or $cx -lt 0 -or $cy -lt 0 -or $cz -lt 0) { continue }
        $n = 0
        foreach ($row in $t.Rows) {
            if ($n -ge 60) { break }
            $xyz = Get-ObjectCoords $sap $row[$cn]
            if ($null -eq $xyz) { continue }
            $n++
            if ((Same-Number $row[$cx] $xyz[0]) -and (Same-Number $row[$cy] $xyz[1]) -and (Same-Number $row[$cz] $xyz[2])) { $ok++ }
            else { $bad++; if (-not $note) { $note = "point $($row[$cn]): table ($($row[$cx]), $($row[$cy]), $($row[$cz])) vs API ($($xyz -join ', '))" } }
        }
    }
    foreach ($key in $MaterialTables) {
        if ($keys -notcontains $key) { continue }
        $t = $null; try { $t = Get-TableRows $db $key $route } catch { continue }
        if ($null -eq $t) { continue }
        $cn = Get-Col $t @('Material', 'Name'); $ce = Get-Col $t @('E1')
        if ($cn -lt 0 -or $ce -lt 0) { continue }
        foreach ($row in $t.Rows) {
            if (-not $row[$ce]) { continue }
            $e = Get-ObjectModulus $sap $row[$cn]
            if ($null -eq $e -or $e -le 0) { continue }
            if (Same-Number $row[$ce] $e) { $ok++ }
            else { $bad++; if (-not $note) { $note = "material $($row[$cn]): table E1=$($row[$ce]) vs API $e" } }
        }
    }
    return @{ Ok = $ok; Bad = $bad; Note = $note }
}

function Repair-Rows($sap, [string]$key, $t) {
    # last resort: numbers that the object API can give back directly
    $fix = 0
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    if ($PointTables -contains $key) {
        $cn = Get-Col $t @('UniqueName', 'Joint', 'Name'); $cx = Get-Col $t @('GlobalX', 'X', 'XorR')
        $cy = Get-Col $t @('GlobalY', 'Y'); $cz = Get-Col $t @('GlobalZ', 'Z')
        if ($cn -ge 0 -and $cx -ge 0 -and $cy -ge 0 -and $cz -ge 0) {
            foreach ($row in $t.Rows) {
                $xyz = Get-ObjectCoords $sap $row[$cn]
                if ($null -eq $xyz) { continue }
                $row[$cx] = $xyz[0].ToString('R', $inv); $row[$cy] = $xyz[1].ToString('R', $inv); $row[$cz] = $xyz[2].ToString('R', $inv)
                $fix++
            }
        }
    }
    if ($MaterialTables -contains $key) {
        $cn = Get-Col $t @('Material', 'Name')
        foreach ($row in $t.Rows) {
            $w = 0.0; $m = 0.0
            try {
                if ($sap.PropMaterial.GetWeightAndMass($row[$cn], [ref]$w, [ref]$m) -eq 0) {
                    $c = Get-Col $t @('UnitWeight'); if ($c -ge 0) { $row[$c] = $w.ToString('R', $inv) }
                    $c = Get-Col $t @('UnitMass'); if ($c -ge 0) { $row[$c] = $m.ToString('R', $inv) }
                }
            } catch { }
            $e = 0.0; $u = 0.0; $a = 0.0; $g = 0.0; $got = $false
            try { $got = ($sap.PropMaterial.GetMPIsotropic($row[$cn], [ref]$e, [ref]$u, [ref]$a, [ref]$g) -eq 0) } catch { }
            if ($got) {
                foreach ($pair in @(@('E1', $e), @('U12', $u), @('A1', $a), @('G12', $g))) {
                    $c = Get-Col $t @($pair[0]); if ($c -ge 0) { $row[$c] = ([double]$pair[1]).ToString('R', $inv) }
                }
                $fix++
                continue
            }
            try { $got = ($sap.PropMaterial.GetMPUniaxial($row[$cn], [ref]$e, [ref]$a) -eq 0) } catch { }
            if ($got) {
                foreach ($pair in @(@('E1', $e), @('A1', $a))) {
                    $c = Get-Col $t @($pair[0]); if ($c -ge 0) { $row[$c] = ([double]$pair[1]).ToString('R', $inv) }
                }
                $fix++
            }
        }
    }
    return $fix
}

# ---------------------------------------------------------------- merging the routes
# Measured on the SAFE 22 verification models: the display route rounds decimals to the
# display format (0.0006944445 -> "0.00069", 1/12 -> "0.0833"); the editing route keeps
# them but drops the trailing zeros of whole numbers; and both lose the zeros of large
# whole numbers such as spring stiffnesses (1E+10 -> "1").  Merge-Routes takes, field by
# field, the editing value when it has decimals and rounds to the display value; the
# spring stiffnesses are then read from the object API (Repair-Springs).
function Test-Decimal([string]$text) { return ($text -match '^[+-]?\d*\.\d+([eE][+-]?\d+)?$') -or ($text -match '[eE]') }

function Merge-Routes($disp, $edit) {
    # $disp, $edit: tables (Fields, Rows) of one key from the two routes; returns the count of values taken
    if ($null -eq $disp -or $null -eq $edit -or $disp.Rows.Count -ne $edit.Rows.Count) { return 0 }
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $map = @{}
    for ($f = 0; $f -lt $disp.Fields.Count; $f++) {
        $g = Get-Col $edit @($disp.Fields[$f]); if ($g -ge 0) { $map[$f] = $g }
    }
    $k0 = 0; $k1 = Get-Col $edit @($disp.Fields[0])
    $taken = 0
    for ($r = 0; $r -lt $disp.Rows.Count; $r++) {
        $dr = $disp.Rows[$r]; $er = $edit.Rows[$r]
        if ($k1 -lt 0 -or $dr[$k0] -ne $er[$k1]) { continue }   # rows must be the same records
        foreach ($f in $map.Keys) {
            $dv = $dr[$f]; $ev = $er[$map[$f]]
            if (-not $dv -or -not $ev -or $dv -eq $ev -or -not (Test-Decimal $ev)) { continue }
            $d = 0.0; $e = 0.0
            if (-not [double]::TryParse($dv, [System.Globalization.NumberStyles]::Float, $inv, [ref]$d)) { continue }
            if (-not [double]::TryParse($ev, [System.Globalization.NumberStyles]::Float, $inv, [ref]$e)) { continue }
            # the display value is the editing value rounded: half a unit of its last digit
            $dec = 0; if ($dv -match '\.(\d+)') { $dec = $Matches[1].Length }
            $half = 0.5 * [math]::Pow(10, -$dec) * 1.0001
            if ([math]::Abs($e - $d) -le [math]::Max($half, 1e-9 * [math]::Abs($e))) { $dr[$f] = $ev; $taken++ }
        }
    }
    return $taken
}

function Repair-Springs($sap, [string]$key, $t) {
    # spring stiffnesses straight from the object API (the tables lose their zeros); returns
    # @{ Fixed = n; Missed = n }
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $fixed = 0; $missed = 0
    $cn = Get-Col $t @('Name')
    if ($cn -lt 0) { return @{ Fixed = 0; Missed = 0 } }
    foreach ($row in $t.Rows) {
        $name = $row[$cn]
        if ($key -match 'Line Springs') {
            $u1 = 0.0; $u2 = 0.0; $u3 = 0.0; $r1 = 0.0; $o2 = 0; $o3 = 0; $col = 0; $nt = ''; $gd = ''
            $ok = $false
            try { $ok = ($sap.PropLineSpring.GetLineSpringProp($name, [ref]$u1, [ref]$u2, [ref]$u3, [ref]$r1, [ref]$o2, [ref]$o3, [ref]$col, [ref]$nt, [ref]$gd) -eq 0) } catch { }
            if (-not $ok) { try { $ok = ($sap.PropLineSpring.GetLineSpringProp($name, [ref]$u1, [ref]$u2, [ref]$u3, [ref]$r1, [ref]$o3, [ref]$col, [ref]$nt, [ref]$gd) -eq 0) } catch { } }
            if ($ok) {
                $c = Get-Col $t @('VertStiff'); if ($c -ge 0) { $row[$c] = $u3.ToString('R', $inv) }
                $c = Get-Col $t @('RotStiff'); if ($c -ge 0) { $row[$c] = $r1.ToString('R', $inv) }
                $fixed++
            } else { $missed++ }
        } elseif ($key -match 'Area Springs') {
            $u1 = 0.0; $u2 = 0.0; $u3 = 0.0; $o3 = 0; $so = 0; $sp = ''; $el = 0.0; $pe = 0.0; $col = 0; $nt = ''; $gd = ''
            $ok = $false
            try { $ok = ($sap.PropAreaSpring.GetAreaSpringProp($name, [ref]$u1, [ref]$u2, [ref]$u3, [ref]$o3, [ref]$so, [ref]$sp, [ref]$el, [ref]$pe, [ref]$col, [ref]$nt, [ref]$gd) -eq 0) } catch { }
            if (-not $ok) { try { $ok = ($sap.PropAreaSpring.GetAreaSpringProp($name, [ref]$u1, [ref]$u2, [ref]$u3, [ref]$o3, [ref]$col, [ref]$nt, [ref]$gd) -eq 0) } catch { } }
            if ($ok) {
                $c = Get-Col $t @('SubModulus'); if ($c -ge 0) { $row[$c] = $u3.ToString('R', $inv) }
                $fixed++
            } else { $missed++ }
        } elseif ($key -match 'Point Springs') {
            $so = 0; [double[]]$k = New-Object double[] 6; $cs = ''; $sp = ''; $ft = ''; $pe = 0.0; $col = 0; $nt = ''; $gd = ''
            $ok = $false
            try { $ok = ($sap.PropPointSpring.GetPointSpringProp($name, [ref]$so, [ref]$k, [ref]$cs, [ref]$sp, [ref]$ft, [ref]$pe, [ref]$col, [ref]$nt, [ref]$gd) -eq 0) } catch { }
            if (-not $ok) { try { $ok = ($sap.PropPointSpring.GetPointSpringProp($name, [ref]$k, [ref]$col, [ref]$nt, [ref]$gd) -eq 0) } catch { } }
            if ($ok) {
                $i = 0
                foreach ($f in @('StiffUX', 'StiffUY', 'StiffUZ', 'StiffRX', 'StiffRY', 'StiffRZ')) {
                    $c = Get-Col $t @($f); if ($c -ge 0) { $row[$c] = $k[$i].ToString('R', $inv) }; $i++
                }
                $fixed++
            } else { $missed++ }
        }
    }
    return @{ Fixed = $fixed; Missed = $missed }
}

function Write-Tables($sap, [string]$path, [string]$program, [string]$label) {
    $db = $sap.DatabaseTables
    $n = 0; [string[]]$keys = @(); [string[]]$names = @(); [int[]]$imp = @()
    $rc = $db.GetAvailableTables([ref]$n, [ref]$keys, [ref]$names, [ref]$imp)
    if ($rc -ne 0) { throw "GetAvailableTables returned $rc" }
    $units = ''
    try { $units = $sap.GetPresentUnits().ToString() } catch { }
    $version = ''; $build = 0.0
    try { $sap.GetVersion([ref]$version, [ref]$build) | Out-Null } catch { }

    # choose the table route whose numbers agree with the object API
    $route = $null; $suspect = $false; $notes = @()
    foreach ($r in $Routes) {
        $test = Test-Route $sap $db $keys $r
        if ($test.Bad -eq 0) { $route = $r; Write-Host "     numbers checked: route $r ($($test.Ok) values agree with the object API)"; break }
        $notes += "$r : $($test.Bad) wrong ($($test.Note))"
        Write-Host "     route $r rejected: $($test.Bad) values differ from the object API, e.g. $($test.Note)"
    }
    if ($null -eq $route) {
        $route = 'EditArray'; $suspect = $true
        Write-Host "WARN no table route returns correct numbers: point coordinates and materials are repaired from"
        Write-Host "     the object API, other whole numbers in the tables may still be wrong (NumbersSuspect=Yes)"
    }

    $body = New-Object System.Text.StringBuilder
    $written = 0; $failed = @(); $repaired = 0; $merged = 0; $springsFixed = 0; $springsMissed = 0
    for ($i = 0; $i -lt $n; $i++) {
        $key = $keys[$i]
        if ($imp[$i] -le 0) { continue }          # results / design / non-importable tables
        if ($key -match '^(Program Control|PROGRAM CONTROL)$') { continue }
        $t = $null
        try { $t = Get-TableRows $db $key $route } catch { $failed += $key; continue }
        if ($null -eq $t -and $route -ne 'EditArray') {   # a route may not serve every table
            try { $t = Get-TableRows $db $key 'EditArray' } catch { }
        }
        if ($null -eq $t -or $t.Rows.Count -eq 0) { continue }
        if ($route -ne 'EditArray') {   # decimals at full precision from the editing route
            $e = $null; try { $e = Get-TableRows $db $key 'EditArray' } catch { }
            $merged += Merge-Routes $t $e
        }
        if ($suspect) { $repaired += Repair-Rows $sap $key $t }
        if ($key -match '^Spring Property Definitions') {
            $sr = Repair-Springs $sap $key $t
            $springsFixed += $sr.Fixed; $springsMissed += $sr.Missed
        }
        $sb = $body
        [void]$sb.AppendLine("TABLE:  `"$key`"")
        $nf = $t.Fields.Count
        foreach ($row in $t.Rows) {
            $parts = New-Object System.Collections.Generic.List[string]
            for ($f = 0; $f -lt $nf; $f++) {
                $v = $row[$f]
                if ($null -eq $v -or $v -eq '') { continue }
                $parts.Add("$($t.Fields[$f])=$(Quote $v)")
            }
            [void]$sb.AppendLine('   ' + ($parts -join '   '))
        }
        [void]$sb.AppendLine('')
        $written++
    }
    [void]$body.AppendLine('END TABLE DATA')
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine('TABLE:  "PROGRAM CONTROL"')
    $extra = "   TableRoute=$route"
    if ($route -ne 'EditArray') { $extra += "   DecimalsFrom=EditArray" }
    if ($suspect) { $extra += '   NumbersSuspect=Yes' }
    if ($springsFixed -gt 0) { $extra += "   SpringsFrom=ObjectAPI" }
    if ($springsMissed -gt 0) { $extra += '   SpringsSuspect=Yes' }
    [void]$sb.AppendLine("   ProgramName=$(Quote $program)   Version=$(Quote $version)   CurrUnits=$(Quote $units)   ConvertedBy=$(Quote ('Structural Studio / ' + $label))$extra")
    [void]$sb.AppendLine('')
    [void]$sb.Append($body.ToString())
    [System.IO.File]::WriteAllText($path, $sb.ToString(), (New-Object System.Text.UTF8Encoding($false)))
    if ($repaired -gt 0) { Write-Host "     $repaired rows repaired from the object API" }
    if ($merged -gt 0) { Write-Host "     $merged decimals taken at full precision from the editing route" }
    if ($springsFixed -gt 0) { Write-Host "     $springsFixed spring properties read from the object API" }
    if ($springsMissed -gt 0) { Write-Host "WARN $springsMissed spring properties could not be read from the object API: their stiffness in the tables may have lost zeros (SpringsSuspect=Yes)" }
    return @{ Tables = $written; Failed = $failed }
}

if (-not $Files -or $Files.Count -eq 0) {
    Write-Host 'Usage: csi_convert.ps1 [-OutDir <folder>] <model.edb|.sdb|.fdb> ...'
    exit 2
}

$exitCode = 0
$started = @{}
$broken = @{}
try {
    foreach ($file in $Files) {
        if (-not $file) { continue }
        $src = (Resolve-Path -LiteralPath $file).Path
        $ext = [System.IO.Path]::GetExtension($src).ToLower()
        $info = $Programs[$ext]
        if ($null -eq $info) { Write-Host "SKIP $src (not .edb/.sdb/.fdb/.bdb)"; $exitCode = 1; continue }
        $base = [System.IO.Path]::GetFileNameWithoutExtension($src)
        $dest = $OutDir
        if (-not $dest) { $dest = Join-Path $env:LOCALAPPDATA ("StructuralStudio\import\" + $base) }
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        $found = Find-Program $info
        if ($null -eq $found) { Write-Host "FAIL $src : $($info.Name) is not installed"; $exitCode = 1; continue }
        if ($broken.ContainsKey($info.Name)) { Write-Host "SKIP $src : $($broken[$info.Name])"; $exitCode = 1; continue }
        if (-not $started.ContainsKey($info.Name)) {
            Write-Host "Starting $($found.Label) ..."
            try {
                $started[$info.Name] = Start-CsiProgram $found $info
            } catch {
                $broken[$info.Name] = "$($found.Label) could not be started: $($_.Exception.Message)"
                Write-Host "FAIL $($broken[$info.Name])"
                $exitCode = 1
                continue
            }
        }
        try {
            $app = $started[$info.Name]
            $sap = $app.SapModel
            $copy = Join-Path $dest ([System.IO.Path]::GetFileName($src))
            if ($copy -ne $src) { Copy-Item -LiteralPath $src -Destination $copy -Force }
            Write-Host "Opening $([System.IO.Path]::GetFileName($src)) ..."
            $rc = $sap.File.OpenFile($copy)
            if ($rc -ne 0) { throw "OpenFile returned $rc" }
            $tables = Join-Path $dest ("$base." + $info.Name + '.tables.txt')
            $res = Write-Tables $sap $tables $info.Name $found.Label
            Write-Host "OK   $tables ($($res.Tables) tables)"
            if ($res.Failed.Count -gt 0) { Write-Host ("     tables not readable: " + ($res.Failed -join ', ')) }
            if ($info.Name -eq 'ETABS') {
                $e2k = Join-Path $dest "$base.e2k"
                $rc = $sap.File.Save($e2k)          # ETABS writes its text model for an .e2k name
                $et = Join-Path $dest ($base + '.$et')  # ... or the .$et companion of the .edb
                if (-not (Test-Path -LiteralPath $e2k) -and (Test-Path -LiteralPath $et)) {
                    Copy-Item -LiteralPath $et -Destination $e2k -Force
                }
                if (Test-Path -LiteralPath $e2k) { Write-Host "OK   $e2k" } else { Write-Host "     e2k export rc=$rc" }
            }
        } catch {
            Write-Host "FAIL $src : $($_.Exception.Message)"
            $exitCode = 1
        }
    }
} finally {
    foreach ($app in $started.Values) { try { $app.ApplicationExit($false) | Out-Null } catch { } }
}
exit $exitCode
