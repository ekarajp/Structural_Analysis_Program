# Starts Structural Studio (Python desktop app).  On the first run it installs the
# package and its UI libraries (PySide6, VTK, openpyxl) into the project's Python
# at E:\REA ETABS\tools\python (internet needed once, about 300 MB).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File "E:\REA ETABS\studio\scripts\Start-StructuralStudio.ps1" [model]
#
# Every step prints what it is doing and how long it took, and the app's own output goes to
# E:\REA ETABS\cache\app-stderr.log / app-stdout.log, so a slow start is visible and a failed
# start shows its reason instead of silently doing nothing.
param([string]$Model = "")
# 'Continue': in Windows PowerShell 5.1 anything a program writes to stderr (a Python traceback,
# pip warnings) becomes a terminating error under 'Stop' - exit codes are checked instead.
$ErrorActionPreference = 'Continue'
$Studio = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)   # ...\studio
$Repo   = Split-Path -Parent $Studio                                              # E:\REA ETABS
$LogDir = Join-Path $Repo 'cache'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$sw = [System.Diagnostics.Stopwatch]::StartNew()
# every start writes what it did to cache\launcher.log (also when the console is closed early)
$LaunchLog = Join-Path $LogDir 'launcher.log'
Set-Content -Path $LaunchLog -Value ("Structural Studio launcher, " + (Get-Date -Format s)) -ErrorAction SilentlyContinue
function Say([string]$Text, [string]$Color = 'Gray') {
    $line = "[{0,5:N1} s] {1}" -f $sw.Elapsed.TotalSeconds, $Text
    Write-Host $line -ForegroundColor $Color
    Add-Content -Path $LaunchLog -Value $line -ErrorAction SilentlyContinue
}
function Stop-WithMessage([string]$Text) {
    Write-Host ''
    Say $Text 'Red'
    Write-Host "Launcher log: $LaunchLog" -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}
# Runs Python with a time limit; returns @{Code; Out; TimedOut}.  A hung Python (antivirus
# scan of a locked file, a stuck DLL load) is reported instead of waiting for ever.
function Invoke-PyTimed([string]$Arguments, [int]$Seconds) {
    $o = [System.IO.Path]::GetTempFileName(); $e = [System.IO.Path]::GetTempFileName()
    $p = Start-Process -FilePath $Py -ArgumentList $Arguments -NoNewWindow -PassThru `
        -RedirectStandardOutput $o -RedirectStandardError $e
    $null = $p.Handle  # keeps the exit code readable (Start-Process -PassThru quirk)
    $done = $p.WaitForExit($Seconds * 1000)
    if (-not $done) { try { $p.Kill() } catch { } }
    $text = ((Get-Content -Path $o, $e -ErrorAction SilentlyContinue) -join "`n")
    Remove-Item -Path $o, $e -Force -ErrorAction SilentlyContinue
    $code = if ($done) { $p.ExitCode } else { -1 }
    return @{ Code = $code; Out = $text; TimedOut = (-not $done) }
}

# 1. find Python: the project's venv first, then the launcher / PATH
$Py = Join-Path $Repo 'tools\python\Scripts\python.exe'
if (-not (Test-Path $Py)) {
    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) { $Py = (& py -3 -c "import sys; print(sys.executable)").Trim() }
}
if (-not (Test-Path $Py)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $Py = $cmd.Source }
}
if (-not $Py -or -not (Test-Path $Py)) {
    Write-Host 'Python 3.11+ was not found. Install it from https://www.python.org and run this again.' -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}
Say "Python: $Py"

# 2. first run (or after an update added a library): install Structural Studio + UI libraries
$ImportCheck = "import PySide6, vtk, numpy, scipy, openpyxl, structstudio.app.mainwindow"
$CheckSeconds = 180
function Test-Ready {
    $r = Invoke-PyTimed ('-c "' + $ImportCheck + '"') $CheckSeconds
    if ($r.TimedOut) {
        Say ("The library check did not finish in {0} s - Python is blocked (antivirus scanning E:, " -f $CheckSeconds) 'Yellow'
        Say 'a full disk, or another Python process holding the files). Running Python processes:' 'Yellow'
        Get-Process -Name python, pythonw -ErrorAction SilentlyContinue |
            ForEach-Object { Say ("  PID {0}  started {1}  {2}" -f $_.Id, $_.StartTime, $_.Path) 'Yellow' }
        Stop-WithMessage 'Structural Studio was not started: the library check hung (see above).'
    }
    if ($r.Code -ne 0 -and $r.Out) { Add-Content -Path $LaunchLog -Value $r.Out -ErrorAction SilentlyContinue }
    return ($r.Code -eq 0)
}
# a Structural Studio that is already running (its window may be minimised or on another screen)
$running = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*structstudio.app*' })
foreach ($r in $running) {
    Say ("Structural Studio is already running (PID {0}, since {1}) - check the taskbar; a new window is started anyway." -f `
        $r.ProcessId, $r.CreationDate) 'Yellow'
}
Say "Checking the libraries (can take 20-40 s right after a restart; limit $CheckSeconds s) ..."
if (-not (Test-Ready)) {
    $logDir = Join-Path $env:LOCALAPPDATA 'StructuralStudio'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $log = Join-Path $logDir 'install.log'
    Say 'Installing Structural Studio and its UI libraries (PySide6, VTK) - a few minutes, internet needed ...' 'Cyan'
    & $Py -m pip install --upgrade pip 2>&1 | Tee-Object -FilePath $log
    & $Py -m pip install -e "$Studio[app]" 2>&1 | Tee-Object -FilePath $log -Append
    if (-not (Test-Ready)) {
        Write-Host ''
        Write-Host 'Structural Studio could not be started. The reason is below; the install log is' -ForegroundColor Red
        Write-Host "$log" -ForegroundColor Red
        & $Py -c $ImportCheck 2>&1 | Out-Host
        Read-Host 'Press Enter to close'
        exit 1
    }
}
Say 'Libraries are ready.'

# 3. once: the fast equation solver (Intel MKL PARDISO through pypardiso, about 200 MB).
#    The analysis works without it (SciPy SuperLU, slower on large models); a failed
#    install is not retried every start - delete the marker file to try again.
#    Once the marker exists nothing is checked here (importing MKL on every start is slow).
$marker = Join-Path $LogDir 'pypardiso-install-tried.txt'
if (-not (Test-Path $marker)) {
    $null = & $Py -c "import pypardiso" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Say 'Installing the fast equation solver (Intel MKL PARDISO) - first time only, about 200 MB ...' 'Cyan'
        $pipCache = Join-Path $Repo 'cache\pip'
        & $Py -m pip install --cache-dir $pipCache pypardiso 2>&1 | Out-Host
        Set-Content -Path $marker -Value ("tried " + (Get-Date -Format s) + ", exit code " + $LASTEXITCODE)
    } else {
        Set-Content -Path $marker -Value ("present " + (Get-Date -Format s))
    }
}

# 4. start the app without a console window; its output is kept in the cache folder.
#    A running Structural Studio keeps its own log files open: this start then writes to new
#    (dated) ones instead of failing on the locked files.
$outLog = Join-Path $LogDir 'app-stdout.log'
$errLog = Join-Path $LogDir 'app-stderr.log'
Remove-Item -Path $outLog, $errLog -Force -ErrorAction SilentlyContinue
if ((Test-Path $outLog) -or (Test-Path $errLog)) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $outLog = Join-Path $LogDir "app-stdout-$stamp.log"
    $errLog = Join-Path $LogDir "app-stderr-$stamp.log"
    Say "The previous log files are in use (another Structural Studio is running); writing $errLog" 'Yellow'
}
$env:PYTHONFAULTHANDLER = '1'
$env:PYTHONUNBUFFERED = '1'
$appArgs = @('-X', 'faulthandler', '-m', 'structstudio.app')
if ($Model) { $appArgs += ('"' + $Model + '"') }
Say 'Starting Structural Studio ...'
$proc = $null
try {
    $proc = Start-Process -FilePath $Py -ArgumentList $appArgs -WorkingDirectory $Studio -WindowStyle Hidden `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru -ErrorAction Stop
} catch {
    Stop-WithMessage ("Structural Studio could not be started: " + $_.Exception.Message)
}
if ($null -eq $proc) { Stop-WithMessage 'Structural Studio could not be started (no process was created).' }
$null = $proc.Handle  # keeps the exit code readable

# watch the first seconds: if the program ends by itself, show why
$deadline = (Get-Date).AddSeconds(15)
while ((Get-Date) -lt $deadline -and -not $proc.HasExited) { Start-Sleep -Milliseconds 300 }
if ($proc.HasExited) {
    Write-Host ''
    Say ("Structural Studio stopped right after starting (exit code {0})." -f $proc.ExitCode) 'Red'
    if (Test-Path $errLog) {
        $tail = Get-Content -Path $errLog -Tail 40
        $tail | Out-Host
        Add-Content -Path $LaunchLog -Value $tail -ErrorAction SilentlyContinue
    }
    Write-Host "Full output: $errLog" -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}
Say ("Structural Studio is running (PID {0}). The window should appear now; you can close this console." -f $proc.Id) 'Green'
Start-Sleep -Seconds 3
