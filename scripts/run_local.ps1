param(
    [int]$Port = 8502
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$NgspiceDll = 'C:\Users\aakas\AppData\Local\Programs\KiCad\10.0\bin\ngspice.dll'
$ScratchLong = Join-Path $ProjectRoot 'scratch\ngspice_temp'
$LogRoot = Join-Path $ProjectRoot 'scratch\logs'
$PrivateTemp = Join-Path $env:LOCALAPPDATA 'Temp\SiliconPilotAI'

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Project Python environment is missing: $PythonExe"
}
if (-not (Test-Path -LiteralPath $NgspiceDll -PathType Leaf)) {
    throw "KiCad 10 ngspice DLL not found: $NgspiceDll"
}

$portsToCheck = @($Port, 8501, 8502, 8503, 8765) | Sort-Object -Unique
$listeners = @()
foreach ($candidatePort in $portsToCheck) {
    $listeners += @(Get-NetTCPConnection -State Listen -LocalPort $candidatePort -ErrorAction SilentlyContinue)
}
$listeners = @($listeners | Sort-Object LocalPort, OwningProcess -Unique)
if ($listeners.Count -gt 0) {
    foreach ($listener in $listeners) {
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)" -ErrorAction SilentlyContinue
        $url = "http://127.0.0.1:$($listener.LocalPort)"
        Write-Host "Existing listener on $url (PID $($listener.OwningProcess)); process: $($processInfo.Name)"
        try {
            $health = Invoke-WebRequest -UseBasicParsing -Uri "$url/_stcore/health" -TimeoutSec 3
            Write-Host "HTTP health: $($health.StatusCode)"
        }
        catch {
            Write-Warning "A listener exists, but its Streamlit health endpoint did not respond. No second server was started."
        }
    }
    exit 0
}

New-Item -ItemType Directory -Force -Path $ScratchLong, $LogRoot, $PrivateTemp | Out-Null

if (-not ('SiliconPilotShortPath' -as [type])) {
    Add-Type -TypeDefinition @'
using System.Text;
using System.Runtime.InteropServices;
public static class SiliconPilotShortPath {
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  public static extern uint GetShortPathName(string longPath, StringBuilder shortPath, uint bufferLength);
}
'@
}
$shortBuffer = New-Object System.Text.StringBuilder 32768
$shortLength = [SiliconPilotShortPath]::GetShortPathName($ScratchLong, $shortBuffer, [uint32]$shortBuffer.Capacity)
if ($shortLength -eq 0) {
    if ($ScratchLong.Contains(' ')) {
        throw 'Windows did not provide an 8.3 alias for the project scratch path; ngspice may fail on this space-containing path.'
    }
    $Scratch = $ScratchLong
}
else {
    $Scratch = $shortBuffer.ToString()
}

# These variables are inherited only by this launched process tree.
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:NGSPICE_DLL = $NgspiceDll
$env:PATH = (Split-Path -Parent $NgspiceDll) + ';' + $env:PATH
$env:TEMP = $Scratch
$env:TMP = $Scratch
$env:TMPDIR = $Scratch
$env:HOME = $Scratch.TrimEnd('\') + '\'
$env:SILICONPILOT_PRIVATE_TEMP = $PrivateTemp

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdoutLog = Join-Path $LogRoot "streamlit_$stamp.stdout.log"
$stderrLog = Join-Path $LogRoot "streamlit_$stamp.stderr.log"
$arguments = @(
    '-m', 'streamlit', 'run', 'app.py',
    "--server.address=127.0.0.1",
    "--server.port=$Port",
    '--server.headless=true'
)

$server = Start-Process -FilePath $PythonExe -ArgumentList $arguments -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
Write-Host "Started SiliconPilot AI (PID $($server.Id)); logs: $stdoutLog and $stderrLog"

$healthUrl = "http://127.0.0.1:$Port/_stcore/health"
$healthy = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Seconds 1
    if ($server.HasExited) { break }
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            $healthy = $true
            break
        }
    }
    catch {
        # The server can take a few seconds to initialize; keep polling this process only.
    }
}

if ($healthy) {
    Write-Host "HTTP health: 200 ($healthUrl)"
    Write-Host "Open http://127.0.0.1:$Port"
    exit 0
}

if ($server.HasExited) {
    Write-Warning "Streamlit exited with code $($server.ExitCode). Startup logs follow."
}
else {
    Write-Warning "Streamlit did not return HTTP 200 within 30 seconds. Stopping only the process started by this script."
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    Write-Warning 'Startup logs follow.'
}
Write-Host '--- stdout (last 80 lines) ---'
if (Test-Path -LiteralPath $stdoutLog) { Get-Content -LiteralPath $stdoutLog -Tail 80 }
Write-Host '--- stderr (last 80 lines) ---'
if (Test-Path -LiteralPath $stderrLog) { Get-Content -LiteralPath $stderrLog -Tail 80 }
exit 1
