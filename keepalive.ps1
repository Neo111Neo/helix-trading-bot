# Keeps Helix running: starts run.py, and if it ever exits (crash, killed, Windows), logs it and starts it again
# after 5 seconds. Stop Helix for good by closing this watchdog first (Stop-Process on its PowerShell), then run.py.
# Start it hidden:  Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','keepalive.ps1' -WindowStyle Hidden
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot
# only one watchdog at a time (a second one would start a second Helix fighting for port 8787)
$mutex = New-Object System.Threading.Mutex($false, "Global\HelixKeepalive")
if (-not $mutex.WaitOne(0)) { exit 0 }
$env:PYTHONUNBUFFERED = "1"
$log = Join-Path $PSScriptRoot "data\keepalive.log"
while ($true) {
    Add-Content -Path $log -Value ("{0:u}  starting Helix" -f (Get-Date).ToUniversalTime())
    $p = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "run.py","--no-browser" `
        -WorkingDirectory $PSScriptRoot -RedirectStandardOutput "data\engine.out.log" `
        -RedirectStandardError "data\engine.err.log" -WindowStyle Hidden -PassThru
    $p.WaitForExit()
    Add-Content -Path $log -Value ("{0:u}  Helix exited (code {1}) - restarting in 5 s" -f (Get-Date).ToUniversalTime(), $p.ExitCode)
    Start-Sleep -Seconds 5
}
