$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "Missing .venv. Create it with the project Python runtime before building."
}

& $Python -m pip install -r requirements-desktop.txt
& $Python -m PyInstaller --noconfirm --clean RavenWatchAgentsPro.spec

Write-Host ""
Write-Host "Built: $ProjectRoot\dist\RavenWatchAgentsPro\RavenWatchAgentsPro.exe"
