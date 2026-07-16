param(
  [string]$HostAddress = "127.0.0.1",
  [int]$Port = 8765,
  [switch]$Reload
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  throw "Missing .venv. Create it and install requirements-web.txt first."
}

$Arguments = @("web_app.py", "--host", $HostAddress, "--port", $Port)
if ($Reload) {
  $Arguments += "--reload"
}

Set-Location $ProjectRoot
& $Python @Arguments
