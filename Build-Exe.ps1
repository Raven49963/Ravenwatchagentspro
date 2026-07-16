$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "Missing .venv. Create it with the project Python runtime before building."
}

& $Python -m pip install -r requirements.txt

& $Python -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name TradingAgentsCN `
  --paths src `
  --collect-submodules akshare `
  --collect-data akshare `
  --collect-submodules yfinance `
  --collect-data yfinance `
  --collect-submodules curl_cffi `
  --collect-data curl_cffi `
  trading_agents_app.py

Write-Host ""
Write-Host "Built: $ProjectRoot\dist\TradingAgentsCN\TradingAgentsCN.exe"
