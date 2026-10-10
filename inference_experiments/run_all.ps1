$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

python .\experiments.py --out .\results --repeats 5 --seed 12345

Write-Host ""
Write-Host "Done. Open results\REPORT.md for the summary."
