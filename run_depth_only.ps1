-param(
    [double]$Duration = 60
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Python)) {
    throw "Virtual environment not found. Run .\install.ps1 first."
}

& $Python scripts\s5_forward_yaw.py --duration $Duration
