-param(
    [string]$MicDevice = "",
    [string]$SpeakerDevice = "",
    [string]$BrainModel = "gemma4:e4b",
    [string]$VisionModel = "qwen2.5vl:3b"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Python)) {
    throw "Virtual environment not found. Run .\install.ps1 first."
}

$ArgsList = @(
    "scripts\s7_voice_agent.py",
    "--model", $BrainModel,
    "--vision-model", $VisionModel,
    "--agent-interval", "3",
    "--listen-input-warmup", "1.0"
)

if ($MicDevice -ne "") {
    $ArgsList += @("--mic-device", $MicDevice)
}

if ($SpeakerDevice -ne "") {
    $ArgsList += @("--speaker-device", $SpeakerDevice)
}

& $Python @ArgsList
