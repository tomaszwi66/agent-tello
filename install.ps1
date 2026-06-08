-param(
    [string]$Python = "python",
    [switch]$SkipWhisper,
    [switch]$SkipPiperVoice,
    [switch]$SkipYolo
)

$ErrorActionPreference = "Stop"

function Step($msg) {
    Write-Host ""
    Write-Host "== $msg ==" -ForegroundColor Cyan
}

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Step "Checking Python"
& $Python --version

if (!(Test-Path ".venv")) {
    Step "Creating virtual environment"
    & $Python -m venv .venv
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $VenvPython)) {
    throw "Cannot find $VenvPython"
}

Step "Upgrading pip"
& $VenvPython -m pip install --upgrade pip setuptools wheel

Step "Installing Tello Local AI Drone Agent dependencies"
& $VenvPython -m pip install -r requirements.txt

if (!$SkipYolo) {
    if (!(Test-Path "models\yolov8n.pt")) {
        Step "Downloading YOLO model"
        & $VenvPython scripts\download_yolo.py
    } else {
        Write-Host "YOLO already present: models\yolov8n.pt"
    }
}

if (!$SkipWhisper) {
    if (!(Test-Path "models\whisper-large-v3-turbo\model.bin")) {
        Step "Downloading faster-whisper large-v3-turbo"
        & $VenvPython scripts\download_whisper.py --repo h2oai/faster-whisper-large-v3-turbo --out models\whisper-large-v3-turbo
    } else {
        Write-Host "Whisper already present: models\whisper-large-v3-turbo"
    }
}

if (!$SkipPiperVoice) {
    if (!(Test-Path "models\tts\piper\en_US-lessac-medium.onnx")) {
        Step "Downloading Piper English voice"
        New-Item -ItemType Directory -Path "models\tts\piper" -Force | Out-Null
        & $VenvPython -m piper.download_voices en_US-lessac-medium --download-dir models\tts\piper
    } else {
        Write-Host "Piper voice already present: models\tts\piper\en_US-lessac-medium.onnx"
    }
}

Step "Checking local files"
& $VenvPython -c "from pathlib import Path; required=['scripts/s5_forward_yaw.py','scripts/s7_voice_agent.py','src/navigation/yaw_policy.py','models/yolov8n.pt']; missing=[p for p in required if not Path(p).exists()]; print('missing=' + str(missing)); raise SystemExit(1 if missing else 0)"

Write-Host ""
Write-Host "Tello Local AI Drone Agent setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Official full-agent command:"
Write-Host ".\run_agent.ps1"
Write-Host ""
Write-Host "Depth-only autonomous flight:"
Write-Host ".\run_depth_only.ps1"
