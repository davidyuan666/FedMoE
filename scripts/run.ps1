# PowerShell script to setup environment and run FedMoE simulation

param(
    [string]$Config = "configs/sample_run.json",
    [string]$Tool = "auto",
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvPath = Join-Path $ProjectRoot $VenvDir

# Setup virtual environment
& "$ScriptDir\setup.ps1" -Tool $Tool -VenvDir $VenvDir

# Ensure PYTHONPATH includes project root
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $ProjectRoot
}

# Run simulation
Write-Host "`nRunning FedMoE simulation..." -ForegroundColor Cyan
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$RunScript = Join-Path $ScriptDir "run_fedmoe.py"

& $PythonPath $RunScript --config $Config

