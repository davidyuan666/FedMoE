# PowerShell script to setup virtual environment and install dependencies
# Supports: uv, pdm, venv (in order of preference)

param(
    [string]$Tool = "auto",
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot $VenvDir
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"

# Function to check if command exists
function Test-Command {
    param([string]$Command)
    try {
        & $Command --version 2>$null | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Determine which tool to use
if ($Tool -eq "auto") {
    if (Test-Command "uv") {
        $Tool = "uv"
    } elseif (Test-Command "pdm") {
        $Tool = "pdm"
    } else {
        $Tool = "venv"
    }
}

Write-Host "Using $Tool for virtual environment management..." -ForegroundColor Cyan

# Setup virtual environment based on tool
if ($Tool -eq "uv") {
    if (-not (Test-Path $VenvPath)) {
        Write-Host "Creating virtual environment with uv at $VenvPath..." -ForegroundColor Yellow
        & uv venv $VenvPath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to create virtual environment with uv" -ForegroundColor Red
            exit 1
        }
    }
    
    Write-Host "Installing dependencies with uv pip..." -ForegroundColor Yellow
    $env:VIRTUAL_ENV = $VenvPath
    & uv pip install -r $RequirementsFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
    
} elseif ($Tool -eq "pdm") {
    if (-not (Test-Path $VenvPath)) {
        Write-Host "Creating virtual environment with pdm at $VenvPath..." -ForegroundColor Yellow
        Push-Location $ProjectRoot
        & pdm venv create --path $VenvPath
        Pop-Location
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to create virtual environment with pdm" -ForegroundColor Red
            exit 1
        }
    }
    
    Write-Host "Installing dependencies with pip in pdm venv..." -ForegroundColor Yellow
    $PipPath = Join-Path $VenvPath "Scripts\pip.exe"
    & $PipPath install -r $RequirementsFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
    
} else {
    # Use standard venv
    if (-not (Test-Path $VenvPath)) {
        Write-Host "Creating virtual environment with venv at $VenvPath..." -ForegroundColor Yellow
        & python -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to create virtual environment" -ForegroundColor Red
            exit 1
        }
    }
    
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    $PipPath = Join-Path $VenvPath "Scripts\pip.exe"
    & $PipPath install -r $RequirementsFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`nVirtual environment ready at $VenvPath" -ForegroundColor Green
Write-Host "To activate it manually:" -ForegroundColor Cyan
Write-Host "  .\$VenvPath\Scripts\Activate.ps1" -ForegroundColor White

