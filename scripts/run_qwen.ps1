# PowerShell 快速启动脚本：直接运行 Qwen 模型微调

param(
    [string]$VenvDir = ".venv",
    [string]$BaseModel = "Qwen/Qwen2-0.5B-Instruct"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvPath = Join-Path $ProjectRoot $VenvDir

# 设置虚拟环境（如果不存在则创建并安装依赖）
if (-not (Test-Path $VenvPath)) {
    Write-Host "虚拟环境不存在，正在创建..." -ForegroundColor Yellow
    & "$ScriptDir\setup.ps1" -Tool "auto" -VenvDir $VenvDir
} else {
    # 检查依赖是否已安装
    $PipPath = Join-Path $VenvPath "Scripts\pip.exe"
    if (-not (Test-Path $PipPath)) {
        Write-Host "虚拟环境存在但未完整设置，正在安装依赖..." -ForegroundColor Yellow
        & "$ScriptDir\setup.ps1" -Tool "auto" -VenvDir $VenvDir
    }
}

# 确保 PYTHONPATH 包含项目根目录
$env:PYTHONPATH = $ProjectRoot
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $ProjectRoot
}

# 直接运行 Qwen 微调
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "FedMoE - Qwen 模型真实微调" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "基础模型: $BaseModel" -ForegroundColor Green
Write-Host "虚拟环境: $VenvPath" -ForegroundColor Green
Write-Host ""

$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$RunScript = Join-Path $ProjectRoot "run_qwen_finetune.py"

& $PythonPath $RunScript

