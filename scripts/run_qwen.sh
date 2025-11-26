#!/bin/bash
# 快速启动脚本：直接运行 Qwen 模型微调

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${1:-.venv}"
BASE_MODEL="${2:-Qwen/Qwen2-0.5B-Instruct}"

VENV_PATH="$PROJECT_ROOT/$VENV_DIR"

# 确定平台和路径
UNAME_OUT="$(uname -s)"
case "${UNAME_OUT}" in
    CYGWIN*|MINGW*|MSYS*)
        BIN_DIR="Scripts"
        PYTHON_PATH="$VENV_PATH/Scripts/python.exe"
        PIP_PATH="$VENV_PATH/Scripts/pip.exe"
        ;;
    *)
        BIN_DIR="bin"
        PYTHON_PATH="$VENV_PATH/bin/python"
        PIP_PATH="$VENV_PATH/bin/pip"
        ;;
esac

# 设置虚拟环境（如果不存在则创建并安装依赖）
if [ ! -d "$VENV_PATH" ]; then
    echo "虚拟环境不存在，正在创建..."
    bash "$SCRIPT_DIR/setup.sh" "auto" "$VENV_DIR"
else
    # 检查依赖是否已安装
    if [ ! -f "$PIP_PATH" ]; then
        echo "虚拟环境存在但未完整设置，正在安装依赖..."
        bash "$SCRIPT_DIR/setup.sh" "auto" "$VENV_DIR"
    fi
fi

# 确保 PYTHONPATH 包含项目根目录
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 直接运行 Qwen 微调
echo ""
echo "=========================================="
echo "FedMoE - Qwen 模型真实微调"
echo "=========================================="
echo "基础模型: $BASE_MODEL"
echo "虚拟环境: $VENV_PATH"
echo ""

"$PYTHON_PATH" "$PROJECT_ROOT/run_qwen_finetune.py"

