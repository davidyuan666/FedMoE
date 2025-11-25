#!/bin/bash
# Bash script to setup virtual environment and install dependencies
# Supports: uv, pdm, venv (in order of preference)

set -e

TOOL="${1:-auto}"
VENV_DIR="${2:-.venv}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/$VENV_DIR"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"

# Detect Windows (MSYS/MinGW) vs POSIX layout
UNAME_OUT="$(uname -s)"
case "${UNAME_OUT}" in
    CYGWIN*|MINGW*|MSYS*)
        IS_WINDOWS=1
        BIN_DIR="Scripts"
        PYTHON_BIN="python.exe"
        ACTIVATE_CMD="source $VENV_PATH/Scripts/activate"
        ;;
    *)
        IS_WINDOWS=0
        BIN_DIR="bin"
        PYTHON_BIN="python"
        ACTIVATE_CMD="source $VENV_PATH/bin/activate"
        ;;
esac

# Function to check if command exists
check_command() {
    command -v "$1" >/dev/null 2>&1
}

# Determine which tool to use
if [ "$TOOL" = "auto" ]; then
    if check_command uv; then
        TOOL="uv"
    elif check_command pdm; then
        TOOL="pdm"
    else
        TOOL="venv"
    fi
fi

echo "Using $TOOL for virtual environment management..."

# Setup virtual environment based on tool
if [ "$TOOL" = "uv" ]; then
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment with uv at $VENV_PATH..."
        uv venv "$VENV_PATH"
    fi
    
    echo "Installing dependencies with uv pip..."
    export VIRTUAL_ENV="$VENV_PATH"
    if [ "$IS_WINDOWS" -eq 1 ]; then
        uv pip install --link-mode=copy -r "$REQUIREMENTS_FILE"
    else
        uv pip install -r "$REQUIREMENTS_FILE"
    fi
    
elif [ "$TOOL" = "pdm" ]; then
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment with pdm at $VENV_PATH..."
        cd "$PROJECT_ROOT"
        pdm venv create --path "$VENV_PATH"
    fi
    
    echo "Installing dependencies with pip in pdm venv..."
    PIP_PATH="$VENV_PATH/$BIN_DIR/pip"
    "$PIP_PATH" install -r "$REQUIREMENTS_FILE"
    
else
    # Use standard venv
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment with venv at $VENV_PATH..."
        python3 -m venv "$VENV_PATH"
    fi
    
    echo "Installing dependencies..."
    PIP_PATH="$VENV_PATH/$BIN_DIR/pip"
    "$PIP_PATH" install -r "$REQUIREMENTS_FILE"
fi

echo ""
echo "Virtual environment ready at $VENV_PATH"
echo "To activate it manually:"
echo "  $ACTIVATE_CMD"

