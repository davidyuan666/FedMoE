#!/bin/bash
# Bash script to setup environment and run FedMoE simulation

set -e

CONFIG="${1:-configs/sample_run.json}"
TOOL="${2:-auto}"
VENV_DIR="${3:-.venv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/$VENV_DIR"

# Determine platform for path handling
UNAME_OUT="$(uname -s)"
case "${UNAME_OUT}" in
    CYGWIN*|MINGW*|MSYS*)
        PYTHON_PATH="$VENV_PATH/Scripts/python.exe"
        PATH_SEP=";"
        ;;
    *)
        PYTHON_PATH="$VENV_PATH/bin/python"
        PATH_SEP=":"
        ;;
esac

# Setup virtual environment
bash "$SCRIPT_DIR/setup.sh" "$TOOL" "$VENV_DIR"

# Ensure PYTHONPATH includes project root
if [ -n "$PYTHONPATH" ]; then
    export PYTHONPATH="$PROJECT_ROOT$PATH_SEP$PYTHONPATH"
else
    export PYTHONPATH="$PROJECT_ROOT"
fi

# Run simulation
echo ""
echo "Running FedMoE simulation..."
RUN_SCRIPT="$SCRIPT_DIR/run_fedmoe.py"

"$PYTHON_PATH" "$RUN_SCRIPT" --config "$CONFIG"

