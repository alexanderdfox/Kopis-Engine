#!/bin/bash
# Runner script for Kopis Engine Python version
# This script is executed by Xcode to run the Python game engine

set -e

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_SCRIPT="$PROJECT_ROOT/kopis_engine.py"

echo "=========================================="
echo "Kopis Engine - Python Version"
echo "=========================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    echo "Please install Python 3:"
    echo "  brew install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ Found: $PYTHON_VERSION"
echo ""

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Python script not found at: $PYTHON_SCRIPT"
    exit 1
fi

# Check if virtual environment exists, create if not
VENV_DIR="$PROJECT_ROOT/KopisEnginePython/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install/update dependencies
echo "Checking dependencies..."
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing/updating dependencies from requirements.txt..."
    pip install --upgrade pip --quiet
    pip install -r "$REQUIREMENTS_FILE" --quiet
    echo "✓ Dependencies ready"
else
    echo "⚠ Warning: requirements.txt not found"
fi

echo ""
echo "Starting Kopis Engine..."
echo "=========================================="
echo ""

# Change to project root and run the Python script
cd "$PROJECT_ROOT"
python3 "$PYTHON_SCRIPT"
