#!/bin/bash
# Generate .xcodeproj file for Kopis Engine Python version

set -e

# Get the absolute path to the script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_PROJECT_DIR="$PROJECT_ROOT/KopisEnginePython"

# Check if the directory exists
if [ ! -d "$PYTHON_PROJECT_DIR" ]; then
    echo "Error: KopisEnginePython directory not found at: $PYTHON_PROJECT_DIR"
    exit 1
fi

# Check if project.yml exists
if [ ! -f "$PYTHON_PROJECT_DIR/project.yml" ]; then
    echo "Error: project.yml not found in $PYTHON_PROJECT_DIR"
    echo "xcodegen requires project.yml to generate the Xcode project"
    exit 1
fi

# Change to the KopisEnginePython directory
cd "$PYTHON_PROJECT_DIR" || {
    echo "Error: Failed to change to KopisEnginePython directory"
    exit 1
}

echo "Generating Xcode project for Kopis Engine (Python)..."
echo ""

# Check if xcodegen is installed
if command -v xcodegen &> /dev/null; then
    echo "✓ Found xcodegen"
    echo "Generating .xcodeproj from project.yml..."
    xcodegen generate
    echo ""
    echo "✓ Xcode project generated successfully!"
    echo ""
    echo "Project location: $PYTHON_PROJECT_DIR/KopisEnginePython.xcodeproj"
    echo ""
    echo "To open in Xcode:"
    echo "  open KopisEnginePython.xcodeproj"
    echo ""
    echo "Or use: make xcode-python"
else
    echo "⚠ xcodegen not found"
    echo ""
    echo "Installing xcodegen..."
    echo "Run: brew install xcodegen"
    echo ""
    echo "Then run this script again or use: make xcodeproj-python"
    exit 1
fi
