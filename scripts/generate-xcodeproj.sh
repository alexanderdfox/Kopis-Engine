#!/bin/bash
# Generate .xcodeproj file for Kopis Engine

set -e

# Get the absolute path to the project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KOPIS_ENGINE_DIR="$PROJECT_ROOT/KopisEngine"

# Check if the directory exists
if [ ! -d "$KOPIS_ENGINE_DIR" ]; then
    echo "Error: KopisEngine directory not found at: $KOPIS_ENGINE_DIR"
    exit 1
fi

# Check if project.yml exists
if [ ! -f "$KOPIS_ENGINE_DIR/project.yml" ]; then
    echo "Error: project.yml not found in $KOPIS_ENGINE_DIR"
    echo "xcodegen requires project.yml to generate the Xcode project"
    exit 1
fi

# Change to the KopisEngine directory
cd "$KOPIS_ENGINE_DIR" || {
    echo "Error: Failed to change to KopisEngine directory"
    exit 1
}

echo "Generating Xcode project for Kopis Engine..."
echo ""

# Check if xcodegen is installed
if command -v xcodegen &> /dev/null; then
    echo "✓ Found xcodegen"
    echo "Generating .xcodeproj from project.yml..."
    xcodegen generate
    echo ""
    echo "✓ Xcode project generated successfully!"
    echo ""
    echo "To open in Xcode:"
    echo "  open KopisEngine.xcodeproj"
    echo ""
    echo "Or use: make xcode"
else
    echo "⚠ xcodegen not found"
    echo ""
    echo "Installing xcodegen..."
    echo "Run: brew install xcodegen"
    echo ""
    echo "Or use the Swift Package Manager approach:"
    echo "  make xcode  # Opens Package.swift in Xcode"
    echo ""
    echo "Alternative: Generate project using Xcode"
    echo "1. Open Package.swift in Xcode"
    echo "2. File > Save As Workspace..."
    echo "3. Or use: xcodebuild -resolvePackageDependencies"
    exit 1
fi

