#!/bin/bash
# Create .xcodeproj file for Kopis Engine using Xcode's built-in tools

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

# Change to the KopisEngine directory
cd "$KOPIS_ENGINE_DIR" || {
    echo "Error: Failed to change to KopisEngine directory"
    exit 1
}

echo "Creating Xcode project for Kopis Engine..."
echo ""

# Method 1: Try using xcodegen if available
if command -v xcodegen &> /dev/null; then
    echo "✓ Using xcodegen to generate project..."
    xcodegen generate
    if [ -f "KopisEngine.xcodeproj" ]; then
        echo "✓ Project created: KopisEngine.xcodeproj"
        exit 0
    fi
fi

# Method 2: Create project using Xcode command line
echo "Attempting to create project using Xcode tools..."
echo ""

# Check if we can use xcodebuild
if command -v xcodebuild &> /dev/null; then
    echo "Opening Package.swift in Xcode..."
    echo "Please follow these steps:"
    echo "1. Wait for Xcode to open"
    echo "2. Go to File > Save As Workspace..."
    echo "3. Save as 'KopisEngine.xcworkspace'"
    echo "4. Or use File > Export > Export Project..."
    echo ""
    open Package.swift
    
    echo ""
    echo "Alternatively, install xcodegen for automatic generation:"
    echo "  brew install xcodegen"
    echo "Then run: make xcodeproj"
else
    echo "⚠ Xcode command line tools not found"
    echo ""
    echo "Please install Xcode command line tools:"
    echo "  xcode-select --install"
    echo ""
    echo "Or install xcodegen:"
    echo "  brew install xcodegen"
    echo "Then run: make xcodeproj"
    exit 1
fi

