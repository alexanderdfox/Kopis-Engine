#!/bin/bash
# Open the Swift package in Xcode

set -e

# Get the absolute path to the project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KOPIS_ENGINE_DIR="$PROJECT_ROOT/KopisEngine"

# Check if the directory exists
if [ ! -d "$KOPIS_ENGINE_DIR" ]; then
    echo "Error: KopisEngine directory not found at: $KOPIS_ENGINE_DIR"
    exit 1
fi

# Check if Package.swift exists
if [ ! -f "$KOPIS_ENGINE_DIR/Package.swift" ]; then
    echo "Error: Package.swift not found in $KOPIS_ENGINE_DIR"
    exit 1
fi

# Change to the KopisEngine directory
cd "$KOPIS_ENGINE_DIR" || {
    echo "Error: Failed to change to KopisEngine directory"
    exit 1
}

echo "Opening Kopis Engine in Xcode..."
echo ""
echo "To run the GUI app:"
echo "1. Select the 'KopisEngineGUI' scheme from the scheme selector (top toolbar)"
echo "2. Press ⌘R (or click the Run button) to build and run"
echo ""
echo "Available schemes:"
echo "  - KopisEngineGUI (GUI app with SwiftUI)"
echo "  - KopisEngineApp (CLI executable)"
echo "  - KopisEngine (Library)"
echo ""

# Open Package.swift in Xcode
open Package.swift

# Wait a moment, then try to set the scheme
sleep 2

# Try to set the default scheme to KopisEngineGUI using xcodebuild
if command -v xcodebuild &> /dev/null; then
    echo "Setting default scheme to KopisEngineGUI..."
    # We're already in the KopisEngine directory, so just run xcodebuild
    xcodebuild -scheme KopisEngineGUI -showBuildSettings > /dev/null 2>&1
    echo "✓ Ready! Select KopisEngineGUI scheme and press ⌘R to run"
fi
