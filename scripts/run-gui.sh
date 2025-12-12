#!/bin/bash
# Run the Kopis Engine GUI application

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

echo "🚀 Launching Kopis Engine GUI (SwiftUI/AppKit/Metal)..."
echo ""

# Check if Swift is available
if ! command -v swift &> /dev/null; then
    echo "Error: Swift is not installed. Install Xcode or Swift toolchain."
    exit 1
fi

# Build the GUI application
echo "Building GUI application (SwiftUI/AppKit/Metal integration)..."
swift build --product KopisEngineGUI
if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi

echo ""
echo "✓ Build complete"
echo "✓ Starting GUI window with full Metal rendering..."
echo ""

# Run the GUI
swift run KopisEngineGUI
