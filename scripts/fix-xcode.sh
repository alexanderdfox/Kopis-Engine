#!/bin/bash
# Fix Xcode compilation issues for Kopis Engine

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

# Check if project.yml exists (needed for xcodegen)
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

echo "Fixing Xcode compilation issues..."
echo ""

# Clean derived data
echo "1. Cleaning derived data..."
rm -rf ~/Library/Developer/Xcode/DerivedData/KopisEngine-* 2>/dev/null || true
echo "   ✓ Cleaned"

# Regenerate project
echo "2. Regenerating Xcode project..."
if command -v xcodegen &> /dev/null; then
    xcodegen generate
    echo "   ✓ Project regenerated"
else
    echo "   ⚠ xcodegen not found - install with: brew install xcodegen"
    exit 1
fi

# Check if .xcodeproj exists, if not generate it
if [ ! -d "KopisEngine.xcodeproj" ]; then
    echo "   ⚠ .xcodeproj not found, generating..."
    if command -v xcodegen &> /dev/null; then
        xcodegen generate
        echo "   ✓ Project generated"
    else
        echo "   ✗ Cannot generate project - xcodegen not found"
        exit 1
    fi
fi

# Clean build
echo "3. Cleaning build folder..."
if [ -d "KopisEngine.xcodeproj" ]; then
    xcodebuild -project KopisEngine.xcodeproj -scheme KopisEngineGUI clean 2>&1 | tail -1 || true
    echo "   ✓ Cleaned"
else
    echo "   ⚠ Skipping xcodebuild clean (project not found)"
fi

# Test build
echo "4. Testing build..."
if [ -d "KopisEngine.xcodeproj" ]; then
    if xcodebuild -project KopisEngine.xcodeproj -scheme KopisEngineGUI -configuration Debug build 2>&1 | grep -q "BUILD SUCCEEDED"; then
        echo "   ✓ Build successful!"
        echo ""
        echo "Project is ready. Open with:"
        echo "  open KopisEngine.xcodeproj"
        echo ""
        echo "Or use: make xcode-open"
    else
        echo "   ✗ Build failed - check errors above"
        exit 1
    fi
else
    echo "   ⚠ Cannot test build - .xcodeproj not found"
    echo "   Run 'make xcodeproj' first to generate the project"
    exit 1
fi
