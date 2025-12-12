#!/bin/bash
# Fix Xcode compilation issues for Kopis Engine

set -e

cd "$(dirname "$0")/../KopisEngine"

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

# Clean build
echo "3. Cleaning build folder..."
xcodebuild -project KopisEngine.xcodeproj -scheme KopisEngineGUI clean 2>&1 | tail -1
echo "   ✓ Cleaned"

# Test build
echo "4. Testing build..."
if xcodebuild -project KopisEngine.xcodeproj -scheme KopisEngineGUI -configuration Debug build 2>&1 | grep -q "BUILD SUCCEEDED"; then
    echo "   ✓ Build successful!"
    echo ""
    echo "Project is ready. Open with:"
    echo "  open KopisEngine.xcodeproj"
else
    echo "   ✗ Build failed - check errors above"
    exit 1
fi
