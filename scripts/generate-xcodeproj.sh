#!/bin/bash
# Generate .xcodeproj file for Kopis Engine

set -e

cd "$(dirname "$0")/../KopisEngine"

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

