#!/bin/bash
# Open the Swift package in Xcode

cd "$(dirname "$0")/../KopisEngine"

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
    cd "$(dirname "$0")/../KopisEngine"
    xcodebuild -scheme KopisEngineGUI -showBuildSettings > /dev/null 2>&1
    echo "✓ Ready! Select KopisEngineGUI scheme and press ⌘R to run"
fi
