#!/bin/bash
# Run the Kopis Engine GUI application

cd "$(dirname "$0")/../KopisEngine"

echo "🚀 Launching Kopis Engine GUI..."
echo ""
echo "Make sure you're running: KopisEngineGUI (not KopisEngineApp)"
echo ""

# Build if needed
if [ ! -f ".build/debug/KopisEngineGUI" ]; then
    echo "Building GUI application..."
    swift build --product KopisEngineGUI
fi

# Run the GUI
echo "Starting GUI window..."
swift run KopisEngineGUI
