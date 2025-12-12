# Kopis Engine - Xcode Setup

## Quick Start

### Option 1: Open in Xcode (Recommended)
```bash
make xcode
```

Or manually:
```bash
cd KopisEngine
open Package.swift
```

### Option 2: Run from Command Line
```bash
make gui
```

## Running in Xcode

1. **Open the project**: Run `make xcode` or open `KopisEngine/Package.swift` in Xcode

2. **Select the scheme**: 
   - Click the scheme selector in the top toolbar (next to the stop/play buttons)
   - Select **"KopisEngineGUI"** from the dropdown

3. **Run the app**:
   - Press **⌘R** (Command + R) or click the **Run** button
   - The GUI app will launch automatically

## Available Schemes

- **KopisEngineGUI** - macOS GUI application with SwiftUI (recommended)
- **KopisEngineApp** - Command-line executable
- **KopisEngine** - Core library

## Features

- ✅ SwiftUI-based GUI
- ✅ Real-time raycasting 3D rendering
- ✅ Mouse capture for FPV controls
- ✅ WASD movement + mouse look
- ✅ Fullscreen support (F11)
- ✅ Game of Life blood patterns
- ✅ Entity rendering with depth sorting
- ✅ FPS counter and stats overlay

## Controls

- **WASD** - Move
- **Space** - Jump
- **Mouse** - Look around (captured in FPV mode)
- **ESC** - Toggle mouse capture
- **F11** - Toggle fullscreen

## Troubleshooting

### Scheme not showing?
- Make sure you opened `Package.swift` (not a .xcodeproj file)
- Xcode should automatically recognize Swift Package Manager projects
- Try: `File > Open` and select the `Package.swift` file

### Build errors?
- Make sure you're using Xcode 14+ (macOS 13+ required)
- Clean build folder: `Product > Clean Build Folder` (⇧⌘K)
- Try building from command line first: `make build`

### GUI not launching?
- Check that the scheme is set to "KopisEngineGUI" (not "KopisEngineApp")
- Look for errors in Xcode's issue navigator (⌘5)

## Project Structure

```
KopisEngine/
├── Package.swift              # Swift Package Manager manifest
├── Sources/
│   ├── KopisEngine/           # Core game engine library
│   ├── KopisEngineApp/        # CLI executable
│   └── KopisEngineGUI/        # macOS GUI app
│       ├── KopisEngineApp.swift    # Main app entry
│       ├── ContentView.swift       # SwiftUI view
│       ├── GameView.swift          # Game rendering view
│       └── GameViewModel.swift     # Game state management
```

## Building from Command Line

```bash
# From project root:
make build    # Build all targets
make gui      # Build and run GUI
make run      # Build and run CLI
make test     # Run tests
make python   # Run Python version
```

## Scripts

Utility scripts are located in the `scripts/` directory:
- `scripts/run-gui.sh` - Launch GUI application
- `scripts/open-xcode.sh` - Open project in Xcode
- `scripts/generate-xcodeproj.sh` - Generate .xcodeproj file
- `scripts/fix-xcode.sh` - Troubleshooting script
