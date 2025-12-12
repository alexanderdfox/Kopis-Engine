# Xcode Setup for Kopis Engine GUI

## Quick Start

1. **Open the project**:
   ```bash
   make xcode
   ```
   Or manually: Open `KopisEngine/Package.swift` in Xcode

2. **Select the GUI scheme**:
   - In Xcode's top toolbar, click the scheme selector (next to the stop/play buttons)
   - Choose **"KopisEngineGUI"** from the dropdown

3. **Run the app**:
   - Press **⌘R** (Command + R) or click the **Run** button
   - The GUI app will launch in fullscreen

## Scheme Selection

The project has three schemes:

- **KopisEngineGUI** ⭐ - macOS GUI app (use this one!)
- **KopisEngineApp** - Command-line executable
- **KopisEngine** - Core library

## Troubleshooting

### Can't find the scheme?
- Make sure you opened `Package.swift` (not a .xcodeproj)
- Xcode should auto-detect Swift Package Manager projects
- Try: `File > Open` and select `KopisEngine/Package.swift`

### Build fails?
- Ensure Xcode 14+ is installed (macOS 13+ required)
- Clean build: `Product > Clean Build Folder` (⇧⌘K)
- Test from command line: `make build`

### App doesn't launch?
- Verify scheme is "KopisEngineGUI"
- Check Xcode's issue navigator (⌘5) for errors
- Try running from terminal: `make gui`

## Features

✅ Full 3D raycasting rendering  
✅ Mouse capture for FPV controls  
✅ WASD movement + mouse look  
✅ Fullscreen on startup  
✅ Game of Life blood patterns  
✅ Entity billboard sprites  
✅ FPS counter overlay  

## Controls

- **WASD** - Move
- **Space** - Jump  
- **Mouse** - Look (captured automatically)
- **ESC** - Toggle mouse capture
- **F11** - Toggle fullscreen
