# Kopis Engine - Quick Start Guide

## Run GUI in Xcode

### Step 1: Open in Xcode
```bash
make xcode
```

### Step 2: Select Scheme
- In Xcode's toolbar, click the scheme dropdown (next to stop/play buttons)
- Select **"KopisEngineGUI"**

### Step 3: Run
- Press **⌘R** (Command + R) or click the **Run** button
- The GUI app launches automatically in fullscreen

## Alternative: Run from Terminal

```bash
make gui    # Build and run GUI
make run    # Build and run CLI
make build  # Just build
make python # Run Python version
```

## Controls

- **WASD** - Move
- **Space** - Jump
- **Mouse** - Look around (auto-captured)
- **ESC** - Toggle mouse capture
- **F11** - Toggle fullscreen

## Available Schemes

When opening in Xcode, you'll see:
- **KopisEngineGUI** ⭐ - GUI app (use this!)
- **KopisEngineApp** - CLI version
- **KopisEngine** - Library only

## Troubleshooting

**Scheme not showing?**
- Make sure you opened `KopisEngine/Package.swift` (not a .xcodeproj)
- Xcode should auto-detect Swift Package Manager projects
- Try: `make xcode` from the project root

**Build errors?**
- Clean build: `Product > Clean Build Folder` (⇧⌘K)
- Test from terminal: `make build`

**App won't launch?**
- Verify scheme is "KopisEngineGUI"
- Check Xcode's issue navigator (⌘5)
