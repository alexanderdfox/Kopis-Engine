# Kopis Engine - Project Structure

This document describes the organized structure of the Kopis Engine project.

## Directory Structure

```
Kopis-Engine/
├── KopisEngine/              # Swift Package (main project)
│   ├── Sources/              # Source code
│   │   ├── KopisEngine/      # Core engine library
│   │   ├── KopisEngineApp/   # CLI executable
│   │   └── KopisEngineGUI/   # GUI executable (SwiftUI + Metal)
│   ├── Package.swift         # Swift Package Manager manifest
│   ├── project.yml           # xcodegen configuration
│   ├── Info.plist            # App bundle info
│   └── .xcode.env            # Xcode environment
│
├── scripts/                  # Utility scripts
│   ├── open-xcode.sh         # Open project in Xcode
│   ├── generate-xcodeproj.sh # Generate .xcodeproj
│   ├── create-xcodeproj.sh   # Alternative project creation
│   ├── fix-xcode.sh          # Troubleshooting script
│   └── run-gui.sh            # Launch GUI app
│
├── docs/                     # Documentation
│   ├── README.md             # Documentation index
│   ├── QUICK_START.md        # Quick start guide
│   ├── XCODE_SETUP.md        # Xcode setup
│   ├── METAL4_SETUP.md       # Metal 4 integration
│   └── ...                   # Other documentation files
│
├── .github/                  # GitHub workflows
│   └── workflows/
│       └── static.yml
│
├── Makefile                  # Build automation
├── README.md                 # Main project README
├── ROADMAP.md                # Development roadmap
├── LICENSE                   # License file
├── .gitignore                # Git ignore rules
│
├── kopis_engine.py           # Python version
├── index.html                # Web version
├── requirements.txt          # Python dependencies
└── kopis_engine.jpeg         # Project image
```

## Key Directories

### `KopisEngine/`
The main Swift package containing:
- **Sources/KopisEngine/** - Core game engine library
- **Sources/KopisEngineApp/** - Command-line interface
- **Sources/KopisEngineGUI/** - macOS GUI application with Metal 4 rendering

### `scripts/`
Utility scripts for development:
- Build automation
- Xcode project generation
- Troubleshooting tools

### `docs/`
All project documentation:
- Setup guides
- API documentation
- Troubleshooting guides

## File Organization

### Source Code
- **Swift**: `KopisEngine/Sources/`
- **Python**: `kopis_engine.py` (root)
- **Web**: `index.html` (root)

### Build Files
- **Swift Package**: `KopisEngine/Package.swift`
- **Xcode Project**: `KopisEngine/project.yml` (generates `.xcodeproj`)
- **Makefile**: Root level for build automation

### Documentation
- All `.md` files moved to `docs/` directory
- Main `README.md` and `ROADMAP.md` remain in root

## Cleanup

The following were removed:
- `KopisEngine/KopisEngineGame/` - Unused Xcode template project
- Duplicate documentation files
- Temporary build artifacts (handled by `.gitignore`)

## Recent Updates

### Sound System
- ✅ AVFoundation-based sound manager
- ✅ Programmatic sound generation (footsteps, collisions, movement)
- ✅ Integrated into game engine

### Metal 4 Integration
- ✅ Metal Shading Language 2.4+ support
- ✅ GPU-accelerated raycasting
- ✅ MTKView-based rendering
- ✅ Full SwiftUI/AppKit integration

## Build Artifacts (Ignored)

The following are automatically ignored by `.gitignore`:
- `KopisEngine/.build/` - Swift build artifacts
- `KopisEngine/.swiftpm/` - Swift Package Manager cache
- `*.xcodeproj/` - Generated Xcode projects
- `*.xcworkspace/` - Xcode workspaces
- `DerivedData/` - Xcode derived data
