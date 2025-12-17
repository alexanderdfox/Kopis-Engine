# Kopis Engine - Python Xcode Project

This directory contains the Xcode project configuration for running the Python version of Kopis Engine.

## Overview

The Python Xcode project uses a Swift wrapper executable (`KopisEnginePythonRunner`) that:
- Checks for Python 3 installation
- Creates and manages a virtual environment
- Installs Python dependencies from `requirements.txt`
- Executes `kopis_engine.py` with proper environment setup

## Prerequisites

- **Xcode** (15.0 or later)
- **xcodegen** - Install with: `brew install xcodegen`
- **Python 3** - Usually pre-installed on macOS, or install with: `brew install python3`

## Quick Start

### Generate the Xcode Project

```bash
# From the project root
make xcodeproj-python
```

Or manually:
```bash
cd KopisEnginePython
./generate-xcodeproj.sh
```

### Open in Xcode

```bash
# From the project root
make xcode-python
```

Or manually:
```bash
open KopisEnginePython/KopisEnginePython.xcodeproj
```

### Run from Xcode

**macOS Target:**
1. Open `KopisEnginePython.xcodeproj` in Xcode
2. Select the `KopisEnginePython` scheme from the toolbar
3. Press `⌘R` (or click Run) to build and run

**iOS Target:**
1. Open `KopisEnginePython.xcodeproj` in Xcode
2. Select the `KopisEngineiOS` scheme from the toolbar
3. Choose a simulator or connected iOS device
4. Press `⌘R` (or click Run) to build and run

**Important:** To run Python on iOS, you need to:
1. Embed a Python runtime (see `IOS_PYTHON_SETUP.md` for details)
2. Add `kopis_engine.py` to the app bundle
3. Adapt the engine for iOS (remove pygame, handle dependencies)

The iOS target uses PythonKit to execute Python code. See `IOS_PYTHON_SETUP.md` for complete setup instructions.

## Project Structure

```
KopisEnginePython/
├── project.yml              # Xcode project configuration (xcodegen)
├── generate-xcodeproj.sh    # Script to generate .xcodeproj
├── Info-iOS.plist           # iOS app Info.plist
├── IOS_PYTHON_SETUP.md      # Guide for setting up Python on iOS
├── Sources/
│   ├── KopisEnginePythonRunner/
│   │   └── main.swift       # Swift wrapper that executes Python (macOS)
│   └── KopisEngineiOS/
│       ├── KopisEngineiOSApp.swift      # iOS app entry point
│       ├── ContentView.swift            # iOS main view
│       ├── PythonEngineManager.swift   # Python engine manager for iOS
│       └── PythonEngineView.swift       # SwiftUI view for Python engine
└── Scripts/
    └── run_python.sh        # Alternative shell script runner
```

## How It Works

1. **Swift Wrapper**: The `main.swift` file is compiled into an executable that:
   - Locates the Python script (`../kopis_engine.py`)
   - Creates a virtual environment if needed
   - Installs dependencies from `requirements.txt`
   - Executes the Python script with the virtual environment activated

2. **Virtual Environment**: A Python virtual environment is automatically created in `KopisEnginePython/venv/` to isolate dependencies.

3. **Dependencies**: Python packages are installed from the root `requirements.txt` file.

## Troubleshooting

### xcodegen not found
```bash
brew install xcodegen
```

### Python 3 not found
```bash
brew install python3
```

### Build errors
- Make sure Python 3 is installed and accessible
- Check that `kopis_engine.py` exists in the project root
- Verify `requirements.txt` is present in the project root

### Virtual environment issues
If the virtual environment becomes corrupted:
```bash
rm -rf KopisEnginePython/venv
```
The next run will recreate it automatically.

## Alternative: Run Python Directly

You can also run the Python version directly without Xcode:
```bash
# From project root
make python
# or
python3 kopis_engine.py
```

## Targets

### KopisEnginePython (macOS)
- Swift wrapper executable that runs the Python game engine
- Creates and manages Python virtual environment
- Installs dependencies automatically
- Runs `kopis_engine.py` from the project root

### KopisEngineiOS (iOS)
- SwiftUI-based iOS app with PythonKit integration
- Can run Python code on iOS (requires embedded Python runtime)
- Features:
  - Python engine manager (`PythonEngineManager.swift`)
  - SwiftUI rendering interface (`PythonEngineView.swift`)
  - Integration with PythonKit for Python execution
- **Note**: Requires Python runtime to be embedded (see `IOS_PYTHON_SETUP.md`)
- Limitations:
  - pygame not available on iOS (use SwiftUI for rendering)
  - Large dependencies (torch, transformers) may not be practical
  - App size will be significantly larger (~100MB+)

## Notes

- The virtual environment (`venv/`) is gitignored and created automatically
- The Swift wrapper handles all Python environment setup
- Dependencies are installed automatically on first run
- The Python script runs from the project root directory
- iOS target requires iOS 16.0+ and supports iPhone and iPad
