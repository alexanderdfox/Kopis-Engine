# iOS Python Setup Guide

This guide explains how to set up Python to run on iOS using PythonKit.

## Overview

The iOS target uses **PythonKit** to embed and execute Python code. However, PythonKit requires a Python runtime to be embedded in the iOS app bundle.

## Prerequisites

1. **Python-Apple-support**: You need to embed a Python runtime for iOS
   - Download from: https://github.com/beeware/Python-Apple-support
   - Or build your own Python framework for iOS

2. **PythonKit**: Already added as a Swift Package dependency

## Setup Steps

### 1. Get Python Runtime for iOS

You have two options:

#### Option A: Use Python-Apple-support (Recommended)

1. Download the Python framework for iOS from:
   ```
   https://github.com/beeware/Python-Apple-support/releases
   ```

2. Extract `Python.xcframework` 

3. Add it to your Xcode project:
   - Drag `Python.xcframework` into the project
   - Ensure it's added to the `KopisEngineiOS` target
   - Set "Embed & Sign" in the target's General settings

#### Option B: Build Python for iOS

1. Clone Python-Apple-support:
   ```bash
   git clone https://github.com/beeware/Python-Apple-support.git
   cd Python-Apple-support
   ```

2. Build Python framework:
   ```bash
   ./build.sh
   ```

3. Copy the generated `Python.xcframework` to your project

### 2. Bundle Python Scripts

The Python engine script (`kopis_engine.py`) needs to be included in the app bundle:

1. In Xcode, add `kopis_engine.py` to the project
2. Ensure it's added to the `KopisEngineiOS` target
3. Check "Copy Bundle Resources" in Build Phases

### 3. Handle Dependencies

The Python engine uses several dependencies that may not work on iOS:

- **torch**: Very large, may not be practical for iOS
- **transformers**: Large models, may need to be excluded
- **pygame**: Not available on iOS (use SwiftUI instead)
- **numpy**: Should work with Python-Apple-support

**Recommended approach:**
- Create a simplified version of the engine for iOS
- Remove or stub out heavy dependencies (torch, transformers)
- Replace pygame rendering with SwiftUI
- Keep core game logic in Python

### 4. Update PythonEngineManager

The `PythonEngineManager.swift` file needs to be updated to:

1. Initialize Python with the embedded runtime:
   ```swift
   // Set Python home to the embedded framework
   let pythonHome = Bundle.main.path(forResource: "Python", ofType: "framework")
   PythonLibrary.useLibrary(at: pythonHome)
   ```

2. Load Python modules from the app bundle

3. Handle iOS-specific limitations (no pygame, limited libraries)

## Limitations

1. **App Size**: Embedding Python significantly increases app size (100MB+)
2. **Performance**: Python on iOS is slower than native Swift
3. **Dependencies**: Many Python packages don't work on iOS
4. **App Store**: Large apps may face review issues

## Alternative Approaches

### Option 1: Backend Service
- Run Python engine on a server
- iOS app communicates via API
- Better performance, no size issues

### Option 2: Swift Port
- Port the game logic to Swift
- Use the existing Swift engine from `KopisEngine/`
- Native performance, smaller app size

### Option 3: Hybrid
- Core game logic in Swift
- Use Python for AI/ML features only
- Smaller Python runtime, better performance

## Current Implementation

The current iOS implementation provides:

- ✅ PythonKit integration
- ✅ Basic Python engine manager
- ✅ SwiftUI interface
- ⚠️ Requires Python runtime to be embedded
- ⚠️ Needs adaptation for iOS (no pygame)

## Next Steps

1. Embed Python runtime using Python-Apple-support
2. Create iOS-adapted version of `kopis_engine.py`
3. Replace pygame rendering with SwiftUI/Metal
4. Test and optimize performance

## Resources

- [PythonKit Documentation](https://github.com/pvieito/PythonKit)
- [Python-Apple-support](https://github.com/beeware/Python-Apple-support)
- [Python Playground Example](https://github.com/kewlbear/PythonPlayground)
