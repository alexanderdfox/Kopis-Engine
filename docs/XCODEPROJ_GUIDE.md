# Creating .xcodeproj File for Kopis Engine

## Quick Start

### Option 1: Install xcodegen and Generate (Easiest)

```bash
# Install xcodegen
brew install xcodegen

# Generate .xcodeproj
make xcodeproj

# Open the project
open KopisEngine/KopisEngine.xcodeproj
```

### Option 2: Use Package.swift (No .xcodeproj Needed)

Xcode works directly with Swift Package Manager:

```bash
make xcode
```

Then select the **KopisEngineGUI** scheme and press **⌘R**.

## What Was Created

1. **project.yml** - xcodegen configuration file
   - Defines all targets (KopisEngine, KopisEngineApp, KopisEngineGUI)
   - Configures build settings
   - Sets up schemes

2. **Info.plist** - App bundle information for GUI app

3. **scripts/generate-xcodeproj.sh** - Script to generate project using xcodegen

4. **scripts/create-xcodeproj.sh** - Alternative script with fallback methods

## Files Structure

```
KopisEngine/
├── Package.swift          # Swift Package Manager (works in Xcode)
├── project.yml           # xcodegen config (for generating .xcodeproj)
├── Info.plist            # App bundle info
└── Sources/
    ├── KopisEngine/      # Core library
    ├── KopisEngineApp/   # CLI executable
    └── KopisEngineGUI/   # GUI app
```

## Makefile Commands

```bash
make xcode        # Open Package.swift in Xcode (recommended)
make xcodeproj    # Generate .xcodeproj (requires xcodegen)
make xcode-open   # Generate and open .xcodeproj
```

## Why Two Approaches?

### Swift Package Manager (Package.swift)
- ✅ Modern approach
- ✅ No extra tools needed
- ✅ Works directly in Xcode
- ✅ Better dependency management

### .xcodeproj File
- ✅ Traditional Xcode project
- ✅ Can be shared with team
- ✅ More familiar to some developers
- ⚠️ Requires xcodegen tool

## Recommendation

**Use Package.swift approach** (`make xcode`) - it's simpler and works perfectly!

Only create .xcodeproj if you specifically need it for:
- Legacy build systems
- CI/CD that requires .xcodeproj
- Team preferences

