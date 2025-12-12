# Creating .xcodeproj File

## Option 1: Using xcodegen (Recommended)

1. **Install xcodegen**:
   ```bash
   brew install xcodegen
   ```

2. **Generate the project**:
   ```bash
   make xcodeproj
   ```
   Or:
   ```bash
   cd KopisEngine
   xcodegen generate
   ```

3. **Open the project**:
   ```bash
   open KopisEngine/KopisEngine.xcodeproj
   ```

## Option 2: Using Xcode Directly

1. **Open Package.swift in Xcode**:
   ```bash
   make xcode
   ```

2. **Save as Workspace**:
   - In Xcode: `File > Save As Workspace...`
   - Save as `KopisEngine.xcworkspace`
   - This creates a workspace that includes the Swift package

3. **Or Export Project**:
   - `File > Export > Export Project...`
   - This will create a .xcodeproj file

## Option 3: Manual Creation

If you prefer to create the project manually:

1. Open Xcode
2. `File > New > Project...`
3. Choose "macOS" > "App"
4. Name it "KopisEngineGUI"
5. Add the source files from `Sources/KopisEngineGUI/`
6. Link the KopisEngine library target

## Current Setup

The project uses **Swift Package Manager** which works directly with Xcode:

- **Package.swift** - Package manifest (already configured)
- **project.yml** - xcodegen configuration (for generating .xcodeproj)
- **Sources/** - Source code organized by target

## Quick Commands

```bash
make xcode        # Open Package.swift in Xcode (no .xcodeproj needed)
make xcodeproj    # Generate .xcodeproj (requires xcodegen)
make xcode-open   # Generate and open .xcodeproj
```

## Note

You don't actually need a .xcodeproj file! Xcode can work directly with Package.swift:
- Just run `make xcode` or open `Package.swift` in Xcode
- All schemes and targets are automatically available
- This is the modern Swift Package Manager approach

