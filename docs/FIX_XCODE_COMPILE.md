# Fixing Xcode Compilation Issues

## Common Issues and Solutions

### 1. Missing Framework Imports

If you see errors about missing symbols:
- **CVDisplayLink** → Requires `import CoreVideo`
- **CGWarpMouseCursorPosition** → Requires `import CoreGraphics`
- **SwiftUI/AppKit** → Should be automatically available

### 2. Module Not Found: KopisEngine

If Xcode can't find the KopisEngine module:
1. Clean build folder: `Product > Clean Build Folder` (⇧⌘K)
2. Close and reopen Xcode
3. Build the KopisEngine target first, then KopisEngineGUI

### 3. Code Signing Errors

The project is configured to NOT require code signing:
- `CODE_SIGNING_REQUIRED = NO` in Debug configuration
- Should build without signing issues

### 4. Info.plist Not Found

The Info.plist is configured in project.yml:
- Path: `Info.plist` (in KopisEngine directory)
- `INFOPLIST_FILE = Info.plist` in build settings

## Quick Fixes

### Regenerate Project
```bash
# From project root:
make xcodeproj

# Or manually:
cd KopisEngine
xcodegen generate
```

### Clean Build
```bash
# In Xcode: Product > Clean Build Folder (⇧⌘K)
# Or from command line:
cd KopisEngine
xcodebuild -project KopisEngine.xcodeproj -scheme KopisEngineGUI clean
```

### Verify Build Settings
```bash
xcodebuild -project KopisEngine.xcodeproj -target KopisEngineGUI -showBuildSettings | grep INFOPLIST
```

## Current Configuration

- ✅ Code signing disabled for development
- ✅ Info.plist properly configured
- ✅ All frameworks linked (SwiftUI, AppKit, CoreGraphics, CoreVideo)
- ✅ KopisEngine module dependency set

## If Still Having Issues

1. **Delete Derived Data**:
   ```bash
   rm -rf ~/Library/Developer/Xcode/DerivedData/KopisEngine-*
   ```

2. **Reopen Xcode**:
   - Close Xcode completely
   - Reopen `KopisEngine.xcodeproj`

3. **Check Scheme**:
   - Make sure "KopisEngineGUI" scheme is selected
   - Not "KopisEngine" or "KopisEngineApp"

4. **Build Order**:
   - Build "KopisEngine" target first
   - Then build "KopisEngineGUI"
