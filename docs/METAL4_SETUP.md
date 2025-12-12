# Metal 4 Setup for Kopis Engine

This project is configured for **Metal 4** (Metal Shading Language 2.4+) on macOS 13.0+.

## Requirements

- **macOS 13.0+** (Ventura or later)
- **Metal 4 capable GPU** (Apple Silicon or recent Intel Mac with Metal 4 support)
- **Xcode 14.0+** (for Metal 4 features)

## Metal 4 Features Enabled

### 1. **Metal Shading Language 2.4+**
   - Modern compute shaders
   - Enhanced texture access patterns
   - Improved performance optimizations

### 2. **Framework Integration**
   - `Metal` framework (core Metal API)
   - `MetalKit` framework (MTKView, utilities)
   - `QuartzCore` framework (CAMetalDrawable)

### 3. **Build Settings**
   - Metal shader debug info enabled
   - C++17 standard for Metal interop
   - Proper framework linking

## Project Structure

### Metal Files
- `KopisEngine/Sources/KopisEngineGUI/Shaders.metal` - Metal 4 compatible shaders
- `KopisEngine/Sources/KopisEngineGUI/MetalRenderer.swift` - Metal rendering engine
- `KopisEngine/Sources/KopisEngineGUI/MetalView.swift` - MTKView integration

### Shader Features
- **Raycasting compute shader** (`raycast_compute`) - GPU-accelerated Doom-style raycasting
- **Entity rendering shader** (`render_entities_compute`) - Billboard sprite rendering
- **Fullscreen quad shaders** - For texture rendering

## Building

### Using Xcode
```bash
make xcode
# Select KopisEngineGUI scheme and press ⌘R
```

### Using Command Line
```bash
# From project root:
make xcodeproj
cd KopisEngine
xcodebuild -project KopisEngine.xcodeproj -scheme KopisEngineGUI -configuration Debug build
```

### Using Swift Package Manager
```bash
# From project root:
make build

# Or from KopisEngine directory:
cd KopisEngine
swift build
```

## Metal 4 Capabilities

The project uses Metal 4 features including:

1. **Modern Compute Shaders**
   - Parallel raycasting for walls
   - Efficient texture writes
   - Optimized threadgroup dispatch

2. **MTKView Integration**
   - Native Metal rendering
   - Automatic drawable management
   - 60 FPS target

3. **GPU-Accelerated Rendering**
   - All raycasting on GPU
   - No CPU fallback needed
   - Direct drawable blitting

## Troubleshooting

### Metal Not Available
If you see "Metal is not supported on this device":
- Ensure you're running on macOS 13.0+
- Check GPU compatibility
- Verify Metal framework is linked

### Shader Compilation Errors
If shaders fail to compile:
- Check Xcode version (14.0+ required)
- Verify `Shaders.metal` is included in build
- Check for Metal syntax errors

### Performance Issues
- Ensure running on Metal 4 capable GPU
- Check that compute shaders are being used
- Verify no CPU fallback is active

## Metal 4 vs Metal 3

This project uses **Metal 4** (MSL 2.4+), which provides:
- Better performance on Apple Silicon
- Enhanced compute shader capabilities
- Improved memory management
- Modern shader language features

For compatibility with older systems, the project falls back gracefully.
