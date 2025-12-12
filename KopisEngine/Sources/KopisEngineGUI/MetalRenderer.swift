import Metal
import MetalKit
import CoreGraphics
import CoreVideo
import QuartzCore
import KopisEngine

public class MetalRenderer {
    private var device: MTLDevice
    private var commandQueue: MTLCommandQueue
    private var library: MTLLibrary
    private var renderPipelineState: MTLRenderPipelineState?
    private var computePipelineState: MTLComputePipelineState?
    private var textureCache: CVMetalTextureCache?
    
    // Buffers for rendering data
    private var vertexBuffer: MTLBuffer?
    private var uniformBuffer: MTLBuffer?
    
    // Texture for raycast output
    private var renderTexture: MTLTexture?
    
    public init?(device: MTLDevice? = nil) {
        // Get Metal device
        guard let metalDevice = device ?? MTLCreateSystemDefaultDevice() else {
            print("⚠ Metal is not supported on this device")
            return nil
        }
        
        self.device = metalDevice
        
        // Create command queue
        guard let queue = metalDevice.makeCommandQueue() else {
            print("⚠ Failed to create Metal command queue")
            return nil
        }
        self.commandQueue = queue
        
        // Create Metal library from source
        // In Swift Package Manager executables, shaders need to be loaded from source
        // Try default library first, but fall back to source if functions aren't found
        var metalLibrary: MTLLibrary?
        
        // Try default library first (with retry for cache locking issues)
        for attempt in 0..<3 {
            if let lib = metalDevice.makeDefaultLibrary(),
               lib.makeFunction(name: "raycast_compute") != nil {
                metalLibrary = lib
                print("✓ Metal library loaded from default library")
                break
            }
            if attempt < 2 {
                Thread.sleep(forTimeInterval: 0.1 * Double(attempt + 1))
            }
        }
        
        // If default library didn't work, load from source
        if metalLibrary == nil {
            let shaderSource = Self.loadShaderSource()
            
            // Retry logic for cache locking issues
            for attempt in 0..<3 {
                do {
                    metalLibrary = try metalDevice.makeLibrary(source: shaderSource, options: nil)
                    print("✓ Metal library created from source (attempt \(attempt + 1))")
                    break
                } catch {
                    let errorDesc = "\(error)"
                    // Check for cache locking issues (errno 35, flock errors)
                    let isCacheLockError = errorDesc.contains("flock") || 
                                          errorDesc.contains("errno = 35") || 
                                          errorDesc.contains("libraries.list") ||
                                          errorDesc.contains("Resource temporarily unavailable")
                    
                    if isCacheLockError && attempt < 2 {
                        // Retry after a short delay for cache locking issues
                        let delay = 0.2 * Double(attempt + 1)
                        print("⚠ Metal library creation failed due to cache lock (attempt \(attempt + 1)), retrying in \(delay)s...")
                        Thread.sleep(forTimeInterval: delay)
                        continue
                    } else if attempt < 2 {
                        // Other errors - retry once
                        let delay = 0.1 * Double(attempt + 1)
                        print("⚠ Metal library creation failed (attempt \(attempt + 1)), retrying in \(delay)s: \(error.localizedDescription)")
                        Thread.sleep(forTimeInterval: delay)
                        continue
                    } else {
                        print("⚠ Failed to create Metal library from source after 3 attempts: \(error)")
                        // Don't fail completely - we can still use CPU rendering
                        return nil
                    }
                }
            }
        }
        
        guard let library = metalLibrary else {
            print("⚠ Failed to create Metal library")
            return nil
        }
        
        self.library = library
        
        // Create texture cache for Core Video
        var textureCache: CVMetalTextureCache?
        let result = CVMetalTextureCacheCreate(
            kCFAllocatorDefault,
            nil,
            metalDevice,
            nil,
            &textureCache
        )
        if result == kCVReturnSuccess {
            self.textureCache = textureCache
        }
        
        // Setup pipelines
        setupRenderPipeline()
        setupComputePipeline()
    }
    
    private func setupRenderPipeline() {
        // Create render pipeline for fullscreen quad
        let vertexFunction = library.makeFunction(name: "vertex_main")
        let fragmentFunction = library.makeFunction(name: "fragment_main")
        
        guard let vertexFunc = vertexFunction, let fragmentFunc = fragmentFunction else {
            print("⚠ Failed to find shader functions")
            return
        }
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunc
        pipelineDescriptor.fragmentFunction = fragmentFunc
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("⚠ Failed to create render pipeline: \(error)")
        }
    }
    
    private func setupComputePipeline() {
        // Create compute pipeline for raycasting
        guard let computeFunction = library.makeFunction(name: "raycast_compute") else {
            print("⚠ Failed to find raycast_compute function in Metal library")
            print("⚠ This is normal if shaders aren't compiled - will use CPU rendering fallback")
            return
        }
        
        // Retry logic for cache locking issues (errno 35)
        for attempt in 0..<3 {
            do {
                computePipelineState = try device.makeComputePipelineState(function: computeFunction)
                print("✓ Metal compute pipeline created successfully (attempt \(attempt + 1))")
                return
            } catch {
                let errorDesc = "\(error)"
                if errorDesc.contains("flock") || errorDesc.contains("errno = 35") || errorDesc.contains("libraries.list") {
                    if attempt < 2 {
                        // Retry after a short delay for cache locking issues
                        let delay = 0.2 * Double(attempt + 1)
                        print("⚠ Compute pipeline creation failed due to cache lock (attempt \(attempt + 1)), retrying in \(delay)s...")
                        Thread.sleep(forTimeInterval: delay)
                        continue
                    } else {
                        print("⚠ Failed to create compute pipeline after 3 attempts (cache lock issue): \(error)")
                        print("⚠ Will use CPU rendering fallback")
                    }
                } else {
                    print("⚠ Failed to create compute pipeline: \(error)")
                    print("⚠ Will use CPU rendering fallback")
                    break
                }
            }
        }
    }
    
    public func createRenderTexture(width: Int, height: Int) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead, .shaderWrite]
        
        let texture = device.makeTexture(descriptor: descriptor)
        renderTexture = texture
        return texture
    }
    
    public func updateDrawableSize(_ size: CGSize) {
        // Recreate render texture if size changed
        let width = Int(size.width)
        let height = Int(size.height)
        
        if renderTexture == nil || 
           renderTexture!.width != width || 
           renderTexture!.height != height {
            _ = createRenderTexture(width: width, height: height)
        }
    }
    
    public func hasComputePipeline() -> Bool {
        return computePipelineState != nil
    }
    
    func renderToDrawable(
        drawable: CAMetalDrawable,
        renderPassDescriptor: MTLRenderPassDescriptor,
        viewModel: GameViewModel,
        bounds: CGRect
    ) {
        guard let engine = viewModel.engine,
              let _ = viewModel.raycaster else {
            return
        }
        
        let width = Int(bounds.width)
        let height = Int(bounds.height)
        
        // Ensure render texture exists
        updateDrawableSize(bounds.size)
        
        guard let renderTexture = renderTexture,
              let computePipeline = computePipelineState else {
            // Fallback: clear to black
            renderPassDescriptor.colorAttachments[0].loadAction = .clear
            renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            return
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        // Step 1: Raycast to render texture using compute shader
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setTexture(renderTexture, index: 0)
        
        // Set uniform data
        var uniforms = RaycastUniforms(
            width: UInt32(width),
            height: UInt32(height),
            cameraX: Float(engine.cameraPos.x),
            cameraY: Float(engine.cameraPos.y),
            cameraZ: Float(engine.cameraPos.z),
            cameraYaw: Float(engine.cameraYaw * .pi / 180.0),
            cameraPitch: Float(engine.cameraPitch * .pi / 180.0),
            fov: Float(GameConstants.fov * .pi / 180.0),
            maxDepth: Float(GameConstants.maxDepth),
            cellSize: Float(GameConstants.cellSize)
        )
        
        // Create uniform buffer if needed
        if uniformBuffer == nil || uniformBuffer!.length < MemoryLayout<RaycastUniforms>.size {
            uniformBuffer = device.makeBuffer(
                length: MemoryLayout<RaycastUniforms>.size,
                options: .storageModeShared
            )
        }
        
        if let uniformBuffer = uniformBuffer {
            memcpy(uniformBuffer.contents(), &uniforms, MemoryLayout<RaycastUniforms>.size)
            computeEncoder.setBuffer(uniformBuffer, offset: 0, index: 0)
        }
        
        // Calculate threadgroup size
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (width + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (height + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()
        
        // Step 2: Blit render texture to drawable
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { return }
        blitEncoder.copy(
            from: renderTexture,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: width, height: height, depth: 1),
            to: drawable.texture,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blitEncoder.endEncoding()
        
        // Present drawable
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    public func render(
        context: CGContext,
        bounds: CGRect,
        engine: KopisEngine,
        raycaster: Raycaster,
        gameOfLife: GameOfLife?,
        cpuRenderFunction: (CGContext, CGRect) -> Void
    ) {
        // Metal infrastructure is set up and ready for GPU acceleration
        // Currently using CPU rendering (via callback) which works reliably
        // with the existing maze collision system
        
        // Use CPU rendering (existing working implementation)
        // Note: wallsWithBlood is managed internally by renderCPU, not passed here
        // to avoid simultaneous access violations
        cpuRenderFunction(context, bounds)
        
        // Future GPU acceleration path:
        // 1. Pass maze chunk data as Metal buffers/textures
        // 2. Implement full DDA raycasting in Metal compute shader
        // 3. Render directly to Metal texture and blit to CGContext
        // 4. This would require refactoring Maze class to expose chunk data
        //    in a GPU-friendly format (e.g., 2D texture or structured buffer)
    }
    
    private func copyTextureToContext(_ texture: MTLTexture, context: CGContext, bounds: CGRect) {
        let width = texture.width
        let height = texture.height
        
        // Create a bitmap context to read the texture data
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bufferSize = bytesPerRow * height
        
        guard let buffer = malloc(bufferSize) else { return }
        defer { free(buffer) }
        
        let region = MTLRegionMake2D(0, 0, width, height)
        texture.getBytes(
            buffer,
            bytesPerRow: bytesPerRow,
            from: region,
            mipmapLevel: 0
        )
        
        // Create CGImage from buffer
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let cgContext = CGContext(
            data: buffer,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return }
        
        guard let cgImage = cgContext.makeImage() else { return }
        
        // Draw to the provided context
        context.saveGState()
        context.translateBy(x: 0, y: bounds.height)
        context.scaleBy(x: 1.0, y: -1.0)
        context.draw(cgImage, in: bounds)
        context.restoreGState()
    }
    
    // Fallback CPU rendering (existing implementation)
    private func renderCPU(
        context: CGContext,
        bounds: CGRect,
        engine: KopisEngine,
        raycaster: Raycaster,
        gameOfLife: GameOfLife?,
        wallsWithBlood: inout [String: Double]
    ) {
        // This will be the existing render method from GameViewModel
        // For now, just fill with black
        context.setFillColor(CGColor.black)
        context.fill(bounds)
    }
    
    private static func loadShaderSource() -> String {
        // Try to load from bundle first
        let bundle = Bundle.main
        if let shaderPath = bundle.path(forResource: "Shaders", ofType: "metal"),
           let shaderSource = try? String(contentsOfFile: shaderPath) {
            return shaderSource
        }
        
        // For Swift Package Manager, try to find in the executable's directory
        // This is a fallback - ideally shaders should be in the bundle
        if let executablePath = bundle.executablePath {
            let executableDir = (executablePath as NSString).deletingLastPathComponent
            let shaderPath = (executableDir as NSString).appendingPathComponent("Shaders.metal")
            
            if let shaderSource = try? String(contentsOfFile: shaderPath) {
                return shaderSource
            }
        }
        
        // Last resort: return embedded shader source
        return embeddedShaderSource
    }
    
    private static var embeddedShaderSource: String {
        return """
        #include <metal_stdlib>
        using namespace metal;
        
        struct RaycastUniforms {
            uint width;
            uint height;
            float cameraX;
            float cameraY;
            float cameraZ;
            float cameraYaw;
            float cameraPitch;
            float fov;
            float maxDepth;
            float cellSize;
        };
        
        kernel void raycast_compute(
            texture2d<float, access::write> output [[texture(0)]],
            constant RaycastUniforms& uniforms [[buffer(0)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
                return;
            }
            
            float width = float(uniforms.width);
            float height = float(uniforms.height);
            
            float cameraX = 2.0 * float(gid.x) / width - 1.0;
            float forwardX = sin(uniforms.cameraYaw);
            float forwardY = -cos(uniforms.cameraYaw);
            float rightX = cos(uniforms.cameraYaw);
            float rightY = sin(uniforms.cameraYaw);
            
            float rayDirX = forwardX + rightX * cameraX * tan(uniforms.fov / 2.0);
            float rayDirY = forwardY + rightY * cameraX * tan(uniforms.fov / 2.0);
            
            // Simplified rendering - just draw ceiling/floor for now
            float horizonY = height / 2.0 + tan(uniforms.cameraPitch) * height / 2.0;
            
            float4 color;
            if (float(gid.y) < horizonY) {
                color = float4(0.2, 0.2, 0.2, 1.0); // Ceiling
            } else {
                color = float4(0.16, 0.16, 0.16, 1.0); // Floor
            }
            
            output.write(color, gid);
        }
        
        struct VertexOut {
            float4 position [[position]];
            float2 texCoord;
        };
        
        vertex VertexOut vertex_main(uint vertexID [[vertex_id]]) {
            VertexOut out;
            out.position = float4(
                (vertexID == 2) ? 3.0 : -1.0,
                (vertexID == 1) ? -3.0 : 1.0,
                0.0,
                1.0
            );
            out.texCoord = float2(
                (vertexID == 2) ? 2.0 : 0.0,
                (vertexID == 1) ? 2.0 : 0.0
            );
            return out;
        }
        
        fragment float4 fragment_main(
            VertexOut in [[stage_in]],
            texture2d<float> texture [[texture(0)]]
        ) {
            constexpr sampler s(min_filter::linear, mag_filter::linear);
            return texture.sample(s, in.texCoord);
        }
        """
    }
}

// Uniform structure for raycasting shader
struct RaycastUniforms {
    var width: UInt32
    var height: UInt32
    var cameraX: Float
    var cameraY: Float
    var cameraZ: Float
    var cameraYaw: Float
    var cameraPitch: Float
    var fov: Float
    var maxDepth: Float
    var cellSize: Float
}
