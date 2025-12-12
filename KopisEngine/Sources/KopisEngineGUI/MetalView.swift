import SwiftUI
import MetalKit
import Metal
import AppKit
import KopisEngine

struct MetalGameView: NSViewRepresentable {
    @EnvironmentObject var viewModel: GameViewModel
    
    func makeNSView(context: Context) -> MTKGameView {
        // Create view with minimal frame to avoid layout issues
        let frame = CGRect(x: 0, y: 0, width: 800, height: 600)
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("⚠ Metal device not available - GUI will not render")
            // Return a basic view anyway so the app doesn't crash
            let view = MTKGameView(frame: frame, device: nil)
            return view
        }
        
        let mtkView = MTKGameView(frame: frame, device: device)
        mtkView.viewModel = viewModel
        print("✓ MetalGameView created with device: \(device.name)")
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKGameView, context: Context) {
        // Update view model reference if needed
        if nsView.viewModel !== viewModel {
            nsView.viewModel = viewModel
            print("✓ MetalGameView: viewModel updated")
        }
        // Force the view to render
        nsView.needsDisplay = true
        // Ensure view can receive keyboard input
        DispatchQueue.main.async {
            nsView.window?.makeFirstResponder(nsView)
        }
    }
}

class MTKGameView: MTKView {
    var viewModel: GameViewModel?
    private var metalRenderer: MetalRenderer?
    private var trackingArea: NSTrackingArea?
    private var mouseCaptured = false
    private var previousMouseLocation: NSPoint = .zero
    private var isConfigured = false
    private var frameCount = 0
    private var lastLogTime: Date?
    
    override init(frame frameRect: CGRect, device: MTLDevice?) {
        let metalDevice = device ?? MTLCreateSystemDefaultDevice()
        guard metalDevice != nil else {
            print("⚠ Metal is not supported on this device")
            super.init(frame: frameRect, device: nil)
            return
        }
        super.init(frame: frameRect, device: metalDevice)
        // Don't configure here - wait until view is in window
    }
    
    required init(coder: NSCoder) {
        super.init(coder: coder)
        if device == nil {
            device = MTLCreateSystemDefaultDevice()
        }
        // Don't configure here - wait until view is in window
    }
    
    convenience init() {
        self.init(frame: .zero, device: nil)
    }
    
    private func setupMetalViewSafely() {
        guard !isConfigured else { return }
        guard let device = device else {
            print("⚠ Metal device not available")
            return
        }
        
        print("✓ Metal device initialized: \(device.name)")
        
        // Configure MTKView properties - only set essential ones
        colorPixelFormat = .bgra8Unorm
        clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        framebufferOnly = false
        enableSetNeedsDisplay = false
        isPaused = false
        preferredFramesPerSecond = 60
        // Ensure the view renders continuously
        needsDisplay = true
        
        // Create Metal renderer
        metalRenderer = MetalRenderer(device: device)
        if metalRenderer == nil {
            print("⚠ Failed to create Metal renderer")
        } else {
            print("✓ Metal renderer created")
        }
        
        // Set delegate for rendering
        delegate = self
        print("✓ MTKView delegate set")
        
        // Ensure view renders
        needsDisplay = true
        
        isConfigured = true
        print("✓ Metal view configuration complete")
    }
    
    
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        
        // Make this view first responder when window appears to receive keyboard input
        DispatchQueue.main.async { [weak self] in
            self?.window?.makeFirstResponder(self)
        }
        
        guard window != nil else {
            releaseMouse()
            return
        }
        
        // Configure view only after it's in the window
        if !isConfigured {
            setupMetalViewSafely()
        }
        
        // Setup input handling after view is in window
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.setupInputHandling()
            
            // Delay mouse capture to avoid crashes on startup
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.captureMouse()
            }
        }
    }
    
    private func setupInputHandling() {
        // Make view first responder to receive key events
        window?.makeFirstResponder(self)
        
        // Track mouse movement
        updateTrackingArea()
    }
    
    private func updateTrackingArea() {
        if let oldArea = trackingArea {
            removeTrackingArea(oldArea)
        }
        
        let options: NSTrackingArea.Options = [.activeInKeyWindow, .mouseMoved, .inVisibleRect]
        
        let newArea = NSTrackingArea(
            rect: bounds,
            options: options,
            owner: self,
            userInfo: nil
        )
        addTrackingArea(newArea)
        trackingArea = newArea
    }
    
    private func captureMouse() {
        guard !mouseCaptured else { return }
        mouseCaptured = true
        NSCursor.hide()
        previousMouseLocation = NSEvent.mouseLocation
        updateTrackingArea()
    }
    
    private func releaseMouse() {
        guard mouseCaptured else { return }
        mouseCaptured = false
        NSCursor.unhide()
        updateTrackingArea()
    }
    
    override var acceptsFirstResponder: Bool { true }
    
    override func keyDown(with event: NSEvent) {
        // Handle ESC to toggle mouse capture
        if event.keyCode == 53 { // ESC key
            if mouseCaptured {
                releaseMouse()
            } else {
                captureMouse()
            }
            return
        }
        
        viewModel?.handleKeyEvent(event, isDown: true)
    }
    
    override func keyUp(with event: NSEvent) {
        viewModel?.handleKeyEvent(event, isDown: false)
    }
    
    override func flagsChanged(with event: NSEvent) {
        // Handle modifier keys if needed
        super.flagsChanged(with: event)
    }
    
    override func mouseMoved(with event: NSEvent) {
        if mouseCaptured {
            // Calculate delta from center when mouse is captured
            let center = CGPoint(x: bounds.midX, y: bounds.midY)
            let currentLocation = convert(event.locationInWindow, from: nil)
            let dx = Float(currentLocation.x - center.x)
            let dy = Float(currentLocation.y - center.y)
            
            viewModel?.handleMouseMovement(dx: dx * 0.1, dy: dy * 0.1) // Sensitivity
        } else {
            // Use relative movement when not captured
            viewModel?.handleMouseMovement(dx: Float(event.deltaX), dy: Float(event.deltaY))
        }
    }
    
    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        if !mouseCaptured {
            captureMouse()
        }
    }
    
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        updateTrackingArea()
    }
}

extension MTKGameView: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle resize safely
        guard size.width > 0 && size.height > 0 else { return }
        // view parameter is required by protocol but not used here
        _ = view
        metalRenderer?.updateDrawableSize(size)
    }
    
    func draw(in view: MTKView) {
        // Ensure we're on main thread
        guard Thread.isMainThread else {
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.draw(in: view)
            }
            return
        }
        
        // Safety checks
        guard view.bounds.width > 0 && view.bounds.height > 0 else {
            return
        }
        
        // Debug: Log that draw is being called (throttled)
        frameCount += 1
        if frameCount == 1 || frameCount % 60 == 0 {
            print("✓ MTKView draw() called (frame \(frameCount), size: \(view.bounds.width)x\(view.bounds.height))")
        }
        
        guard let viewModel = viewModel else {
            // Clear to black if no view model
            if frameCount == 1 || frameCount % 60 == 0 {
                print("⚠ MTKView draw(): viewModel is nil")
            }
            clearToBlack(view: view)
            return
        }
        
        // Check if game is initialized
        guard let _ = viewModel.engine, let _ = viewModel.raycaster else {
            #if DEBUG
            if frameCount == 1 || frameCount % 60 == 0 {
                print("⚠ MTKView draw(): engine or raycaster not initialized")
                if viewModel.engine == nil { print("  - engine is nil") }
                if viewModel.raycaster == nil { print("  - raycaster is nil") }
            }
            #endif
            clearToBlack(view: view)
            return
        }
        
        // Update game state
        viewModel.updateFrame()
        
        // Try Metal GPU rendering first (if compute pipeline is available)
        // TEMPORARILY DISABLED: Force CPU rendering to ensure walls and Game of Life work
        // TODO: Fix Metal compute shader to render walls properly
        let forceCPURendering = true
        
        if !forceCPURendering, let renderer = metalRenderer,
           renderer.hasComputePipeline(),
           let drawable = view.currentDrawable,
           let renderPassDescriptor = view.currentRenderPassDescriptor {
            // Use Metal compute shader path
            if frameCount == 1 || frameCount % 60 == 0 {
                print("⚠ Using Metal GPU rendering (walls/GameOfLife may not work yet)")
            }
            renderer.renderToDrawable(
                drawable: drawable,
                renderPassDescriptor: renderPassDescriptor,
                viewModel: viewModel,
                bounds: view.bounds
            )
            return
        }
        
        // Debug: Log which rendering path we're using
        if forceCPURendering {
            if frameCount == 1 || frameCount % 60 == 0 {
                print("✓ Using CPU rendering (walls and GameOfLife enabled)")
            }
        } else if metalRenderer == nil {
            print("⚠ Metal renderer not available, using CPU fallback")
        } else if !metalRenderer!.hasComputePipeline() {
            print("⚠ Metal compute pipeline not available, using CPU fallback")
        }
        
        // Fallback: Use CPU rendering and render to a bitmap, then copy to Metal
        // This is more reliable when Metal compute shaders aren't available
        guard let drawable = view.currentDrawable,
              let _ = view.currentRenderPassDescriptor,
              let device = view.device,
              let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            clearToBlack(view: view)
            return
        }
        
        // Create a bitmap context for CPU rendering
        let width = Int(view.bounds.width)
        let height = Int(view.bounds.height)
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            clearToBlack(view: view)
            return
        }
        
        // Render using CPU path (which works reliably)
        // Save the context state before rendering
        context.saveGState()
        defer { context.restoreGState() }
        
        // Flip the coordinate system - CGContext has origin at bottom-left, we want top-left
        // This is necessary because CGContext uses a flipped coordinate system
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1.0, y: -1.0)
        
        // Clear to black first (in flipped coordinates, this is at the "bottom" which is actually top)
        context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        
        // Render the game scene (all coordinates will be automatically flipped)
        viewModel.render(context: context, bounds: view.bounds)
        
        // Get the rendered image
        guard let cgImage = context.makeImage() else {
            print("⚠ Failed to create CGImage from CPU render context")
            clearToBlack(view: view)
            return
        }
        
        // Only log once per second to avoid spam
        if frameCount == 1 || frameCount % 60 == 0 {
            print("✓ CPU rendering completed, copying to Metal texture (size: \(width)x\(height))")
        }
        
        // Create a Metal texture from the CGImage and copy to drawable
        let textureLoader = MTKTextureLoader(device: device)
        do {
            let texture = try textureLoader.newTexture(cgImage: cgImage, options: nil)
            
            // Blit the texture to the drawable
            if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
                blitEncoder.copy(
                    from: texture,
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
            }
            
            commandBuffer.present(drawable)
            commandBuffer.commit()
            // Only log once per second to avoid spam
            let now = Date()
            if lastLogTime == nil || now.timeIntervalSince(lastLogTime!) > 1.0 {
                print("✓ CPU rendering to Metal texture successful")
                lastLogTime = now
            }
        } catch {
            print("⚠ Failed to create texture from CPU render: \(error)")
            clearToBlack(view: view)
        }
    }
    
    private func clearToBlack(view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let device = view.device,
              let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            renderEncoder.endEncoding()
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
