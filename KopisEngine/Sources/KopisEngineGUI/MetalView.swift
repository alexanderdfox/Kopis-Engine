import SwiftUI
import MetalKit
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
        }
    }
}

class MTKGameView: MTKView {
    var viewModel: GameViewModel?
    private var metalRenderer: MetalRenderer?
    private var trackingArea: NSTrackingArea?
    private var mouseCaptured = false
    
    private var isConfigured = false
    
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
        
        // Create Metal renderer
        metalRenderer = MetalRenderer(device: device)
        if metalRenderer == nil {
            print("⚠ Failed to create Metal renderer")
        } else {
            print("✓ Metal renderer created")
        }
        
        // Set delegate for rendering
        delegate = self
        
        isConfigured = true
    }
    
    
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        
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
    
    private var previousMouseLocation: NSPoint = .zero
    
    override var acceptsFirstResponder: Bool { true }
    
    override func keyDown(with event: NSEvent) {
        // Handle ESC to toggle mouse capture
        if event.keyCode == 53 { // ESC key
            if mouseCaptured {
                releaseMouse()
            } else {
                captureMouse()
            }
        }
        
        viewModel?.handleKeyEvent(event, isDown: true)
    }
    
    override func keyUp(with event: NSEvent) {
        viewModel?.handleKeyEvent(event, isDown: false)
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
        
        guard let viewModel = viewModel else {
            // Clear to black if no view model
            clearToBlack(view: view)
            return
        }
        
        guard let renderer = metalRenderer,
              let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor else {
            return
        }
        
        // Update game state
        viewModel.updateFrame()
        
        // Render with Metal
        renderer.renderToDrawable(
            drawable: drawable,
            renderPassDescriptor: renderPassDescriptor,
            viewModel: viewModel,
            bounds: view.bounds
        )
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
