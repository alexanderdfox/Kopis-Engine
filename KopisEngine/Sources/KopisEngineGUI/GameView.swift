import SwiftUI
import AppKit
import CoreGraphics
import CoreVideo
import KopisEngine

struct GameView: NSViewRepresentable {
    @EnvironmentObject var viewModel: GameViewModel
    
    func makeNSView(context: Context) -> GameNSView {
        let view = GameNSView()
        view.viewModel = viewModel
        return view
    }
    
    func updateNSView(_ nsView: GameNSView, context: Context) {
        // View updates handled by the NSView
    }
}

class GameNSView: NSView {
    var viewModel: GameViewModel?
    private var displayLink: CVDisplayLink?
    private var trackingArea: NSTrackingArea?
    private var mouseCaptured = false
    private var previousMouseLocation: NSPoint = .zero
    
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        
        if window != nil {
            setupInputHandling()
            // Delay mouse capture to avoid crashes on startup
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.setupDisplayLink()
                self.captureMouse()
            }
        } else {
            stopDisplayLink()
            releaseMouse()
        }
    }
    
    private func setupDisplayLink() {
        var displayLink: CVDisplayLink?
        let callback: CVDisplayLinkOutputCallback = { (displayLink, inNow, inOutputTime, flagsIn, flagsOut, context) -> CVReturn in
            guard let context = context else { return kCVReturnError }
            let view = Unmanaged<GameNSView>.fromOpaque(context).takeUnretainedValue()
            DispatchQueue.main.async {
                view.renderFrame()
            }
            return kCVReturnSuccess
        }
        
        let result = CVDisplayLinkCreateWithActiveCGDisplays(&displayLink)
        guard result == kCVReturnSuccess, let link = displayLink else {
            print("⚠ Failed to create display link")
            return
        }
        
        let callbackResult = CVDisplayLinkSetOutputCallback(link, callback, Unmanaged.passUnretained(self).toOpaque())
        guard callbackResult == kCVReturnSuccess else {
            print("⚠ Failed to set display link callback")
            return
        }
        
        let startResult = CVDisplayLinkStart(link)
        guard startResult == kCVReturnSuccess else {
            print("⚠ Failed to start display link")
            return
        }
        
        self.displayLink = link
    }
    
    private func stopDisplayLink() {
        if let displayLink = displayLink {
            CVDisplayLinkStop(displayLink)
            self.displayLink = nil
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
        // Note: CGAssociateMouseAndMouseCursorPosition requires accessibility permissions
        // Using a safer approach that doesn't require special permissions
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
            
            // Track relative movement (safer approach without CGWarpMouseCursorPosition)
            // CGWarpMouseCursorPosition requires accessibility permissions and can cause crashes
            // Instead, we just use the relative movement from center
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
    
    private func renderFrame() {
        viewModel?.updateFrame()
        needsDisplay = true
    }
    
    override func draw(_ dirtyRect: NSRect) {
        guard let context = NSGraphicsContext.current?.cgContext else { return }
        
        // Clear background
        context.setFillColor(CGColor.black)
        context.fill(bounds)
        
        // Render game (basic implementation - can be enhanced with Metal)
        // dirtyRect is provided by NSView but we render the full bounds
        _ = dirtyRect
        viewModel?.render(context: context, bounds: bounds)
    }
    
    deinit {
        stopDisplayLink()
    }
}
