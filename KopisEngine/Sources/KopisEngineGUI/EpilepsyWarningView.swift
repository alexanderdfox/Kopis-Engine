import SwiftUI
import AppKit

struct EpilepsyWarningView: View {
    @Binding var acknowledged: Bool
    
    var body: some View {
        ZStack {
            // Dark background
            Color(red: 0.1, green: 0.1, blue: 0.18)
                .ignoresSafeArea()
            
            VStack(spacing: 30) {
                // Warning icon
                Text("⚠️")
                    .font(.system(size: 80))
                    .foregroundColor(.red)
                
                // Title
                Text("EPILEPSY WARNING")
                    .font(.system(size: 60, weight: .bold, design: .rounded))
                    .foregroundColor(.red)
                    .multilineTextAlignment(.center)
                
                // Warning message
                VStack(spacing: 20) {
                    Text("This application contains flashing lights, rapid visual changes,")
                        .font(.system(size: 24))
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                    
                    Text("and intense visual effects that may trigger seizures in people")
                        .font(.system(size: 24))
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                    
                    Text("with photosensitive epilepsy or other photosensitive conditions.")
                        .font(.system(size: 24))
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                    
                    Text("")
                        .frame(height: 20)
                    
                    Text("If you have a history of epilepsy or are sensitive to flashing")
                        .font(.system(size: 24))
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                    
                    Text("lights, please use caution or avoid using this application.")
                        .font(.system(size: 24))
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                }
                .padding(.horizontal, 40)
                
                Spacer()
                
                // Continue button
                Button(action: {
                    print("✓ Warning acknowledged - transitioning to game view...")
                    acknowledged = true
                    print("✓ acknowledged set to true (current value: \(acknowledged))")
                }) {
                    Text("I Understand - Continue")
                        .font(.system(size: 28, weight: .bold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 40)
                        .padding(.vertical, 20)
                        .background(
                            LinearGradient(
                                gradient: Gradient(colors: [Color(red: 1.0, green: 0.27, blue: 0.27), Color(red: 0.8, green: 0.0, blue: 0.0)]),
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .cornerRadius(12)
                        .shadow(color: Color.red.opacity(0.7), radius: 20, x: 0, y: 10)
                }
                .buttonStyle(PlainButtonStyle())
                .padding(.bottom, 50)
            }
            .padding(50)
            .overlay(
                // Red border
                RoundedRectangle(cornerRadius: 20)
                    .stroke(Color.red, lineWidth: 5)
                    .padding(50)
            )
        }
        .onTapGesture {
            print("✓ Warning acknowledged via tap - transitioning to game view...")
            acknowledged = true
            print("✓ acknowledged set to true via tap (current value: \(acknowledged))")
        }
        .background(KeyboardHandlerView(acknowledged: $acknowledged))
    }
}

// Helper view to handle keyboard events
struct KeyboardHandlerView: NSViewRepresentable {
    @Binding var acknowledged: Bool
    
    func makeNSView(context: Context) -> NSView {
        let view = KeyHandlingView()
        // Capture self to access the binding
        view.onKeyPress = { [self] in
            print("✓ Warning acknowledged via keyboard - transitioning to game view...")
            self.acknowledged = true
            print("✓ acknowledged set to true via keyboard (current value: \(self.acknowledged))")
        }
        return view
    }
    
    func updateNSView(_ nsView: NSView, context: Context) {
        // Update the closure if needed
        if let keyView = nsView as? KeyHandlingView {
            // Capture self to access the binding
            keyView.onKeyPress = { [self] in
                print("✓ Warning acknowledged via keyboard (update) - transitioning to game view...")
                self.acknowledged = true
                print("✓ acknowledged set to true via keyboard (update) (current value: \(self.acknowledged))")
            }
        }
    }
}

class KeyHandlingView: NSView {
    var onKeyPress: (() -> Void)?
    private var eventMonitor: Any?
    
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        
        if window != nil {
            // Monitor key events
            eventMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self = self else { return event }
                
                // Check for Enter (36), Space (49), or Escape (53)
                if event.keyCode == 36 || event.keyCode == 49 || event.keyCode == 53 {
                    DispatchQueue.main.async {
                        self.onKeyPress?()
                    }
                    return nil // Consume the event
                }
                return event
            }
        } else {
            if let monitor = eventMonitor {
                NSEvent.removeMonitor(monitor)
                eventMonitor = nil
            }
        }
    }
    
    deinit {
        if let monitor = eventMonitor {
            NSEvent.removeMonitor(monitor)
        }
    }
}
