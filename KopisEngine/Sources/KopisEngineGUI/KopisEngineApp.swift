import SwiftUI
import AppKit
import KopisEngine

@main
@available(macOS 13.0, *)
struct KopisEngineApp: App {
    @StateObject private var gameViewModel = GameViewModel()
    @State private var warningAcknowledged = false
    
    var body: some Scene {
        WindowGroup {
            Group {
                if warningAcknowledged {
                    ContentView()
                        .environmentObject(gameViewModel)
                        .frame(minWidth: 800, minHeight: 600)
                        .background(Color.black)
                        .id("gameView") // Force view refresh
                        .onAppear {
                            print("🚀 Starting Kopis Engine GUI...")
                            print("✓ ContentView appeared - warningAcknowledged = \(warningAcknowledged)")
                            gameViewModel.startGame()
                            // Ensure window is visible and activated
                            DispatchQueue.main.async {
                                NSApplication.shared.activate(ignoringOtherApps: true)
                                if let window = NSApplication.shared.windows.first {
                                    window.makeKeyAndOrderFront(nil)
                                    window.center()
                                    print("✓ Window activated and centered")
                                    // Start in fullscreen like Python version (optional)
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                        window.toggleFullScreen(nil)
                                        print("📺 Toggling fullscreen...")
                                    }
                                } else {
                                    print("⚠ No window found")
                                }
                            }
                        }
                        .transition(.opacity)
                } else {
                    EpilepsyWarningView(acknowledged: $warningAcknowledged)
                        .frame(minWidth: 800, minHeight: 600)
                        .id("warningView") // Force view refresh
                        .onAppear {
                            print("⚠ Warning view appeared - warningAcknowledged = \(warningAcknowledged)")
                            // Ensure warning window is visible
                            DispatchQueue.main.async {
                                NSApplication.shared.activate(ignoringOtherApps: true)
                                // Try multiple times to find window
                                var attempts = 0
                                func showWindow() {
                                    if let window = NSApplication.shared.windows.first {
                                        window.makeKeyAndOrderFront(nil)
                                        window.center()
                                        window.orderFrontRegardless()
                                        print("✓ Warning window activated and ordered front")
                                    } else if attempts < 10 {
                                        attempts += 1
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                                            showWindow()
                                        }
                                    } else {
                                        print("⚠ Failed to find window after 10 attempts")
                                    }
                                }
                                showWindow()
                            }
                        }
                        .onChange(of: warningAcknowledged) { newValue in
                            print("⚠ Warning view - onChange triggered: warningAcknowledged = \(newValue)")
                        }
                }
            }
            .id(warningAcknowledged ? "acknowledged" : "notAcknowledged") // Force complete view refresh on state change
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1280, height: 720)
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
    
    init() {
        print("🚀 KopisEngineApp initializing...")
        print("⚠ IMPORTANT: Make sure you're running the 'KopisEngineGUI' scheme, not 'KopisEngineApp'")
        print("⚠ If no window appears, check the scheme selector in Xcode's top toolbar")
        
        // Ensure AppKit is properly initialized
        if NSApplication.shared.activationPolicy() == .accessory {
            NSApplication.shared.setActivationPolicy(.regular)
            print("✓ Changed activation policy from accessory to regular")
        }
        
        // Set up app delegate to ensure window appears
        if NSApplication.shared.delegate == nil {
            let appDelegate = AppDelegate()
            NSApplication.shared.delegate = appDelegate
            print("✓ App delegate set")
        }
        
        // Activate the app to bring window to front
        NSApplication.shared.activate(ignoringOtherApps: true)
        print("✓ App activated")
        
        // Force window to appear after a short delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            NSApplication.shared.activate(ignoringOtherApps: true)
            for window in NSApplication.shared.windows {
                window.makeKeyAndOrderFront(nil)
                window.orderFrontRegardless()
                window.setIsVisible(true)
                print("✓ Forced window to appear: \(window.title)")
            }
            if NSApplication.shared.windows.isEmpty {
                print("⚠⚠⚠ NO WINDOWS FOUND!")
                print("⚠ This usually means:")
                print("  1. Wrong scheme selected (should be 'KopisEngineGUI')")
                print("  2. App is running as command-line tool")
                print("  3. WindowGroup isn't creating windows")
            }
        }
    }
}

// App delegate to ensure window appears
class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        print("✓ applicationDidFinishLaunching called")
        NSApplication.shared.activate(ignoringOtherApps: true)
        
        // Ensure window appears - try multiple times
        var attempts = 0
        func showWindow() {
            if let window = NSApplication.shared.windows.first {
                window.makeKeyAndOrderFront(nil)
                window.center()
                window.orderFrontRegardless()
                window.setIsVisible(true)
                print("✓ Window made key, ordered front, and set visible")
            } else if attempts < 20 {
                attempts += 1
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    showWindow()
                }
            } else {
                print("⚠ No window found in applicationDidFinishLaunching after 20 attempts")
                print("⚠ Window count: \(NSApplication.shared.windows.count)")
            }
        }
        DispatchQueue.main.async {
            showWindow()
        }
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
    
    func applicationWillFinishLaunching(_ notification: Notification) {
        print("✓ applicationWillFinishLaunching called")
    }
}
