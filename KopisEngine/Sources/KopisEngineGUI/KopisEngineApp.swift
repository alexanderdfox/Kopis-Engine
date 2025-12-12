import SwiftUI
import AppKit
import KopisEngine

@main
struct KopisEngineApp: App {
    @StateObject private var gameViewModel = GameViewModel()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(gameViewModel)
                .frame(minWidth: 800, minHeight: 600)
                .background(Color.black)
                .onAppear {
                    print("🚀 Starting Kopis Engine GUI...")
                    gameViewModel.startGame()
                    // Start in fullscreen like Python version
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        if let window = NSApplication.shared.windows.first {
                            print("📺 Toggling fullscreen...")
                            window.toggleFullScreen(nil)
                        } else {
                            print("⚠ No window found")
                        }
                    }
                }
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1280, height: 720)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
    
    init() {
        // Ensure AppKit is properly initialized
        if NSApplication.shared.activationPolicy() == .accessory {
            NSApplication.shared.setActivationPolicy(.regular)
        }
    }
}
