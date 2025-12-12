import SwiftUI
import KopisEngine

struct ContentView: View {
    @EnvironmentObject var viewModel: GameViewModel
    
    var body: some View {
        ZStack {
            // Game view - using Metal 4 rendering
            MetalGameView()
                .environmentObject(viewModel)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            
            // UI Overlay
            VStack {
                HStack {
                    // Stats
                    VStack(alignment: .leading, spacing: 4) {
                        Text("FPS: \(viewModel.fps, specifier: "%.0f")")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(.white)
                        Text("Entities: \(viewModel.entityCount)")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(.white)
                        if let player = viewModel.player {
                            Text("Health: \(Int(player.health))")
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(.white)
                        }
                    }
                    .padding(8)
                    .background(Color.black.opacity(0.5))
                    .cornerRadius(4)
                    
                    Spacer()
                    
                    // Controls hint
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("WASD: Move")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(.white.opacity(0.7))
                        Text("Space: Jump")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(.white.opacity(0.7))
                        Text("Mouse: Look")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(.white.opacity(0.7))
                        Text("F11: Fullscreen")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(.white.opacity(0.7))
                    }
                    .padding(8)
                    .background(Color.black.opacity(0.5))
                    .cornerRadius(4)
                }
                .padding()
                
                Spacer()
            }
        }
        .background(Color.black)
        .ignoresSafeArea()
    }
}
