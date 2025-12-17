import SwiftUI

/// SwiftUI view that integrates with Python engine
struct PythonEngineView: View {
    @StateObject private var engineManager = PythonEngineManager()
    @State private var showError = false
    
    var body: some View {
        ZStack {
            // Game rendering area (would display Python engine output)
            Color.black
                .ignoresSafeArea()
            
            VStack {
                // Status overlay
                if engineManager.isRunning {
                    VStack {
                        Text("Kopis Engine - Running")
                            .foregroundColor(.green)
                            .font(.headline)
                        
                        Text(engineManager.statusMessage)
                            .foregroundColor(.white)
                            .font(.caption)
                            .padding(.top, 5)
                    }
                    .padding()
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(10)
                    .padding()
                }
                
                Spacer()
                
                // Controls
                if !engineManager.isRunning {
                    VStack(spacing: 15) {
                        Text("Python Engine")
                            .font(.title2)
                            .foregroundColor(.white)
                        
                        if let error = engineManager.errorMessage {
                            Text(error)
                                .foregroundColor(.red)
                                .font(.caption)
                                .multilineTextAlignment(.center)
                                .padding()
                        }
                        
                        Button(action: {
                            engineManager.startEngine()
                        }) {
                            Text("Start Engine")
                                .font(.headline)
                                .foregroundColor(.white)
                                .frame(maxWidth: 200)
                                .padding()
                                .background(Color.blue)
                                .cornerRadius(10)
                        }
                    }
                    .padding()
                    .background(Color.black.opacity(0.8))
                    .cornerRadius(15)
                } else {
                    Button(action: {
                        engineManager.stopEngine()
                    }) {
                        Text("Stop Engine")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: 200)
                            .padding()
                            .background(Color.red)
                            .cornerRadius(10)
                    }
                    .padding()
                }
            }
        }
        .alert("Error", isPresented: $showError) {
            Button("OK") { }
        } message: {
            if let error = engineManager.errorMessage {
                Text(error)
            }
        }
        .onChange(of: engineManager.errorMessage) { error in
            showError = error != nil
        }
    }
}

#Preview {
    PythonEngineView()
}
