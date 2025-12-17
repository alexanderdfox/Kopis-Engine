import SwiftUI

struct ContentView: View {
    @State private var showEngine = false
    
    var body: some View {
        NavigationView {
            if showEngine {
                PythonEngineView()
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .navigationBarLeading) {
                            Button("Back") {
                                showEngine = false
                            }
                        }
                    }
            } else {
                VStack(spacing: 20) {
                    Text("Kopis Engine")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .padding()
                    
                    Text("Python-based Game Engine")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Divider()
                    
                    VStack(alignment: .leading, spacing: 10) {
                        Text("iOS Python Engine")
                            .font(.headline)
                        
                        Text("Run the Python game engine on iOS using PythonKit.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.top, 5)
                        
                        Text("Note: Requires Python runtime to be embedded in the app bundle.")
                            .font(.caption)
                            .foregroundColor(.orange)
                            .padding(.top, 5)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                    
                    Spacer()
                    
                    Button(action: {
                        showEngine = true
                    }) {
                        Text("Start Engine")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                    .padding(.horizontal)
                    
                    Spacer()
                }
                .padding()
                .navigationTitle("Kopis Engine")
            }
        }
    }
}

#Preview {
    ContentView()
}
