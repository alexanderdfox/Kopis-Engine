import Foundation
import AVFoundation

public class SoundManager {
    public var enabled: Bool = false
    private var audioEngine: AVAudioEngine?
    private var sounds: [String: AVAudioPCMBuffer] = [:]
    
    public init() {
        setupAudioEngine()
        // Generate sounds after a short delay to ensure audio engine is ready
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            self?.generateSounds()
        }
    }
    
    private func setupAudioEngine() {
        // AVAudioEngine setup must be on main thread
        if Thread.isMainThread {
            setupAudioEngineInternal()
        } else {
            DispatchQueue.main.sync {
                setupAudioEngineInternal()
            }
        }
    }
    
    private func setupAudioEngineInternal() {
        do {
            let audioEngine = AVAudioEngine()
            
            // The mainMixerNode is automatically part of the engine and connected to outputNode
            // We don't need to attach or connect it manually
            
            // Prepare the engine (this ensures all nodes are ready)
            audioEngine.prepare()
            
            // Start the audio engine
            try audioEngine.start()
            
            self.audioEngine = audioEngine
            self.enabled = true
            print("✓ Sound system initialized")
        } catch {
            print("⚠ Sound system unavailable: \(error)")
            self.enabled = false
        }
    }
    
    private func generateSounds() {
        guard enabled else { return }
        
        let sampleRate: Double = 22050.0
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        
        // Footstep sound (400 Hz, 0.1s)
        if let footstep = generateTone(frequency: 400.0, duration: 0.1, volume: 0.3, sampleRate: sampleRate, format: format) {
            sounds["footstep"] = footstep
        }
        
        // Collision sound (600 Hz, 0.15s)
        if let collision = generateTone(frequency: 600.0, duration: 0.15, volume: 0.4, sampleRate: sampleRate, format: format) {
            sounds["collision"] = collision
        }
        
        // Movement start sound (200 Hz, 0.05s)
        if let moveStart = generateTone(frequency: 200.0, duration: 0.05, volume: 0.2, sampleRate: sampleRate, format: format) {
            sounds["move_start"] = moveStart
        }
    }
    
    private func generateTone(frequency: Double, duration: Double, volume: Double, sampleRate: Double, format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(sampleRate * duration)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return nil
        }
        
        buffer.frameLength = frameCount
        
        guard let channelData = buffer.floatChannelData else {
            return nil
        }
        
        let maxSample: Float = 1.0
        let volumeFloat = Float(volume)
        
        let twoPi = 2.0 * Double.pi
        for frame in 0..<Int(frameCount) {
            let time = Double(frame) / sampleRate
            let phase = time * frequency * twoPi
            let sineValue = sin(phase)
            let sample = Float(sineValue) * volumeFloat
            
            // Stereo (2 channels)
            channelData[0][frame] = sample * maxSample
            channelData[1][frame] = sample * maxSample
        }
        
        return buffer
    }
    
    public func play(_ soundName: String, volume: Float = 0.5) {
        guard enabled,
              let audioEngine = audioEngine,
              let buffer = sounds[soundName] else {
            return
        }
        
        // Ensure we're on main thread for audio operations
        if !Thread.isMainThread {
            DispatchQueue.main.async { [weak self] in
                self?.play(soundName, volume: volume)
            }
            return
        }
        
        // Create a new player node for each sound (allows overlapping sounds)
        let playerNode = AVAudioPlayerNode()
        audioEngine.attach(playerNode)
        
        // Connect player directly to main mixer
        // Use the buffer's format to match the sound data
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: buffer.format)
        
        // Set volume on the player node
        playerNode.volume = volume
        
        // Play sound
        playerNode.scheduleBuffer(buffer, at: nil, options: [], completionHandler: {
            // Clean up after playback
            DispatchQueue.main.async {
                playerNode.stop()
                audioEngine.detach(playerNode)
            }
        })
        
        if !playerNode.isPlaying {
            playerNode.play()
        }
    }
    
    public func cleanup() {
        audioEngine?.stop()
        audioEngine = nil
        enabled = false
    }
    
    deinit {
        cleanup()
    }
}
