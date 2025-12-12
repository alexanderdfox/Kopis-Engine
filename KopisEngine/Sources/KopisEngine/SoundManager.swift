import Foundation
import AVFoundation

public class SoundManager {
    public var enabled: Bool = false
    private var audioEngine: AVAudioEngine?
    private var sounds: [String: AVAudioPCMBuffer] = [:]
    private var engineStarted: Bool = false
    private let setupQueue = DispatchQueue(label: "com.kopisengine.soundmanager")
    
    public init() {
        // Don't initialize audio engine immediately - use lazy initialization
        // Generate sounds without starting the engine
        generateSounds()
    }
    
    private func ensureAudioEngineStarted() -> Bool {
        // Ensure we're on main thread for audio operations
        if !Thread.isMainThread {
            var result = false
            DispatchQueue.main.sync {
                result = ensureAudioEngineStarted()
            }
            return result
        }
        
        // If already started, return success
        if engineStarted, let engine = audioEngine, engine.isRunning {
            return true
        }
        
        // Create engine if needed
        if audioEngine == nil {
            audioEngine = AVAudioEngine()
        }
        
        guard let audioEngine = audioEngine else {
            return false
        }
        
        // Don't start the engine until we have at least one node attached
        // The engine will be started when we play the first sound
        if !engineStarted {
            // Prepare the engine (this ensures all nodes are ready)
            // Note: prepare() doesn't throw, so no try-catch needed
            audioEngine.prepare()
            self.enabled = true
        }
        
        return true
    }
    
    private func generateSounds() {
        // Generate sounds without requiring the engine to be started
        let sampleRate: Double = 22050.0
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2) else {
            print("⚠ Sound system: Failed to create audio format")
            return
        }
        
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
        
        if !sounds.isEmpty {
            print("✓ Sound buffers generated (\(sounds.count) sounds)")
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
        guard let buffer = sounds[soundName] else {
            return
        }
        
        // Ensure we're on main thread for audio operations
        if !Thread.isMainThread {
            DispatchQueue.main.async { [weak self] in
                self?.play(soundName, volume: volume)
            }
            return
        }
        
        // Ensure audio engine is prepared (but don't start until we have a node)
        guard ensureAudioEngineStarted(), let audioEngine = audioEngine else {
            return
        }
        
        // Create a new player node for each sound (allows overlapping sounds)
        let playerNode = AVAudioPlayerNode()
        audioEngine.attach(playerNode)
        
        // Connect player directly to main mixer
        // Use the buffer's format to match the sound data
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: buffer.format)
        
        // Start the engine if not already started (now we have a node attached)
        if !engineStarted {
            do {
                try audioEngine.start()
                engineStarted = true
                print("✓ Sound engine started")
            } catch {
                print("⚠ Failed to start sound engine: \(error)")
                audioEngine.detach(playerNode)
                return
            }
        }
        
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
        if let engine = audioEngine {
            if engine.isRunning {
                engine.stop()
            }
            audioEngine = nil
        }
        engineStarted = false
        enabled = false
    }
    
    deinit {
        cleanup()
    }
}
