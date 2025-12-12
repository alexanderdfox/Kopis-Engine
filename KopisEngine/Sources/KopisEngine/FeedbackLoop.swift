import Foundation

class FeedbackLoop {
    private var stateHistory: [[String: Any]] = []
    private let maxHistory = 100
    private var persistentState: [String: Any] = [:]
    
    func update(signal: Signal, gameState: [String: Any]) -> [String: Any] {
        let historyEntry: [String: Any] = [
            "signal": signal,
            "game_state": gameState,
            "timestamp": Date().timeIntervalSince1970
        ]
        
        stateHistory.append(historyEntry)
        
        if stateHistory.count > maxHistory {
            stateHistory.removeFirst()
        }
        
        let recentSignals = stateHistory.suffix(10).compactMap { $0["signal"] as? Signal }
        let avgVoltage = recentSignals.isEmpty ? 0.0 : 
            recentSignals.map { $0.voltage }.reduce(0, +) / Double(recentSignals.count)
        
        persistentState.updateValue(Date().timeIntervalSince1970, forKey: "last_frame_time")
        persistentState.updateValue(stateHistory.count, forKey: "total_frames")
        persistentState.updateValue(avgVoltage, forKey: "average_voltage")
        
        return persistentState
    }
    
    func getState() -> [String: Any] {
        return persistentState
    }
}
