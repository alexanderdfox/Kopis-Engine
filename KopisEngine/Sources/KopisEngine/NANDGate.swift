import Foundation

class NANDGate {
    private var conditions: [String] = []
    private var rules: [String: (Signal, [String: Any]) -> Bool] = [:]
    
    func addRule(name: String, conditionFunc: @escaping (Signal, [String: Any]) -> Bool) {
        rules[name] = conditionFunc
    }
    
    func nandOperation(inputA: Bool, inputB: Bool) -> Bool {
        return !(inputA && inputB)
    }
    
    func evaluate(signals: [String: Signal], gameState: [String: Any]) -> Signal {
        let physicsActive = signals["physics"]?.voltage ?? 0 > 1.0
        let renderingActive = signals["rendering"]?.voltage ?? 0 > 1.0
        let aiActive = signals["ai"]?.voltage ?? 0 > 1.0
        
        let gameContinues = nandOperation(inputA: !physicsActive, inputB: !renderingActive)
        
        var winCondition = false
        var loseCondition = false
        
        if let entities = gameState["entities"] as? [GameEntity],
           let player = gameState["player"] as? GameEntity {
            let enemies = entities.filter { $0.id != player.id }
            winCondition = player.health > 0 && enemies.isEmpty
            loseCondition = player.health <= 0
        }
        
        let result: [String: Any] = [
            "game_continues": gameContinues,
            "win": winCondition,
            "lose": loseCondition,
            "physics_active": physicsActive,
            "rendering_active": renderingActive,
            "ai_active": aiActive
        ]
        
        let voltage = gameContinues ? 2.51 : 0.0
        return Signal(value: result, voltage: voltage)
    }
}
