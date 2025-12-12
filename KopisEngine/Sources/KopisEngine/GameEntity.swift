import Foundation

public struct Vector3 {
    public var x: Double
    public var y: Double
    public var z: Double
    
    public init(x: Double, y: Double, z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }
    
    public static let zero = Vector3(x: 0, y: 0, z: 0)
    
    public func distance(to other: Vector3) -> Double {
        let dx = x - other.x
        let dy = y - other.y
        let dz = z - other.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
    
    func distance2D(to other: Vector3) -> Double {
        let dx = x - other.x
        let dy = y - other.y
        return sqrt(dx * dx + dy * dy)
    }
}

public struct GameEntity {
    public let id: String
    public var position: Vector3
    public var velocity: Vector3
    public var health: Double
    public var description: String
    public var properties: [String: Any]
    
    public init(id: String, position: Vector3 = .zero, velocity: Vector3 = .zero, 
         health: Double = 100.0, description: String = "", 
         properties: [String: Any] = [:]) {
        self.id = id
        self.position = position
        self.velocity = velocity
        self.health = health
        self.description = description
        self.properties = properties
    }
}

public struct Signal {
    public var value: Any
    public var voltage: Double
    public var timestamp: TimeInterval
    
    public init(value: Any, voltage: Double = 2.51, timestamp: TimeInterval = Date().timeIntervalSince1970) {
        self.value = value
        self.voltage = voltage
        self.timestamp = timestamp
    }
}
