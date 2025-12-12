import Foundation

public struct GameConstants {
    public static let maxNPCs = 25
    public static let maxDepth: Double = 1000.0
    public static let cellSize: Double = 50.0
    public static let chunkSize = 20
    public static let fov: Double = 60.0
    public static let cameraHeight: Double = 150.0 // 5 feet = 150 units
    public static let groundLevel: Double = 0.0
    public static let ceilingLevel: Double = 300.0 // 10 feet = 300 units
    public static let gravity: Double = 32.2 * 30.0 // 966 units/s²
    public static let frictionCoefficient = 0.95
    public static let playerSpeed: Double = 200.0
    public static let jumpSpeed: Double = 400.0
    public static let maxBloodWalls = 10
}
