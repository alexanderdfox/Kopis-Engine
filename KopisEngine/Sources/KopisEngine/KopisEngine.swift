import Foundation

public class KopisEngine {
    private let stackedTransformers: StackedTransformers
    private let parallelBranches: ParallelBranches
    private let nandGate: NANDGate
    private let feedbackLoop: FeedbackLoop
    private let maze: Maze?
    private let soundManager: SoundManager?
    
    public var gameState: GameState = .menu
    public var entities: [GameEntity] = []
    public var player: GameEntity?
    public var frameCount: Int = 0
    private var lastFrameTime: TimeInterval = Date().timeIntervalSince1970
    
    // Camera system
    public var cameraPos: Vector3 = .zero
    public var cameraYaw: Double = 0.0
    public var cameraPitch: Double = 0.0
    public var cameraRoll: Double = 0.0
    public var fpvMode: Bool = true
    public let cameraHeight: Double = GameConstants.cameraHeight
    
    public init(maze: Maze? = nil, soundManager: SoundManager? = nil) {
        self.maze = maze
        self.soundManager = soundManager
        self.stackedTransformers = StackedTransformers(useTransformers: false)
        self.parallelBranches = ParallelBranches(maze: maze)
        self.nandGate = NANDGate()
        self.feedbackLoop = FeedbackLoop()
        // Initialize lastFrameTime to current time to prevent huge deltaTime on first frame
        self.lastFrameTime = Date().timeIntervalSince1970
    }
    
    public func addEntity(_ entity: GameEntity) {
        entities.append(entity)
        if entity.id == "player" {
            player = entity
        }
    }
    
    private var wasMoving = false
    
    func processPlayerInput(inputData: [String: Any], deltaTime: Double) {
        guard let player = player else { return }
        
        let keys = inputData["keys"] as? [String: Bool] ?? [:]
        let playerSpeed = GameConstants.playerSpeed
        let jumpSpeed = GameConstants.jumpSpeed
        
        var vx = 0.0
        var vy = 0.0
        var vz = player.velocity.z
        
        let onGround = player.properties["on_ground"] as? Bool ?? false
        
        // Calculate movement direction based on camera yaw
        let yawRad = cameraYaw * .pi / 180.0
        let forwardX = sin(yawRad)
        let forwardY = -cos(yawRad)
        let rightX = cos(yawRad)
        let rightY = sin(yawRad)
        
        var moveForward = 0.0
        var moveRight = 0.0
        
        if keys["w"] == true { moveForward += 1.0 }
        if keys["s"] == true { moveForward -= 1.0 }
        if keys["a"] == true { moveRight -= 1.0 }
        if keys["d"] == true { moveRight += 1.0 }
        
        if moveForward != 0.0 || moveRight != 0.0 {
            let magnitude = sqrt(moveForward * moveForward + moveRight * moveRight)
            if magnitude > 0.0 {
                moveForward /= magnitude
                moveRight /= magnitude
                
                if abs(moveForward) > 0.1 && abs(moveRight) > 0.1 {
                    moveForward *= 0.707
                    moveRight *= 0.707
                }
            }
            
            vx = (forwardX * moveForward + rightX * moveRight) * playerSpeed
            vy = (forwardY * moveForward + rightY * moveRight) * playerSpeed
        }
        
        // Jumping
        if keys["space"] == true && onGround {
            vz = jumpSpeed
            var props = player.properties
            props["on_ground"] = false
            soundManager?.play("move_start", volume: 0.4) // Jump sound
        }
        
        // Play sound effects for movement
        let isMoving = abs(vx) > 0.1 || abs(vy) > 0.1
        if isMoving && !wasMoving {
            // Just started moving
            soundManager?.play("move_start", volume: 0.3)
        } else if isMoving {
            // Continue moving - play footstep occasionally
            if Double.random(in: 0..<1) < 0.05 { // 5% chance per frame
                soundManager?.play("footstep", volume: 0.2)
            }
        }
        wasMoving = isMoving
        
        // Update player velocity
        if let index = entities.firstIndex(where: { $0.id == "player" }) {
            var updatedPlayer = player
            updatedPlayer.velocity = Vector3(x: vx, y: vy, z: vz)
            entities[index] = updatedPlayer
            self.player = updatedPlayer
        }
    }
    
    public func processFrame(inputData: [String: Any]) -> [String: Any] {
        frameCount += 1
        let currentTime = Date().timeIntervalSince1970
        
        // Calculate deltaTime with safety checks
        var deltaTime = currentTime - lastFrameTime
        lastFrameTime = currentTime
        
        // Clamp deltaTime to prevent huge jumps (e.g., on first frame or after pause)
        // Max 0.1 seconds (100ms) = 10 FPS minimum
        deltaTime = min(deltaTime, 0.1)
        
        // Ensure deltaTime is valid (not NaN or infinite)
        if deltaTime.isNaN || deltaTime.isInfinite || deltaTime <= 0 {
            deltaTime = 1.0 / 60.0 // Default to 60 FPS
        }
        
        // Process player input
        if player != nil {
            processPlayerInput(inputData: inputData, deltaTime: deltaTime)
        }
        
        // Update camera rotation from mouse
        if fpvMode, let mouse = inputData["mouse"] as? [String: Double] {
            if let dx = mouse["dx"] {
                cameraYaw += dx * 0.1
                cameraYaw = cameraYaw.truncatingRemainder(dividingBy: 360.0)
            }
            if let dy = mouse["dy"] {
                cameraPitch += dy * 0.1
                cameraPitch = cameraPitch.truncatingRemainder(dividingBy: 360.0)
            }
        }
        
        // Update camera position
        if let player = player {
            let targetX = player.position.x
            let targetY = player.position.y
            let targetZ = fpvMode ? player.position.z + cameraHeight : player.position.z
            
            cameraPos = Vector3(
                x: targetX,
                y: targetY,
                z: targetZ
            )
        }
        
        // Process through transformer circuit
        let inputSignal = stackedTransformers.process(inputData: inputData)
        
        // Process parallel branches
        let updatedEntities = parallelBranches.processPhysics(
            entities: entities,
            deltaTime: deltaTime,
            maze: maze
        )
        entities = updatedEntities
        
        let renderData = parallelBranches.processRendering(
            entities: entities,
            cameraPos: cameraPos
        )
        
        let aiEntities = parallelBranches.processAI(
            entities: entities,
            playerEntity: player ?? GameEntity(id: "dummy"),
            deltaTime: deltaTime,
            maze: maze
        )
        entities = aiEntities
        
        // NAND gate evaluation
        let signals: [String: Signal] = [
            "physics": Signal(value: renderData, voltage: 2.51),
            "rendering": Signal(value: renderData, voltage: 2.51),
            "ai": Signal(value: renderData, voltage: 2.51)
        ]
        
        let gameStateDict: [String: Any] = [
            "entities": entities,
            "player": player as Any
        ]
        
        let nandResult = nandGate.evaluate(signals: signals, gameState: gameStateDict)
        
        // Feedback loop
        let persistentState = feedbackLoop.update(signal: nandResult, gameState: gameStateDict)
        
        return [
            "entities": entities,
            "camera": [
                "x": cameraPos.x,
                "y": cameraPos.y,
                "z": cameraPos.z,
                "yaw": cameraYaw,
                "pitch": cameraPitch,
                "roll": cameraRoll
            ],
            "frame_count": frameCount,
            "delta_time": deltaTime,
            "persistent_state": persistentState
        ]
    }
}
