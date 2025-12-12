import SwiftUI
import AppKit
import CoreGraphics
import KopisEngine

class GameViewModel: ObservableObject {
    @Published var fps: Double = 60.0
    @Published var entityCount: Int = 0
    @Published var player: GameEntity?
    
    private(set) var engine: KopisEngine?
    private(set) var maze: Maze?
    private(set) var raycaster: Raycaster?
    private(set) var gameOfLife: GameOfLife?
    private var soundManager: SoundManager?
    private var lastPlayerPos: Vector3?
    private var lastFrameTime: Date = Date()
    private var frameCount: Int = 0
    private var fpsUpdateTime: Date = Date()
    private var golUpdateCounter: Int = 0
    
    private var keysPressed: Set<String> = []
    private var mouseDeltaX: Float = 0
    private var mouseDeltaY: Float = 0
    private var wallsWithBlood: [String: Double] = [:] // wallKey -> distance
    
    func startGame() {
        print("Initializing Kopis Engine GUI...")
        
        // Initialize sound manager
        print("Initializing sound system...")
        soundManager = SoundManager()
        if soundManager?.enabled == true {
            print("✓ Sound system initialized")
        } else {
            print("⚠ Sound system unavailable")
        }
        
        // Create maze
        maze = Maze(chunkSize: 20, cellSize: 50.0, loadRadius: 3)
        print("✓ Infinite maze system initialized")
        
        guard let safeMaze = maze else {
            print("⚠ Failed to create maze")
            return
        }
        
        // Create engine with sound manager
        engine = KopisEngine(maze: safeMaze, soundManager: soundManager)
        print("✓ Kopis Engine initialized")
        
        // Create raycaster
        raycaster = Raycaster(maze: safeMaze)
        
        // Create Game of Life for blood patterns
        gameOfLife = GameOfLife(seed: 42)
        print("✓ Raycaster and Game of Life initialized")
        
        // Find valid starting position (like Python version)
        safeMaze.ensureChunksLoaded(worldPos: Vector3.zero)
        
        // Search for valid starting positions
        var validPositions: [Vector3] = []
        let searchRadius = 20
        let cellSize = safeMaze.cellSize
        
        for cellX in -searchRadius...searchRadius {
            for cellY in -searchRadius...searchRadius {
                let testWorld = safeMaze.cellToWorld(cellPos: (cellX, cellY))
                let testPos = Vector3(x: testWorld.x, y: testWorld.y, z: GameConstants.groundLevel)
                
                // Check collision with proper margin
                if !safeMaze.checkCollision(position: testPos, radius: 12.0 + 2.0) {
                    // Verify it's a path cell
                    if safeMaze.isPathCell(cellPos: (cellX, cellY)) {
                        validPositions.append(testPos)
                    }
                }
            }
        }
        
        // Select random valid position or use origin
        let startPos: Vector3
        if let randomPos = validPositions.randomElement() {
            startPos = randomPos
            print("✓ Found \(validPositions.count) valid starting positions, selected random position")
        } else {
            // Fallback: try to find any safe position
            startPos = Vector3(x: 0, y: 0, z: GameConstants.groundLevel)
            print("⚠ Using default starting position")
        }
        
        // Create player entity at valid maze position
        let playerEntity = GameEntity(
            id: "player",
            position: startPos,
            velocity: .zero,
            health: 100.0,
            properties: [
                "affected_by_gravity": true,
                "radius": 12.0,
                "on_ground": true
            ]
        )
        engine?.addEntity(playerEntity)
        self.player = playerEntity
        
        // Set camera to look at a nearby wall (like Python version)
        let startCell = safeMaze.worldToCell(worldPos: startPos)
        var nearestWall: (Int, Int)?
        var nearestDistance = Double.infinity
        
        for dx in -5...5 {
            for dy in -5...5 {
                if dx == 0 && dy == 0 { continue }
                let testCell = (startCell.0 + dx, startCell.1 + dy)
                let testWorld = safeMaze.cellToWorld(cellPos: testCell)
                let testPos = Vector3(x: testWorld.x, y: testWorld.y, z: 0)
                
                if safeMaze.checkCollision(position: testPos, radius: cellSize / 2.0) {
                    let dist = sqrt(Double(dx * dx + dy * dy))
                    if dist < nearestDistance {
                        nearestDistance = dist
                        nearestWall = testCell
                    }
                }
            }
        }
        
        // Set initial camera yaw to look at wall
        if let wall = nearestWall {
            let wallWorld = safeMaze.cellToWorld(cellPos: wall)
            let dx = wallWorld.x - startPos.x
            let dy = wallWorld.y - startPos.y
            engine?.cameraYaw = atan2(dy, dx) * 180.0 / .pi + 90.0 + 180.0
        }
        
        // Create NPCs
        for i in 0..<GameConstants.maxNPCs {
            let npc = GameEntity(
                id: "npc_\(i)",
                position: Vector3(x: Double(i * 100), y: Double(i * 100), z: GameConstants.groundLevel),
                velocity: .zero,
                health: 50.0,
                properties: ["radius": 8.0, "on_ground": true]
            )
            engine?.addEntity(npc)
        }
        
        print("✓ Game entities created")
        print("Game running...")
    }
    
    func updateFrame() {
        guard let engine = engine else { return }
        
        // Check for collision sound (player hit wall)
        if let player = engine.player, let lastPos = lastPlayerPos {
            let dx = player.position.x - lastPos.x
            let dy = player.position.y - lastPos.y
            let dz = player.position.z - lastPos.z
            let distance = sqrt(dx * dx + dy * dy + dz * dz)
            
            // If player moved very little but velocity suggests collision
            if distance < 1.0 && (abs(player.velocity.x) > 0.1 || abs(player.velocity.y) > 0.1) {
                soundManager?.play("collision", volume: 0.4)
            }
        }
        
        lastPlayerPos = engine.player?.position
        
        // Calculate FPS
        frameCount += 1
        let now = Date()
        if now.timeIntervalSince(fpsUpdateTime) >= 1.0 {
            fps = Double(frameCount)
            frameCount = 0
            fpsUpdateTime = now
        }
        
        // Prepare input data
        var inputData: [String: Any] = [
            "keys": keysPressed.reduce(into: [String: Bool]()) { dict, key in
                dict[key] = true
            },
            "mouse": [
                "dx": Double(mouseDeltaX),
                "dy": Double(mouseDeltaY)
            ]
        ]
        
        // Reset mouse delta
        mouseDeltaX = 0
        mouseDeltaY = 0
        
        // Process frame
        let result = engine.processFrame(inputData: inputData)
        
        // Update UI state
        if let entities = result["entities"] as? [GameEntity] {
            entityCount = entities.count
            player = entities.first { $0.id == "player" }
        }
        
        // Update Game of Life periodically
        golUpdateCounter += 1
        if golUpdateCounter >= 10 {
            golUpdateCounter = 0
            gameOfLife?.update()
        }
    }
    
    func handleKeyEvent(_ event: NSEvent, isDown: Bool) {
        guard let characters = event.charactersIgnoringModifiers?.lowercased() else { return }
        
        let key = characters.first?.description ?? ""
        
        switch key {
        case "w", "a", "s", "d":
            if isDown {
                keysPressed.insert(key)
            } else {
                keysPressed.remove(key)
            }
        case " ":
            if isDown {
                keysPressed.insert("space")
            } else {
                keysPressed.remove("space")
            }
        case "q", "e":
            if isDown {
                keysPressed.insert(key)
            } else {
                keysPressed.remove(key)
            }
        default:
            break
        }
        
        // Handle F11 for fullscreen
        if event.keyCode == 122 { // F11
            if isDown {
                toggleFullscreen()
            }
        }
    }
    
    func handleMouseMovement(dx: Float, dy: Float) {
        // Apply sensitivity (matching Python version)
        mouseDeltaX += dx * 0.1
        mouseDeltaY += dy * 0.1
    }
    
    private var metalRenderer: MetalRenderer?
    
    func render(context: CGContext, bounds: CGRect) {
        guard let engine = engine, let raycaster = raycaster else { return }
        
        // Initialize Metal renderer if needed
        if metalRenderer == nil {
            metalRenderer = MetalRenderer()
        }
        
        // Use Metal renderer if available (currently uses CPU rendering with Metal infrastructure)
        if let renderer = metalRenderer {
            renderer.render(
                context: context,
                bounds: bounds,
                engine: engine,
                raycaster: raycaster,
                gameOfLife: gameOfLife,
                cpuRenderFunction: { [weak self] ctx, bnds in
                    self?.renderCPU(context: ctx, bounds: bnds)
                }
            )
            return
        }
        
        // Fallback to CPU rendering
        renderCPU(context: context, bounds: bounds)
    }
    
    private func renderCPU(context: CGContext, bounds: CGRect) {
        guard let engine = engine, let raycaster = raycaster else { return }
        
        let width = Int(bounds.width)
        let height = Int(bounds.height)
        let fov: Double = GameConstants.fov
        let maxDepth = GameConstants.maxDepth
        
        // Calculate camera direction
        let yawRad = engine.cameraYaw * .pi / 180.0
        let pitchRad = engine.cameraPitch * .pi / 180.0
        
        let forwardX = sin(yawRad)
        let forwardY = -cos(yawRad)
        let rightX = cos(yawRad)
        let rightY = sin(yawRad)
        
        // Calculate horizon based on pitch
        let horizonY = Double(height) / 2.0 + tan(pitchRad) * Double(height) / 2.0
        
        // Draw ceiling
        let ceilingRect = CGRect(x: 0, y: 0, width: bounds.width, height: CGFloat(horizonY))
        context.setFillColor(CGColor(red: 0.2, green: 0.2, blue: 0.2, alpha: 1.0))
        context.fill(ceilingRect)
        
        // Draw floor
        let floorRect = CGRect(x: 0, y: CGFloat(horizonY), width: bounds.width, height: bounds.height - CGFloat(horizonY))
        context.setFillColor(CGColor(red: 0.16, green: 0.16, blue: 0.16, alpha: 1.0))
        context.fill(floorRect)
        
        // Raycast for walls
        wallsWithBlood.removeAll()
        var rayResults: [(x: Int, distance: Double, wallStart: Double, wallEnd: Double, mapX: Int, mapY: Int)] = []
        
        let raycastCamX = engine.cameraPos.x
        let raycastCamY = engine.cameraPos.y
        
        for x in 0..<width {
            // Calculate ray direction
            let cameraX = 2.0 * Double(x) / Double(width) - 1.0
            let rayDirX = forwardX + rightX * cameraX * tan(fov * .pi / 360.0)
            let rayDirY = forwardY + rightY * cameraX * tan(fov * .pi / 360.0)
            
            // Raycast
            if let result = raycaster.raycastWall(camX: raycastCamX, camY: raycastCamY, rayDirX: rayDirX, rayDirY: rayDirY, maxDepth: maxDepth) {
                let distance = result.distance
                let lineHeight = abs(Double(height) / (distance / GameConstants.cellSize))
                let drawStart = -lineHeight / 2.0 + Double(height) / 2.0 + tan(pitchRad) * Double(height) / 2.0
                let drawEnd = lineHeight / 2.0 + Double(height) / 2.0 + tan(pitchRad) * Double(height) / 2.0
                
                // Calculate wall color
                let brightness = max(0.3, min(1.0, 1.0 - distance / 500.0))
                let baseR = 30.0 + Double((result.mapX * 7 + result.mapY * 11) % 50)
                let baseG = 25.0 + Double((result.mapX * 5 + result.mapY * 13) % 35)
                let baseB = 20.0 + Double((result.mapX * 3 + result.mapY * 17) % 30)
                
                let wallR = Int(baseR * brightness * (result.side == 1 ? 0.8 : 1.0))
                let wallG = Int(baseG * brightness * (result.side == 1 ? 0.8 : 1.0))
                let wallB = Int(baseB * brightness * (result.side == 1 ? 0.8 : 1.0))
                
                // Draw wall column
                context.setFillColor(CGColor(red: Double(wallR) / 255.0, green: Double(wallG) / 255.0, blue: Double(wallB) / 255.0, alpha: 1.0))
                context.fill(CGRect(x: x, y: Int(max(0, drawStart)), width: 1, height: Int(min(Double(height), drawEnd) - max(0, drawStart))))
                
                rayResults.append((x: x, distance: distance, wallStart: drawStart, wallEnd: drawEnd, mapX: result.mapX, mapY: result.mapY))
                
                // Track walls for blood pattern
                if distance < 200 {
                    let wallKey = "\(result.mapX)_\(result.mapY)"
                    if wallsWithBlood[wallKey] == nil || wallsWithBlood[wallKey]! > distance {
                        wallsWithBlood[wallKey] = distance
                    }
                }
            } else {
                rayResults.append((x: x, distance: maxDepth, wallStart: 0, wallEnd: 0, mapX: 0, mapY: 0))
            }
        }
        
        // Render blood pattern on 10 closest walls
        if let gol = gameOfLife, wallsWithBlood.count > 0 {
            let sortedWalls = wallsWithBlood.sorted { $0.value < $1.value }.prefix(10)
            let bloodWallsSet = Set(sortedWalls.map { $0.key })
            
            let pattern = gol.getPattern()
            let golHeight = gol.height
            let golWidth = gol.width
            
            for result in rayResults {
                if result.distance < 200 {
                    let wallKey = "\(result.mapX)_\(result.mapY)"
                    if bloodWallsSet.contains(wallKey) {
                        let wallHeight = result.wallEnd - result.wallStart
                        let cellH = max(1.0, wallHeight / Double(golHeight))
                        
                        if cellH >= 1.0 {
                            let patternX = Int(Double(result.x) / Double(width) * Double(golWidth))
                            
                            for y in 0..<golHeight {
                                if pattern[y][patternX] {
                                    let screenY = result.wallStart + (Double(y) * wallHeight) / Double(golHeight)
                                    if screenY >= result.wallStart && screenY < result.wallEnd {
                                        let variation = Double((result.x * 7 + y * 11) % 31) - 15.0
                                        let bloodRed = max(120, min(200, Int(139 + (Double(y) / Double(golHeight)) * 39.0 + variation)))
                                        let bloodGreen = max(0, min(60, Int((Double(y) / Double(golHeight)) * 34.0 + variation / 2.0)))
                                        let bloodBlue = max(0, min(50, Int((Double(y) / Double(golHeight)) * 33.0 + variation / 3.0)))
                                        
                                        context.setFillColor(CGColor(red: Double(bloodRed) / 255.0, green: Double(bloodGreen) / 255.0, blue: Double(bloodBlue) / 255.0, alpha: 1.0))
                                        context.fill(CGRect(x: result.x, y: Int(screenY), width: 1, height: Int(max(1, cellH))))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Draw entities as Doom-style billboard sprites (like Python version)
        let entities = engine.entities.filter { $0.id != "player" } // Don't render player in FPV
        
        var sprites: [(entity: GameEntity, screenX: Double, screenY: Double, distance: Double, depth: Double)] = []
        
        let entityCamX = engine.cameraPos.x
        let entityCamY = engine.cameraPos.y
        let entityCamZ = engine.cameraPos.z
        
        for entity in entities {
            let ex = entity.position.x
            let ey = entity.position.y
            let ez = entity.position.z
            
            let dx = ex - entityCamX
            let dy = ey - entityCamY
            let dz = ez - entityCamZ
            
            // Transform to camera space
            let yawRad = engine.cameraYaw * .pi / 180.0
            let pitchRad = engine.cameraPitch * .pi / 180.0
            
            // Rotate by yaw
            let tempX = dx * cos(yawRad) - dy * sin(yawRad)
            let tempY = dx * sin(yawRad) + dy * cos(yawRad)
            let tempZ = dz
            
            // Rotate by pitch
            let finalY = tempY * cos(pitchRad) - tempZ * sin(pitchRad)
            let _ = tempY * sin(pitchRad) + tempZ * cos(pitchRad) // finalZ (not used in 2D projection)
            
            // Project to screen
            let fovScale = 1.0 / tan(fov * .pi / 360.0)
            let depth = max(0.1, -tempX)
            
            if depth <= 0 || tempX < 0 {
                continue // Behind camera
            }
            
            let screenX = Double(width) / 2.0 + tempY * fovScale * (Double(height) / depth)
            let screenY = Double(height) / 2.0 - finalY * fovScale * (Double(height) / depth)
            let entityDistance = sqrt(dx * dx + dy * dy + dz * dz)
            
            sprites.append((entity: entity, screenX: screenX, screenY: screenY, distance: entityDistance, depth: depth))
        }
        
        // Sort by distance (farthest first for proper depth)
        sprites.sort { $0.distance > $1.distance }
        
        // Render sprites
        for sprite in sprites {
            let entity = sprite.entity
            let isNPC = entity.id.contains("npc")
            let baseRadius: Double = isNPC ? 10.0 : 8.0
            
            // Calculate sprite size based on distance
            let spriteSize = max(5.0, min(50.0, baseRadius * 2.0 * (Double(height) / sprite.depth)))
            let spriteX = sprite.screenX - spriteSize / 2.0
            let spriteY = sprite.screenY - spriteSize / 2.0
            
            // Check if sprite is on screen
            if spriteX + spriteSize < 0 || spriteX > Double(width) ||
               spriteY + spriteSize < 0 || spriteY > Double(height) {
                continue
            }
            
            // Check if sprite is behind a wall (depth check)
            let screenXInt = Int(sprite.screenX)
            if screenXInt >= 0 && screenXInt < width {
                // Find corresponding ray result
                var behindWall = false
                for result in rayResults {
                    if result.x == screenXInt && sprite.distance > result.distance {
                        behindWall = true
                        break
                    }
                }
                if behindWall {
                    continue
                }
            }
            
            // Brightness based on distance
            let brightness = max(0.5, min(1.0, 1.0 - sprite.distance / 500.0))
            
            // Color
            let color: CGColor
            if isNPC {
                color = CGColor(red: 1.0, green: 0.0, blue: 0.0, alpha: CGFloat(brightness))
            } else {
                color = CGColor(red: 0.0, green: 1.0, blue: 0.0, alpha: CGFloat(brightness))
            }
            
            // Draw sprite as circle (Doom-style)
            context.setFillColor(color)
            let rect = CGRect(x: spriteX, y: spriteY, width: spriteSize, height: spriteSize)
            context.fillEllipse(in: rect)
        }
    }
    
    
    private func toggleFullscreen() {
        if let window = NSApplication.shared.windows.first {
            window.toggleFullScreen(nil)
        }
    }
}
