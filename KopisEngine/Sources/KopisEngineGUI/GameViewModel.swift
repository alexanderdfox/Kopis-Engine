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
    private var lastRenderWarningTime: Date?
    private var renderLogCount: Int = 0
    
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
        let inputData: [String: Any] = [
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
            gameOfLife?.update()
            golUpdateCounter = 0
            if renderLogCount % 60 == 0 {
                print("✓ GameOfLife updated")
            }
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
        guard let engine = engine, let raycaster = raycaster else {
            // Fill with dark red/brown to show something is rendering (Doom-style)
            context.setFillColor(CGColor(red: 0.1, green: 0.05, blue: 0.03, alpha: 1))
            context.fill(bounds)
            return
        }
        
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
        guard let engine = engine, let raycaster = raycaster else {
            // Debug: Log why rendering isn't happening
            let now = Date()
            if lastRenderWarningTime == nil || now.timeIntervalSince(lastRenderWarningTime!) > 2.0 {
                if engine == nil {
                    print("⚠ renderCPU: engine is nil - game may not be initialized")
                }
                if raycaster == nil {
                    print("⚠ renderCPU: raycaster is nil - game may not be initialized")
                }
                lastRenderWarningTime = now
            }
            // Fill with dark red/brown to show something is rendering (Doom-style)
            context.setFillColor(CGColor(red: 0.1, green: 0.05, blue: 0.03, alpha: 1))
            context.fill(bounds)
            return
        }
        
        let width = Int(bounds.width)
        let height = Int(bounds.height)
        guard width > 0 && height > 0 else {
            return
        }
        
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
        
        // Draw ceiling - Doom-style dark brown/red
        let ceilingHeight = max(0.0, min(Double(height), horizonY))
        let ceilingRect = CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(ceilingHeight))
        context.setFillColor(CGColor(red: 0.15, green: 0.08, blue: 0.05, alpha: 1.0)) // Dark brown-red
        context.fill(ceilingRect)
        
        // Draw floor - Doom-style darker brown/red
        let floorY = CGFloat(ceilingHeight)
        let floorHeight = CGFloat(height) - floorY
        let floorRect = CGRect(x: 0, y: floorY, width: CGFloat(width), height: floorHeight)
        context.setFillColor(CGColor(red: 0.12, green: 0.06, blue: 0.04, alpha: 1.0)) // Darker brown-red
        context.fill(floorRect)
        
        // TEST: Draw a visible test pattern to verify rendering works
        // Draw a dark red rectangle in the center (Doom-style)
        let testRect = CGRect(x: width / 2 - 50, y: height / 2 - 25, width: 100, height: 50)
        context.setFillColor(CGColor(red: 0.6, green: 0.1, blue: 0.05, alpha: 1.0))
        context.fill(testRect)
        
        // Debug: Log rendering info (throttled)
        renderLogCount += 1
        if renderLogCount == 1 || renderLogCount % 60 == 0 {
            print("✓ renderCPU: Drawing scene - size: \(width)x\(height), camera: (\(String(format: "%.1f", engine.cameraPos.x)), \(String(format: "%.1f", engine.cameraPos.y)), \(String(format: "%.1f", engine.cameraPos.z))), yaw: \(String(format: "%.1f", engine.cameraYaw))°")
        }
        
        // Ensure maze chunks are loaded at camera position
        if let maze = maze {
            maze.ensureChunksLoaded(worldPos: engine.cameraPos)
        }
        
        // Raycast for walls
        wallsWithBlood.removeAll()
        var rayResults: [(x: Int, distance: Double, wallStart: Double, wallEnd: Double, mapX: Int, mapY: Int)] = []
        var wallsFound = 0
        
        let raycastCamX = engine.cameraPos.x
        let raycastCamY = engine.cameraPos.y
        
        for x in 0..<width {
            // Calculate ray direction
            let cameraX = 2.0 * Double(x) / Double(width) - 1.0
            let rayDirX = forwardX + rightX * cameraX * tan(fov * .pi / 360.0)
            let rayDirY = forwardY + rightY * cameraX * tan(fov * .pi / 360.0)
            
            // Raycast
            if let result = raycaster.raycastWall(camX: raycastCamX, camY: raycastCamY, rayDirX: rayDirX, rayDirY: rayDirY, maxDepth: maxDepth) {
                wallsFound += 1
                let distance = result.distance
                let lineHeight = abs(Double(height) / (distance / GameConstants.cellSize))
                let drawStart = -lineHeight / 2.0 + Double(height) / 2.0 + tan(pitchRad) * Double(height) / 2.0
                let drawEnd = lineHeight / 2.0 + Double(height) / 2.0 + tan(pitchRad) * Double(height) / 2.0
                
                // Calculate wall color - Doom-style red/brown palette
                let brightness = max(0.4, min(1.0, 1.0 - distance / 600.0))
                
                // Doom-style red/brown base colors
                // Vary by map position for texture-like effect
                let hueVariation = Double((result.mapX * 7 + result.mapY * 11) % 30) - 15.0
                let baseR = 180.0 + hueVariation * 0.5  // Red component (180-195)
                let baseG = 60.0 + hueVariation * 0.3   // Brown/green component (45-75)
                let baseB = 40.0 + hueVariation * 0.2   // Dark component (25-55)
                
                // Apply brightness and side shading
                let wallR = Int(baseR * brightness * (result.side == 1 ? 0.75 : 1.0))
                let wallG = Int(baseG * brightness * (result.side == 1 ? 0.75 : 1.0))
                let wallB = Int(baseB * brightness * (result.side == 1 ? 0.75 : 1.0))
                
                // Draw wall column - ensure valid coordinates
                // Note: Context is flipped (Y-axis inverted), so smaller Y = top, larger Y = bottom
                let wallStartY = max(0.0, min(Double(height), drawStart))
                let wallEndY = max(0.0, min(Double(height), drawEnd))
                let wallHeight = abs(wallEndY - wallStartY)
                
                // Ensure we have a valid wall to draw
                if wallHeight > 0 && x >= 0 && x < width && wallStartY < Double(height) && wallEndY > 0 {
                    context.setFillColor(CGColor(red: Double(wallR) / 255.0, green: Double(wallG) / 255.0, blue: Double(wallB) / 255.0, alpha: 1.0))
                    // In flipped coordinates, drawStart is top (smaller Y) and drawEnd is bottom (larger Y)
                    context.fill(CGRect(x: x, y: Int(wallStartY), width: 1, height: Int(wallHeight)))
                }
                
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
        
        // Debug: Log wall detection
        if renderLogCount == 1 || renderLogCount % 60 == 0 {
            print("✓ renderCPU: Found \(wallsFound) walls out of \(width) rays")
            print("  - Camera position: (\(String(format: "%.1f", raycastCamX)), \(String(format: "%.1f", raycastCamY)))")
            print("  - Camera yaw: \(String(format: "%.1f", engine.cameraYaw))°")
            print("  - GameOfLife: \(gameOfLife != nil ? "initialized" : "nil")")
            print("  - Walls with blood: \(wallsWithBlood.count)")
            if wallsFound == 0 {
                print("⚠ WARNING: No walls found!")
                print("⚠ Check: 1) Maze chunks loaded? 2) Camera in valid position? 3) Raycast working?")
                // Draw a test wall to verify rendering works (Doom-style dark red)
                context.setFillColor(CGColor(red: 0.6, green: 0.1, blue: 0.05, alpha: 1.0))
                context.fill(CGRect(x: width / 2 - 10, y: 0, width: 20, height: height))
                print("⚠ Drew test red wall in center to verify rendering")
            }
        }
        
        // Render blood pattern on 10 closest walls
        if let gol = gameOfLife, wallsWithBlood.count > 0 {
            let sortedWalls = wallsWithBlood.sorted { $0.value < $1.value }.prefix(10)
            let bloodWallsSet = Set(sortedWalls.map { $0.key })
            
            let pattern = gol.getPattern()
            let golHeight = gol.height
            let golWidth = gol.width
            
            var bloodPixelsDrawn = 0
            for result in rayResults {
                if result.distance < 200 && result.distance > 0 {
                    let wallKey = "\(result.mapX)_\(result.mapY)"
                    if bloodWallsSet.contains(wallKey) {
                        let wallHeight = result.wallEnd - result.wallStart
                        let cellH = max(1.0, wallHeight / Double(golHeight))
                        
                        if cellH >= 1.0 && wallHeight > 0 {
                            let patternX = Int(Double(result.x) / Double(width) * Double(golWidth))
                            let clampedPatternX = max(0, min(golWidth - 1, patternX))
                            
                            for y in 0..<golHeight {
                                if pattern[y][clampedPatternX] {
                                    let screenY = result.wallStart + (Double(y) * wallHeight) / Double(golHeight)
                                    let clampedScreenY = max(result.wallStart, min(result.wallEnd - 1, screenY))
                                    if clampedScreenY >= result.wallStart && clampedScreenY < result.wallEnd {
                                        // Doom-style blood red - darker, more saturated
                                        let variation = Double((result.x * 7 + y * 11) % 31) - 15.0
                                        let bloodRed = max(150, min(220, Int(180 + (Double(y) / Double(golHeight)) * 20.0 + variation)))
                                        let bloodGreen = max(0, min(30, Int((Double(y) / Double(golHeight)) * 15.0 + variation / 3.0)))
                                        let bloodBlue = max(0, min(20, Int((Double(y) / Double(golHeight)) * 10.0 + variation / 4.0)))
                                        
                                        context.setFillColor(CGColor(red: Double(bloodRed) / 255.0, green: Double(bloodGreen) / 255.0, blue: Double(bloodBlue) / 255.0, alpha: 1.0))
                                        let pixelHeight = Int(max(1, cellH))
                                        context.fill(CGRect(x: result.x, y: Int(clampedScreenY), width: 1, height: pixelHeight))
                                        bloodPixelsDrawn += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            if renderLogCount == 1 || renderLogCount % 60 == 0 {
                print("✓ GameOfLife: Rendered \(bloodPixelsDrawn) blood pixels on \(bloodWallsSet.count) walls")
            }
        } else {
            if renderLogCount == 1 || renderLogCount % 60 == 0 {
                if gameOfLife == nil {
                    print("⚠ GameOfLife is nil - not rendering blood pattern")
                } else if wallsWithBlood.count == 0 {
                    print("⚠ No walls with blood to render (wallsWithBlood.count = 0)")
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
            // finalZ = tempY * sin(pitchRad) + tempZ * cos(pitchRad) // Not used in 2D projection
            
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
            
            // Color - Doom-style red/brown for all entities
            let color: CGColor
            if isNPC {
                // NPCs: Darker red
                color = CGColor(red: 0.8, green: 0.15, blue: 0.1, alpha: CGFloat(brightness))
            } else {
                // Other entities: Reddish brown
                color = CGColor(red: 0.7, green: 0.2, blue: 0.1, alpha: CGFloat(brightness))
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
