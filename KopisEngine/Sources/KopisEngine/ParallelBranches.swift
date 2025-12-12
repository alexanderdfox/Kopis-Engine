import Foundation

class ParallelBranches {
    private var maze: Maze?
    
    init(maze: Maze? = nil) {
        self.maze = maze
    }
    
    func processPhysics(entities: [GameEntity], deltaTime: Double, maze: Maze?) -> [GameEntity] {
        // Validate deltaTime
        var safeDeltaTime = deltaTime
        if deltaTime.isNaN || deltaTime.isInfinite || deltaTime <= 0 {
            safeDeltaTime = 1.0 / 60.0 // Default to 60 FPS
        }
        safeDeltaTime = min(safeDeltaTime, 0.1) // Clamp to max 100ms
        let gravity = GameConstants.gravity
        let frictionCoefficient = GameConstants.frictionCoefficient
        let groundLevel = GameConstants.groundLevel
        let ceilingLevel = GameConstants.ceilingLevel
        
        var updatedEntities: [GameEntity] = []
        
        for entity in entities {
            // Get position and velocity with safety checks
            var x = entity.position.x
            var y = entity.position.y
            var z = entity.position.z
            var vx = entity.velocity.x
            var vy = entity.velocity.y
            var vz = entity.velocity.z
            
            // Validate and fix NaN/infinite values
            if x.isNaN || x.isInfinite { x = 0.0 }
            if y.isNaN || y.isInfinite { y = 0.0 }
            if z.isNaN || z.isInfinite { z = 0.0 }
            if vx.isNaN || vx.isInfinite { vx = 0.0 }
            if vy.isNaN || vy.isInfinite { vy = 0.0 }
            if vz.isNaN || vz.isInfinite { vz = 0.0 }
            
            // Apply gravity
            if entity.properties["affected_by_gravity"] as? Bool ?? true {
                vz -= gravity * safeDeltaTime
            }
            
            // Apply friction (only to NPCs, not player)
            let isPlayer = entity.id == "player"
            if !isPlayer {
                vx *= frictionCoefficient
                vy *= frictionCoefficient
                if abs(vx) < 1.0 { vx = 0.0 }
                if abs(vy) < 1.0 { vy = 0.0 }
            } else {
                if abs(vx) < 0.1 { vx = 0.0 }
                if abs(vy) < 0.1 { vy = 0.0 }
            }
            
            // Update position
            var newX = x + vx * safeDeltaTime
            var newY = y + vy * safeDeltaTime
            var newZ = z + vz * safeDeltaTime
            
            // Ground collision
            if newZ < groundLevel {
                newZ = groundLevel
                if vz < 0 { vz = 0 }
                var props = entity.properties
                props["on_ground"] = true
            } else {
                var props = entity.properties
                props["on_ground"] = false
            }
            
            // Ceiling collision
            if newZ > ceilingLevel {
                newZ = ceilingLevel
                if vz > 0 { vz = 0 }
            }
            
            // Maze collision
            var entityRadius = entity.properties["radius"] as? Double ?? 10.0
            // Ensure entityRadius is valid and not too small
            if entityRadius.isNaN || entityRadius.isInfinite || entityRadius <= 0 {
                entityRadius = 10.0
            }
            
            if let maze = maze {
                // Check if stuck in wall
                if maze.checkCollision(position: entity.position, radius: entityRadius) {
                    if let safePos = maze.findNearestSafePosition(position: entity.position, radius: entityRadius) {
                        newX = safePos.x
                        newY = safePos.y
                        vx = 0.0
                        vy = 0.0
                    }
                }
                
                // Continuous collision detection with safety checks
                // Ensure values are valid before conversion
                let stepSize = max(0.1, entityRadius * 0.5) // Prevent division by zero
                let vxStep = abs(vx * safeDeltaTime)
                let vyStep = abs(vy * safeDeltaTime)
                
                // Check for NaN/infinite before converting to Int
                let stepsX = (vxStep.isNaN || vxStep.isInfinite) ? 0 : Int(vxStep / stepSize)
                let stepsY = (vyStep.isNaN || vyStep.isInfinite) ? 0 : Int(vyStep / stepSize)
                let steps = max(1, min(10, stepsX + stepsY))
                
                if steps > 1 {
                    let stepDx = (newX - x) / Double(steps)
                    let stepDy = (newY - y) / Double(steps)
                    var lastValidX = x
                    var lastValidY = y
                    
                    for step in 1...steps {
                        let testX = x + stepDx * Double(step)
                        let testY = y + stepDy * Double(step)
                        let testPos = Vector3(x: testX, y: testY, z: newZ)
                        
                        if !maze.checkCollision(position: testPos, radius: entityRadius) {
                            lastValidX = testX
                            lastValidY = testY
                        } else {
                            newX = lastValidX
                            newY = lastValidY
                            vx *= 0.3
                            vy *= 0.3
                            break
                        }
                    }
                }
                
                // Final collision check
                let newPos = Vector3(x: newX, y: newY, z: newZ)
                if maze.checkCollision(position: newPos, radius: entityRadius) {
                    // Try X or Y only
                    let testXPos = Vector3(x: newX, y: y, z: newZ)
                    let testYPos = Vector3(x: x, y: newY, z: newZ)
                    
                    if !maze.checkCollision(position: testXPos, radius: entityRadius) {
                        newY = y
                        vx *= 0.5
                    } else if !maze.checkCollision(position: testYPos, radius: entityRadius) {
                        newX = x
                        vy *= 0.5
                    } else {
                        // Stuck - find safe position
                        if let safePos = maze.findNearestSafePosition(position: entity.position, radius: entityRadius) {
                            newX = safePos.x
                            newY = safePos.y
                            vx = 0.0
                            vy = 0.0
                        } else {
                            newX = x
                            newY = y
                            vx = 0.0
                            vy = 0.0
                        }
                    }
                }
            }
            
            var newEntity = entity
            newEntity.position = Vector3(x: newX, y: newY, z: newZ)
            newEntity.velocity = Vector3(x: vx, y: vy, z: vz)
            updatedEntities.append(newEntity)
        }
        
        return updatedEntities
    }
    
    func processRendering(entities: [GameEntity], cameraPos: Vector3) -> [String: Any] {
        var renderData: [String: Any] = [:]
        
        var entityData: [[String: Any]] = []
        for entity in entities {
            entityData.append([
                "id": entity.id,
                "x": entity.position.x,
                "y": entity.position.y,
                "z": entity.position.z,
                "health": entity.health
            ])
        }
        
        renderData["entities"] = entityData
        renderData["camera"] = [
            "x": cameraPos.x,
            "y": cameraPos.y,
            "z": cameraPos.z,
            "zoom": 1.0
        ]
        
        return renderData
    }
    
    func processAI(entities: [GameEntity], playerEntity: GameEntity, deltaTime: Double, maze: Maze?) -> [GameEntity] {
        // Simplified AI - NPCs move toward player
        var updatedEntities: [GameEntity] = []
        
        for entity in entities {
            if entity.id == playerEntity.id {
                updatedEntities.append(entity)
                continue
            }
            
            guard entity.id.contains("npc") else {
                updatedEntities.append(entity)
                continue
            }
            
            // Simple movement toward player
            let dx = playerEntity.position.x - entity.position.x
            let dy = playerEntity.position.y - entity.position.y
            let distance = sqrt(dx * dx + dy * dy)
            
            if distance > 400.0 {
                updatedEntities.append(entity)
                continue
            }
            
            let speed: Double = 50.0
            var vx = (dx / distance) * speed
            var vy = (dy / distance) * speed
            
            var newEntity = entity
            newEntity.velocity = Vector3(x: vx, y: vy, z: entity.velocity.z)
            updatedEntities.append(newEntity)
        }
        
        return updatedEntities
    }
}
