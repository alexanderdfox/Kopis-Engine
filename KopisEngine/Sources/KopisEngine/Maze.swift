import Foundation

public class Maze {
    public let chunkSize: Int
    public let cellSize: Double
    public let loadRadius: Int
    private var chunks: [ChunkKey: MazeChunk] = [:]
    private var lastCleanupPos: Vector3 = .zero
    
    public init(chunkSize: Int = 20, cellSize: Double = 50.0, loadRadius: Int = 3) {
        self.chunkSize = chunkSize
        self.cellSize = cellSize
        self.loadRadius = loadRadius
    }
    
    func getChunkCoords(worldPos: Vector3) -> (Int, Int) {
        let worldCellX = Int(worldPos.x / cellSize)
        let worldCellY = Int(worldPos.y / cellSize)
        var chunkX = worldCellX / chunkSize
        var chunkY = worldCellY / chunkSize
        
        if worldCellX < 0 {
            chunkX = (worldCellX - chunkSize + 1) / chunkSize
        }
        if worldCellY < 0 {
            chunkY = (worldCellY - chunkSize + 1) / chunkSize
        }
        
        return (chunkX, chunkY)
    }
    
    private func getOrCreateChunk(chunkX: Int, chunkY: Int) -> MazeChunk {
        let key = ChunkKey(x: chunkX, y: chunkY)
        
        if chunks[key] == nil {
            var adjacentChunks: [String: MazeChunk] = [:]
            
            if let north = chunks[ChunkKey(x: chunkX, y: chunkY - 1)] {
                adjacentChunks["north"] = north
            }
            if let south = chunks[ChunkKey(x: chunkX, y: chunkY + 1)] {
                adjacentChunks["south"] = south
            }
            if let west = chunks[ChunkKey(x: chunkX - 1, y: chunkY)] {
                adjacentChunks["west"] = west
            }
            if let east = chunks[ChunkKey(x: chunkX + 1, y: chunkY)] {
                adjacentChunks["east"] = east
            }
            
            let newChunk = MazeChunk(chunkX: chunkX, chunkY: chunkY, 
                                    chunkSize: chunkSize, cellSize: cellSize,
                                    adjacentChunks: adjacentChunks)
            chunks[key] = newChunk
        }
        
        return chunks[key]!
    }
    
    public func ensureChunksLoaded(worldPos: Vector3) {
        let (centerChunkX, centerChunkY) = getChunkCoords(worldPos: worldPos)
        
        for dx in -loadRadius...loadRadius {
            for dy in -loadRadius...loadRadius {
                let chunkX = centerChunkX + dx
                let chunkY = centerChunkY + dy
                _ = getOrCreateChunk(chunkX: chunkX, chunkY: chunkY)
            }
        }
        
        // Periodic cleanup
        if worldPos.distance2D(to: lastCleanupPos) > cellSize * Double(chunkSize) {
            cleanupDistantChunks(worldPos: worldPos)
            lastCleanupPos = worldPos
        }
    }
    
    private func cleanupDistantChunks(worldPos: Vector3) {
        let (centerChunkX, centerChunkY) = getChunkCoords(worldPos: worldPos)
        let cleanupRadius = loadRadius + 2
        
        let keysToRemove = chunks.keys.filter { key in
            let distX = abs(key.x - centerChunkX)
            let distY = abs(key.y - centerChunkY)
            return distX > cleanupRadius || distY > cleanupRadius
        }
        
        for key in keysToRemove {
            chunks.removeValue(forKey: key)
        }
    }
    
    public func worldToCell(worldPos: Vector3) -> (Int, Int) {
        let cellX = Int(worldPos.x / cellSize)
        let cellY = Int(worldPos.y / cellSize)
        return (cellX, cellY)
    }
    
    public func cellToWorld(cellPos: (Int, Int)) -> Vector3 {
        let worldX = Double(cellPos.0) * cellSize + cellSize / 2.0
        let worldY = Double(cellPos.1) * cellSize + cellSize / 2.0
        return Vector3(x: worldX, y: worldY, z: 0)
    }
    
    public func checkCollision(position: Vector3, radius: Double) -> Bool {
        ensureChunksLoaded(worldPos: position)
        
        let (cellX, cellY) = worldToCell(worldPos: position)
        
        for dx in -1...1 {
            for dy in -1...1 {
                let checkCell = (cellX + dx, cellY + dy)
                
                var chunkX = checkCell.0 / chunkSize
                var chunkY = checkCell.1 / chunkSize
                if checkCell.0 < 0 {
                    chunkX = (checkCell.0 - chunkSize + 1) / chunkSize
                }
                if checkCell.1 < 0 {
                    chunkY = (checkCell.1 - chunkSize + 1) / chunkSize
                }
                
                let chunk = getOrCreateChunk(chunkX: chunkX, chunkY: chunkY)
                
                var localX = checkCell.0 % chunkSize
                var localY = checkCell.1 % chunkSize
                if checkCell.0 < 0 {
                    localX = (checkCell.0 % chunkSize + chunkSize) % chunkSize
                }
                if checkCell.1 < 0 {
                    localY = (checkCell.1 % chunkSize + chunkSize) % chunkSize
                }
                
                let cellKey = "\(localX),\(localY)"
                if chunk.walls.contains(cellKey) {
                    let wallWorld = cellToWorld(cellPos: checkCell)
                    let dist = sqrt(pow(position.x - wallWorld.x, 2) + pow(position.y - wallWorld.y, 2))
                    let collisionMargin = radius + cellSize / 2.0 - 2.0
                    if dist < collisionMargin {
                        return true
                    }
                }
            }
        }
        
        return false
    }
    
    func findNearestSafePosition(position: Vector3, radius: Double, maxSearchRadius: Int = 10) -> Vector3? {
        ensureChunksLoaded(worldPos: position)
        
        for searchRadius in 1...maxSearchRadius {
            let stepSize = cellSize * 0.5
            for dx in -searchRadius...searchRadius {
                for dy in -searchRadius...searchRadius {
                    if abs(dx) + abs(dy) > searchRadius {
                        continue
                    }
                    
                    let testX = position.x + Double(dx) * stepSize
                    let testY = position.y + Double(dy) * stepSize
                    let testPos = Vector3(x: testX, y: testY, z: position.z)
                    
                    if !checkCollision(position: testPos, radius: radius) {
                        return testPos
                    }
                }
            }
        }
        
        return nil
    }
    
    public func isPathCell(cellPos: (Int, Int)) -> Bool {
        let (cellX, cellY) = cellPos
        
        var chunkX = cellX / chunkSize
        var chunkY = cellY / chunkSize
        if cellX < 0 {
            chunkX = (cellX - chunkSize + 1) / chunkSize
        }
        if cellY < 0 {
            chunkY = (cellY - chunkSize + 1) / chunkSize
        }
        
        let chunk = getOrCreateChunk(chunkX: chunkX, chunkY: chunkY)
        
        var localX = cellX % chunkSize
        var localY = cellY % chunkSize
        if cellX < 0 {
            localX = (cellX % chunkSize + chunkSize) % chunkSize
        }
        if cellY < 0 {
            localY = (cellY % chunkSize + chunkSize) % chunkSize
        }
        
        let cellKey = "\(localX),\(localY)"
        return chunk.paths.contains(cellKey) && !chunk.walls.contains(cellKey)
    }
}
