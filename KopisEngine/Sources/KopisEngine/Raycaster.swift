import Foundation

public struct RaycastResult {
    public let hit: Bool
    public let distance: Double
    public let side: Int // 0 = X-side, 1 = Y-side
    public let mapX: Int
    public let mapY: Int
    
    public init(hit: Bool, distance: Double, side: Int, mapX: Int, mapY: Int) {
        self.hit = hit
        self.distance = distance
        self.side = side
        self.mapX = mapX
        self.mapY = mapY
    }
}

public class Raycaster {
    private let maze: Maze
    
    public init(maze: Maze) {
        self.maze = maze
    }
    
    public func raycastWall(camX: Double, camY: Double, rayDirX: Double, rayDirY: Double, maxDepth: Double = 1000.0) -> RaycastResult? {
        let cellSize = maze.cellSize
        
        // Avoid division by zero
        var safeRayDirX = rayDirX
        var safeRayDirY = rayDirY
        if abs(safeRayDirX) < 0.0001 {
            safeRayDirX = safeRayDirX < 0 ? -0.0001 : 0.0001
        }
        if abs(safeRayDirY) < 0.0001 {
            safeRayDirY = safeRayDirY < 0 ? -0.0001 : 0.0001
        }
        
        // DDA algorithm
        var mapX = Int(camX / cellSize)
        var mapY = Int(camY / cellSize)
        
        let deltaDistX = abs(1.0 / safeRayDirX)
        let deltaDistY = abs(1.0 / safeRayDirY)
        
        var stepX: Int
        var stepY: Int
        var sideDistX: Double
        var sideDistY: Double
        
        if safeRayDirX < 0 {
            stepX = -1
            sideDistX = (camX / cellSize - Double(mapX)) * deltaDistX
        } else {
            stepX = 1
            sideDistX = (Double(mapX) + 1.0 - camX / cellSize) * deltaDistX
        }
        
        if safeRayDirY < 0 {
            stepY = -1
            sideDistY = (camY / cellSize - Double(mapY)) * deltaDistY
        } else {
            stepY = 1
            sideDistY = (Double(mapY) + 1.0 - camY / cellSize) * deltaDistY
        }
        
        // Perform DDA
        var hit = false
        var side = 0
        var perpWallDist = 0.0
        var steps = 0
        let maxSteps = Int(maxDepth / cellSize) + 1
        
        while !hit && steps < maxSteps {
            if sideDistX < sideDistY {
                sideDistX += deltaDistX
                mapX += stepX
                side = 0
            } else {
                sideDistY += deltaDistY
                mapY += stepY
                side = 1
            }
            steps += 1
            
            // Check if we hit a wall
            let worldPos = Vector3(x: Double(mapX) * cellSize, y: Double(mapY) * cellSize, z: 0)
            if maze.checkCollision(position: worldPos, radius: cellSize / 2.0) {
                hit = true
                if side == 0 {
                    perpWallDist = (Double(mapX) - camX / cellSize + Double(1 - stepX) / 2.0) / safeRayDirX
                } else {
                    perpWallDist = (Double(mapY) - camY / cellSize + Double(1 - stepY) / 2.0) / safeRayDirY
                }
            }
        }
        
        if hit {
            let distance = abs(perpWallDist) * cellSize
            return RaycastResult(hit: true, distance: distance, side: side, mapX: mapX, mapY: mapY)
        }
        
        return nil
    }
}
