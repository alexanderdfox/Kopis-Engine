import Foundation

struct ChunkKey: Hashable {
    let x: Int
    let y: Int
}

struct ConnectionPoints {
    var north: [(Int, Int)] = []
    var south: [(Int, Int)] = []
    var east: [(Int, Int)] = []
    var west: [(Int, Int)] = []
}

class MazeChunk {
    let chunkX: Int
    let chunkY: Int
    let chunkSize: Int
    let cellSize: Double
    var walls: Set<String> = [] // "x,y" format
    var paths: Set<String> = []
    var connectionPoints: ConnectionPoints
    
    init(chunkX: Int, chunkY: Int, chunkSize: Int, cellSize: Double, adjacentChunks: [String: MazeChunk]? = nil) {
        self.chunkX = chunkX
        self.chunkY = chunkY
        self.chunkSize = chunkSize
        self.cellSize = cellSize
        self.connectionPoints = ConnectionPoints()
        self.connectionPoints = getConnectionPoints()
        generateChunk(adjacentChunks: adjacentChunks ?? [:])
    }
    
    private func getConnectionPoints() -> ConnectionPoints {
        let seed = hashString("\(chunkX)_\(chunkY)")
        var rng = SeededRandom(seed: UInt64(seed))
        
        var connections = ConnectionPoints()
        let width = chunkSize
        let height = chunkSize
        
        // North edge
        let northCount = Int.random(in: 1...3, using: &rng)
        let northPoints = (1..<width-1).shuffled(using: &rng).prefix(northCount).sorted()
        connections.north = northPoints.map { ($0, 0) }
        
        // South edge
        let southCount = Int.random(in: 1...3, using: &rng)
        let southPoints = (1..<width-1).shuffled(using: &rng).prefix(southCount).sorted()
        connections.south = southPoints.map { ($0, height - 1) }
        
        // West edge
        let westCount = Int.random(in: 1...3, using: &rng)
        let westPoints = (1..<height-1).shuffled(using: &rng).prefix(westCount).sorted()
        connections.west = westPoints.map { (0, $0) }
        
        // East edge
        let eastCount = Int.random(in: 1...3, using: &rng)
        let eastPoints = (1..<height-1).shuffled(using: &rng).prefix(eastCount).sorted()
        connections.east = eastPoints.map { (width - 1, $0) }
        
        return connections
    }
    
    private func getMatchingConnectionPoints(direction: String, adjacentChunk: MazeChunk) -> [(Int, Int)] {
        let width = chunkSize
        let height = chunkSize
        
        switch direction {
        case "north":
            return adjacentChunk.connectionPoints.south.map { ($0.0, 0) }
        case "south":
            return adjacentChunk.connectionPoints.north.map { ($0.0, height - 1) }
        case "west":
            return adjacentChunk.connectionPoints.east.map { (0, $0.1) }
        case "east":
            return adjacentChunk.connectionPoints.west.map { (width - 1, $0.1) }
        default:
            return []
        }
    }
    
    private func generateChunk(adjacentChunks: [String: MazeChunk]) {
        let seed = hashString("\(chunkX)_\(chunkY)")
        var rng = SeededRandom(seed: UInt64(seed))
        
        let width = chunkSize
        let height = chunkSize
        
        // Initialize all cells as walls
        var grid = Array(repeating: Array(repeating: 1, count: width), count: height)
        
        // Collect connection points
        var connectionCells: Set<String> = []
        
        // Helper to convert tuple to string key
        func tupleToKey(_ tuple: (Int, Int)) -> String {
            return "\(tuple.0),\(tuple.1)"
        }
        
        func keyToTuple(_ key: String) -> (Int, Int)? {
            let parts = key.split(separator: ",")
            guard parts.count == 2,
                  let x = Int(parts[0]),
                  let y = Int(parts[1]) else {
                return nil
            }
            return (x, y)
        }
        
        if let north = adjacentChunks["north"] {
            let points = getMatchingConnectionPoints(direction: "north", adjacentChunk: north)
            connectionCells.formUnion(Set(points.map(tupleToKey)))
        } else {
            connectionCells.formUnion(Set(connectionPoints.north.map(tupleToKey)))
        }
        
        if let south = adjacentChunks["south"] {
            let points = getMatchingConnectionPoints(direction: "south", adjacentChunk: south)
            connectionCells.formUnion(Set(points.map(tupleToKey)))
        } else {
            connectionCells.formUnion(Set(connectionPoints.south.map(tupleToKey)))
        }
        
        if let west = adjacentChunks["west"] {
            let points = getMatchingConnectionPoints(direction: "west", adjacentChunk: west)
            connectionCells.formUnion(Set(points.map(tupleToKey)))
        } else {
            connectionCells.formUnion(Set(connectionPoints.west.map(tupleToKey)))
        }
        
        if let east = adjacentChunks["east"] {
            let points = getMatchingConnectionPoints(direction: "east", adjacentChunk: east)
            connectionCells.formUnion(Set(points.map(tupleToKey)))
        } else {
            connectionCells.formUnion(Set(connectionPoints.east.map(tupleToKey)))
        }
        
        // Start from first connection point or (1, 1)
        var stack: [(Int, Int)] = []
        if let firstKey = connectionCells.first,
           let start = keyToTuple(firstKey) {
            stack.append(start)
            grid[start.1][start.0] = 0
        } else {
            // Use a valid starting position (avoid edges)
            let startX = max(1, min(width - 2, width / 2))
            let startY = max(1, min(height - 2, height / 2))
            stack.append((startX, startY))
            grid[startY][startX] = 0
        }
        
        // Ensure all connection points are paths
        for key in connectionCells {
            if let (x, y) = keyToTuple(key) {
                if x >= 0 && x < width && y >= 0 && y < height {
                    grid[y][x] = 0
                }
            }
        }
        
        // Directions: up, right, down, left (step by 2)
        let directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        
        // Generate maze with recursive backtracking
        // Ensure we have enough paths by continuing until stack is empty
        var maxIterations = width * height * 2 // Prevent infinite loops
        var iterations = 0
        
        while !stack.isEmpty && iterations < maxIterations {
            iterations += 1
            let current = stack.last!
            let (x, y) = current
            
            // Find unvisited neighbors (walls that can be carved)
            var neighbors: [(Int, Int, Int, Int)] = []
            for (dx, dy) in directions {
                let nx = x + dx
                let ny = y + dy
                // Check bounds and ensure it's a wall
                if nx >= 0 && nx < width && ny >= 0 && ny < height && grid[ny][nx] == 1 {
                    // Calculate wall position between current and neighbor
                    let wallX = x + dx / 2
                    let wallY = y + dy / 2
                    neighbors.append((nx, ny, wallX, wallY))
                }
            }
            
            if !neighbors.isEmpty {
                let neighbor = neighbors.randomElement(using: &rng)!
                let (nx, ny, wallX, wallY) = neighbor
                
                // Carve path to neighbor
                grid[ny][nx] = 0
                // Carve wall between current and neighbor
                if wallX >= 0 && wallX < width && wallY >= 0 && wallY < height {
                    grid[wallY][wallX] = 0
                }
                
                stack.append((nx, ny))
            } else {
                // Backtrack
                stack.removeLast()
            }
        }
        
        // Ensure connection points are still paths (they might have been overwritten)
        for key in connectionCells {
            if let (x, y) = keyToTuple(key) {
                if x >= 0 && x < width && y >= 0 && y < height {
                    grid[y][x] = 0
                }
            }
        }
        
        // Build walls and paths sets
        for y in 0..<height {
            for x in 0..<width {
                let key = "\(x),\(y)"
                if grid[y][x] == 1 {
                    walls.insert(key)
                } else {
                    paths.insert(key)
                }
            }
        }
    }
    
    private func hashString(_ str: String) -> Int {
        var hash = 0
        for char in str.utf8 {
            hash = ((hash << 5) - hash) + Int(char)
            hash = hash & hash
        }
        return abs(hash)
    }
}

// Seeded random number generator
struct SeededRandom: RandomNumberGenerator {
    private var state: UInt64
    
    init(seed: UInt64) {
        self.state = seed
    }
    
    mutating func next() -> UInt64 {
        state = state &* 1103515245 &+ 12345
        return state
    }
}
