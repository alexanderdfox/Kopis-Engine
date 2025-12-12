import Foundation

public class GameOfLife {
    public let width: Int
    public let height: Int
    private var grid: [[Bool]]
    private var frameCount: Int = 0
    
    public init(seed: Int? = nil) {
        self.width = 100
        self.height = 100
        self.grid = Array(repeating: Array(repeating: false, count: width), count: height)
        
        // Initialize with blood-dripping pattern
        var rng: SystemRandomNumberGenerator
        if let seed = seed {
            rng = SystemRandomNumberGenerator(seed: UInt64(seed))
        } else {
            rng = SystemRandomNumberGenerator()
        }
        
        for y in 0..<height {
            for x in 0..<width {
                let topProbability = 0.4
                let bottomProbability = 0.1
                let probability = topProbability - (topProbability - bottomProbability) * Double(y) / Double(height)
                
                if Double.random(in: 0...1, using: &rng) < probability {
                    grid[y][x] = true
                }
            }
        }
    }
    
    public func update() {
        var newGrid = grid
        
        // Count neighbors
        for y in 0..<height {
            for x in 0..<width {
                var neighbors = 0
                for dy in -1...1 {
                    for dx in -1...1 {
                        if dx == 0 && dy == 0 { continue }
                        let ny = (y + dy + height) % height
                        let nx = (x + dx + width) % width
                        if grid[ny][nx] {
                            neighbors += 1
                        }
                    }
                }
                
                // Apply Conway's rules
                if grid[y][x] {
                    newGrid[y][x] = neighbors == 2 || neighbors == 3
                } else {
                    newGrid[y][x] = neighbors == 3
                }
            }
        }
        
        // Add gravity effect (blood dripping)
        frameCount += 1
        for y in (1..<height).reversed() {
            for x in 0..<width {
                if newGrid[y-1][x] && !newGrid[y][x] {
                    if (x * height + y + frameCount) % 10 < 1 {
                        newGrid[y][x] = true
                        newGrid[y-1][x] = false
                    }
                }
            }
        }
        
        grid = newGrid
    }
    
    public func getPattern() -> [[Bool]] {
        return grid
    }
}

// Simple seeded random number generator
struct SystemRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64
    
    init(seed: UInt64 = UInt64.random(in: 0...UInt64.max)) {
        self.state = seed
    }
    
    mutating func next() -> UInt64 {
        state = state &* 1103515245 &+ 12345
        return state
    }
}
