# Kopis Engine - Transformer-Based Game Engine

A game engine architecture based on transformer circuits, implementing the circuit structure from the kopis-engine diagram with stacked transformers, parallel branches, NAND gates, and feedback loops.

## Architecture

The engine follows a transformer circuit architecture:

### Top Section: Stacked Transformers
- **Layer 1**: Raw input processing (keyboard, mouse, controller)
- **Layer 2**: Input interpretation (actions, intentions)
- **Layer 3**: High-level game logic (player decisions, AI reasoning)

### Middle Section: Parallel Branches
- **Physics Engine**: Collision detection, movement, forces
- **Rendering Pipeline**: Graphics processing, visual output
- **AI/NPC System**: NPC behavior, pathfinding, decision-making

### Bottom Section: NAND Gate
- **Logical Operations**: Game rules, win/lose conditions, state transitions
- **Condition Evaluation**: Combines signals from parallel branches

### Feedback Loop: State Persistence
- **Memory System**: Maintains game state across frames
- **State History**: Tracks frame-by-frame changes
- **Persistent Data**: Save/load system support

## Project Structure

The project is organized into the following directories:
- **`KopisEngine/`** - Swift Package (main project)
- **`scripts/`** - Utility scripts for building and development
- **`docs/`** - All documentation files
- **Root** - Main files (Python version, web version, README, etc.)

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed structure information.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Python Version
```bash
python3 kopis_engine.py
# Or:
make python
```

### Swift GUI Version (Recommended)
```bash
make gui
# Or open in Xcode:
make xcode
# Then select KopisEngineGUI scheme and press ⌘R
```

### Swift CLI Version
```bash
make run
```

### Web Version
Open `index.html` in a web browser

## Features

### Game Entities
- Create and manage game entities (players, NPCs, objects)
- Position, velocity, health tracking
- Custom properties system

### Game Loop
- Frame-based processing through the complete circuit
- Delta time calculation
- State management

### Example Usage

```python
from kopis_engine import KopisEngine, GameEntity

# Initialize engine
engine = KopisEngine()

# Create player
player = GameEntity(
    id='player',
    position=(100.0, 100.0),
    health=100.0
)
engine.add_entity(player)

# Process game frame
input_data = {
    'keys': {'w': True, 'a': False},
    'mouse': {'x': 100, 'y': 200}
}
result = engine.process_frame(input_data)
```

## Circuit Flow

```
Input Data
    ↓
[Stacked Transformers] → Process Game State
    ↓
[Parallel Branches]
    ├─ Physics → Entity Updates
    ├─ Rendering → Visual Data
    └─ AI → NPC Behavior
    ↓
[NAND Gate] → Evaluate Conditions
    ↓
[Feedback Loop] → Update Persistent State
    ↓
Next Frame
```

## Components

- **StackedTransformers**: Sequential state processing
- **ParallelBranches**: Multi-system parallel processing
- **NANDGate**: Logical operations and game rules
- **FeedbackLoop**: State persistence and memory
- **KopisEngine**: Main engine orchestrating all components

## Customization

- Add custom game rules via `nand_gate.add_rule()`
- Extend entity properties in `GameEntity`
- Modify parallel branch processing logic
- Adjust transformer layers for different abstraction levels

## Platforms

- **Python** - Cross-platform with Pygame
- **Swift/macOS** - Native macOS GUI with Metal 4 rendering
- **Web** - Browser-based with HTML5 Canvas

## Features

### Swift GUI Version
- ✅ Metal 4 GPU-accelerated raycasting
- ✅ SwiftUI + AppKit integration
- ✅ Sound system with AVFoundation
- ✅ Fullscreen support
- ✅ Mouse capture for FPV controls
- ✅ Game of Life blood patterns
- ✅ Entity billboard rendering

### Python Version
- ✅ Pygame-based rendering
- ✅ Sound effects
- ✅ Fullscreen support
- ✅ 3D raycasting

### Web Version
- ✅ HTML5 Canvas rendering
- ✅ Browser-based gameplay
- ✅ 3D raycasting

## Documentation

See the [docs/](docs/) directory for detailed documentation:
- [Quick Start Guide](docs/QUICK_START.md)
- [Xcode Setup](docs/XCODE_SETUP.md)
- [Metal 4 Setup](docs/METAL4_SETUP.md)
- [Project Structure](PROJECT_STRUCTURE.md)

## Notes

- Swift version requires macOS 13.0+ and Xcode 14.0+
- Python version requires pygame (install with `pip install -r requirements.txt`)
- Web version works in modern browsers
- Designed for extensibility and customization
- Supports real-time game loop processing
