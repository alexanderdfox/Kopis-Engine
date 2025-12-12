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

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the engine:

```bash
python sword.py
```

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
from sword import KopisEngine, GameEntity

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

## Notes

- Automatically uses GPU if available (CUDA)
- Models are downloaded from Hugging Face on first use
- Designed for extensibility and customization
- Supports real-time game loop processing
