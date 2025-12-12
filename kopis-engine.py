"""
Kopis Engine - Transformer-Based Game Engine
A game engine architecture based on transformer circuits with:
- Stacked transformers for state processing
- Parallel branches for multi-system processing
- NAND gate for logical operations
- Feedback loop for state persistence
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    pipeline
)
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class GameState(Enum):
    """Game state enumeration"""
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    VICTORY = "victory"


@dataclass
class GameEntity:
    """Represents a game entity (player, NPC, object)"""
    id: str
    position: Tuple[float, float] = (0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    health: float = 100.0
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Represents a signal with voltage/strength level"""
    value: Any
    voltage: float = 2.51  # Default voltage level
    timestamp: float = field(default_factory=time.time)


class StackedTransformers:
    """
    Top Section: Stacked transformers for sequential game state processing
    Each layer processes game state at increasing abstraction levels
    
    Note: Transformers are optimized for game use by:
    - Using CPU mode to prevent bus errors
    - Sharing a single model instance across layers
    - Providing graceful fallback when models fail
    """
    
    _shared_pipeline = None  # Class variable to share a single model instance
    
    @classmethod
    def cleanup(cls):
        """Clean up transformer resources to prevent memory leaks"""
        if cls._shared_pipeline is not None:
            try:
                # Clear pipeline and free memory
                del cls._shared_pipeline
                cls._shared_pipeline = None
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning during cleanup: {e}")
    
    def __init__(self, num_layers: int = 3, use_transformers: bool = False):
        self.layers = []
        self.num_layers = num_layers
        self.use_transformers = use_transformers
        self._initialize_layers()
    
    def _initialize_layers(self):
        """Initialize transformer layers for game state processing"""
        # Layer 1: Raw input processing
        # Layer 2: Input interpretation
        # Layer 3: High-level game logic
        
        # Try to load shared pipeline only once if transformers are enabled
        if self.use_transformers and StackedTransformers._shared_pipeline is None:
            try:
                print("Loading transformer model (this may take a moment)...")
                # Use CPU by default to avoid bus errors and resource conflicts
                model_id = 'distilbert-base-uncased-finetuned-sst-2-english'
                # Force CPU to prevent multiprocessing issues that cause bus errors
                StackedTransformers._shared_pipeline = pipeline(
                    'text-classification',
                    model=model_id,
                    device=-1,  # Always use CPU to avoid bus errors
                    torch_dtype=torch.float32
                )
                print("✓ Transformer model loaded successfully (CPU mode for stability)")
            except Exception as e:
                print(f"Warning: Could not load transformer model: {e}")
                print("Falling back to simple processing mode")
                self.use_transformers = False
                # Clean up any partial state
                StackedTransformers._shared_pipeline = None
        
        # Initialize layers (all use the same shared pipeline if transformers enabled)
        for i in range(self.num_layers):
            self.layers.append({
                'layer': i + 1,
                'pipeline': StackedTransformers._shared_pipeline if self.use_transformers else None,
                'name': f'Layer {i+1}: {"Input Processing" if i == 0 else "Interpretation" if i == 1 else "Game Logic"}'
            })
    
    def process(self, input_data: Dict[str, Any]) -> Signal:
        """
        Process game input through stacked transformers
        
        Args:
            input_data: Game input data (keys, mouse, game state)
            
        Returns:
            Processed signal with voltage level
        """
        current_state = json.dumps(input_data)
        voltage = 2.51
        
        for layer in self.layers:
            if layer['pipeline']:
                try:
                    # Process through transformer
                    result = layer['pipeline'](current_state[:512])  # Limit length
                    if isinstance(result, list) and len(result) > 0:
                        score = result[0].get('score', 0.5)
                        voltage = 2.51 * score  # Adjust voltage based on confidence
                        current_state = str(result[0].get('label', current_state))
                except Exception as e:
                    print(f"Error in layer {layer['layer']}: {e}")
            else:
                # Simple fallback processing
                current_state = f"Processed: {current_state[:100]}"
        
        return Signal(value=current_state, voltage=voltage)


class ParallelBranches:
    """
    Middle Section: Parallel processing branches for different game systems
    - Physics Engine
    - Rendering Pipeline
    - AI/NPC System
    """
    
    def __init__(self, use_transformers: bool = False):
        self.branches = {
            'physics': None,
            'rendering': None,
            'ai': None
        }
        self.use_transformers = use_transformers
        # Note: We don't actually need transformers for physics/rendering/AI
        # These are game logic systems, not NLP tasks
    
    def process_physics(self, entities: List[GameEntity], delta_time: float) -> List[GameEntity]:
        """
        Process physics simulation with enhanced features:
        - Gravity
        - Friction
        - Boundary collision
        - Velocity-based movement
        """
        updated_entities = []
        gravity = 9.8 * 50  # Scaled gravity (pixels per second squared)
        friction_coefficient = 0.95  # Friction factor (0.95 = 5% velocity loss per frame)
        world_bounds = {'min_x': 0, 'min_y': 0, 'max_x': 800, 'max_y': 600}
        
        for entity in entities:
            # Get current state
            x, y = entity.position
            vx, vy = entity.velocity
            
            # Apply gravity (if entity is affected by gravity)
            if entity.properties.get('affected_by_gravity', True):
                vy += gravity * delta_time
            
            # Apply friction
            vx *= friction_coefficient
            vy *= friction_coefficient
            
            # Update position based on velocity
            new_x = x + vx * delta_time
            new_y = y + vy * delta_time
            
            # Boundary collision detection (simple AABB)
            if new_x < world_bounds['min_x']:
                new_x = world_bounds['min_x']
                vx = -vx * 0.5  # Bounce with energy loss
            elif new_x > world_bounds['max_x']:
                new_x = world_bounds['max_x']
                vx = -vx * 0.5
            
            if new_y < world_bounds['min_y']:
                new_y = world_bounds['min_y']
                vy = -vy * 0.5  # Bounce with energy loss
            elif new_y > world_bounds['max_y']:
                new_y = world_bounds['max_y']
                vy = -vy * 0.5
            
            # Create updated entity
            updated_entities.append(GameEntity(
                id=entity.id,
                position=(new_x, new_y),
                velocity=(vx, vy),
                health=entity.health,
                description=entity.description,
                properties=entity.properties
            ))
        return updated_entities
    
    def process_rendering(self, entities: List[GameEntity]) -> Dict[str, Any]:
        """
        Process rendering data and generate ASCII visualization
        """
        render_data = {
            'entities': [],
            'camera': {'x': 0, 'y': 0, 'zoom': 1.0},
            'ascii_map': None
        }
        
        # Collect entity data
        for entity in entities:
            render_data['entities'].append({
                'id': entity.id,
                'x': entity.position[0],
                'y': entity.position[1],
                'health': entity.health
            })
        
        # Generate ASCII visualization
        render_data['ascii_map'] = self._generate_ascii_map(entities)
        
        return render_data
    
    def _generate_ascii_map(self, entities: List[GameEntity], width: int = 80, height: int = 24) -> str:
        """
        Generate ASCII text-based visualization of game world
        """
        # Create empty grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Scale factor to fit world coordinates into grid
        scale_x = width / 800.0
        scale_y = height / 600.0
        
        # Place entities on grid
        for entity in entities:
            grid_x = int(entity.position[0] * scale_x)
            grid_y = int(entity.position[1] * scale_y)
            
            # Clamp to grid bounds
            grid_x = max(0, min(width - 1, grid_x))
            grid_y = max(0, min(height - 1, grid_y))
            
            # Choose symbol based on entity type
            if entity.id == 'player':
                symbol = '@'
            elif 'npc' in entity.id:
                symbol = 'N'
            else:
                symbol = 'E'
            
            grid[grid_y][grid_x] = symbol
        
        # Convert grid to string
        lines = [''.join(row) for row in grid]
        return '\n'.join(lines)
    
    def process_ai(self, entities: List[GameEntity], player_entity: GameEntity) -> List[GameEntity]:
        """Process AI/NPC behavior"""
        updated_entities = []
        for entity in entities:
            if entity.id != player_entity.id:
                # Simple AI: move towards player
                dx = player_entity.position[0] - entity.position[0]
                dy = player_entity.position[1] - entity.position[1]
                distance = np.sqrt(dx**2 + dy**2)
                if distance > 0:
                    speed = 50.0  # pixels per second
                    vx = (dx / distance) * speed * 0.016  # Normalize for frame time
                    vy = (dy / distance) * speed * 0.016
                    updated_entities.append(GameEntity(
                        id=entity.id,
                        position=entity.position,
                        velocity=(vx, vy),
                        health=entity.health,
                        description=entity.description,
                        properties=entity.properties
                    ))
                else:
                    updated_entities.append(entity)
            else:
                updated_entities.append(entity)
        return updated_entities
    
    def process_parallel(self, signal: Signal, game_data: Dict[str, Any]) -> Dict[str, Signal]:
        """
        Process through all parallel branches simultaneously
        
        Returns:
            Dictionary of signals from each branch
        """
        entities = game_data.get('entities', [])
        delta_time = game_data.get('delta_time', 0.016)
        player_entity = game_data.get('player', None)
        
        # Process all branches in parallel
        physics_result = self.process_physics(entities, delta_time)
        rendering_result = self.process_rendering(entities)
        ai_result = self.process_ai(entities, player_entity) if player_entity else entities
        
        return {
            'physics': Signal(value=physics_result, voltage=signal.voltage),
            'rendering': Signal(value=rendering_result, voltage=signal.voltage),
            'ai': Signal(value=ai_result, voltage=signal.voltage)
        }


class NANDGate:
    """
    Bottom Section: NAND gate for logical operations
    Implements game rules, conditions, and state transitions
    """
    
    def __init__(self):
        self.conditions = []
        self.rules = {}
    
    def add_rule(self, name: str, condition_func):
        """Add a game rule/condition"""
        self.rules[name] = condition_func
    
    def nand_operation(self, input_a: bool, input_b: bool) -> bool:
        """NAND gate: returns False only if both inputs are True"""
        return not (input_a and input_b)
    
    def evaluate(self, signals: Dict[str, Signal], game_state: Dict[str, Any]) -> Signal:
        """
        Evaluate game conditions using NAND logic
        
        Args:
            signals: Signals from parallel branches
            game_state: Current game state
            
        Returns:
            Result signal with game condition outcomes
        """
        # Extract boolean values from signals
        physics_active = signals['physics'].voltage > 1.0
        rendering_active = signals['rendering'].voltage > 1.0
        ai_active = signals['ai'].voltage > 1.0
        
        # Apply NAND logic for game conditions
        # Example: Game continues if NOT (physics failed AND rendering failed)
        game_continues = self.nand_operation(not physics_active, not rendering_active)
        
        # Check win/lose conditions
        entities = game_state.get('entities', [])
        player = game_state.get('player', None)
        
        win_condition = False
        lose_condition = False
        
        if player:
            # Win if player health > 0 and all enemies defeated
            enemies = [e for e in entities if e.id != player.id]
            win_condition = player.health > 0 and len(enemies) == 0
            
            # Lose if player health <= 0
            lose_condition = player.health <= 0
        
        # Combine conditions
        result = {
            'game_continues': game_continues,
            'win': win_condition,
            'lose': lose_condition,
            'physics_active': physics_active,
            'rendering_active': rendering_active,
            'ai_active': ai_active
        }
        
        # Calculate output voltage based on conditions
        voltage = 2.51 if game_continues else 0.0
        
        return Signal(value=result, voltage=voltage)


class FeedbackLoop:
    """
    Inductor Section: Feedback loop for state persistence
    Maintains game state across frames and enables memory
    """
    
    def __init__(self):
        self.state_history = []
        self.max_history = 100
        self.persistent_state = {}
    
    def update(self, signal: Signal, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update persistent state with feedback from current frame
        
        Args:
            signal: Current frame signal
            game_state: Current game state
            
        Returns:
            Updated persistent state
        """
        # Store in history
        self.state_history.append({
            'signal': signal,
            'game_state': game_state.copy(),
            'timestamp': time.time()
        })
        
        # Limit history size
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        # Update persistent state
        self.persistent_state.update({
            'last_frame_time': time.time(),
            'total_frames': len(self.state_history),
            'average_voltage': np.mean([s['signal'].voltage for s in self.state_history[-10:]])
        })
        
        return self.persistent_state
    
    def get_state(self) -> Dict[str, Any]:
        """Get current persistent state"""
        return self.persistent_state.copy()


class KopisEngine:
    """
    Main Game Engine - Implements the complete transformer circuit architecture
    """
    
    def __init__(self, use_transformers: bool = False):
        """
        Initialize the Kopis Engine
        
        Args:
            use_transformers: If True, loads transformer models (may cause resource issues).
                             If False, uses simple processing (recommended for stability).
        """
        print("Initializing Kopis Engine...")
        self.stacked_transformers = StackedTransformers(num_layers=3, use_transformers=use_transformers)
        self.parallel_branches = ParallelBranches(use_transformers=use_transformers)
        self.nand_gate = NANDGate()
        self.feedback_loop = FeedbackLoop()
        
        # Game state
        self.game_state = GameState.MENU
        self.entities = []
        self.player = None
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Initialize game rules
        self._initialize_rules()
        
        print("✓ Kopis Engine initialized successfully")
    
    def _initialize_rules(self):
        """Initialize game rules and conditions"""
        self.nand_gate.add_rule('win_condition', lambda state: state.get('win', False))
        self.nand_gate.add_rule('lose_condition', lambda state: state.get('lose', False))
    
    def add_entity(self, entity: GameEntity):
        """Add an entity to the game"""
        self.entities.append(entity)
        if entity.id == 'player':
            self.player = entity
    
    def process_frame(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single game frame through the complete circuit
        
        Args:
            input_data: Input data (keys, mouse, etc.)
            
        Returns:
            Complete frame processing results
        """
        self.frame_count += 1
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Prepare game data
        game_data = {
            'entities': self.entities,
            'player': self.player,
            'delta_time': delta_time,
            'frame': self.frame_count
        }
        
        # === STACKED TRANSFORMERS (Top Section) ===
        input_signal = self.stacked_transformers.process(input_data)
        
        # === PARALLEL BRANCHES (Middle Section) ===
        parallel_signals = self.parallel_branches.process_parallel(input_signal, game_data)
        
        # Update entities from physics and AI branches
        if 'physics' in parallel_signals:
            self.entities = parallel_signals['physics'].value
        
        if 'ai' in parallel_signals:
            self.entities = parallel_signals['ai'].value
        
        # === NAND GATE (Logical Operations) ===
        nand_result = self.nand_gate.evaluate(parallel_signals, {
            'entities': self.entities,
            'player': self.player,
            'game_state': self.game_state
        })
        
        # Update game state based on NAND gate results
        if nand_result.value.get('win'):
            self.game_state = GameState.VICTORY
        elif nand_result.value.get('lose'):
            self.game_state = GameState.GAME_OVER
        elif nand_result.value.get('game_continues'):
            if self.game_state == GameState.PLAYING:
                pass  # Continue playing
            else:
                self.game_state = GameState.PLAYING
        
        # === FEEDBACK LOOP (State Persistence) ===
        persistent_state = self.feedback_loop.update(nand_result, {
            'entities': self.entities,
            'player': self.player,
            'game_state': self.game_state,
            'frame': self.frame_count
        })
        
        # Return complete frame results
        return {
            'frame': self.frame_count,
            'delta_time': delta_time,
            'input_signal': {
                'value': input_signal.value,
                'voltage': input_signal.voltage
            },
            'parallel_branches': {
                'physics': {
                    'entities_count': len(parallel_signals['physics'].value),
                    'voltage': parallel_signals['physics'].voltage
                },
                'rendering': parallel_signals['rendering'].value,
                'ai': {
                    'entities_count': len(parallel_signals['ai'].value),
                    'voltage': parallel_signals['ai'].voltage
                }
            },
            'nand_gate': {
                'result': nand_result.value,
                'voltage': nand_result.voltage
            },
            'game_state': self.game_state.value,
            'persistent_state': persistent_state
        }
    
    def visualize_circuit(self):
        """Print visual representation of the engine circuit"""
        print("\n" + "="*60)
        print("KOPIS ENGINE CIRCUIT DIAGRAM")
        print("="*60)
        print("TOP SECTION: Stacked Transformers")
        for i in range(3):
            print(f"  Layer {i+1}: {'Input Processing' if i == 0 else 'Interpretation' if i == 1 else 'Game Logic'}")
            if i < 2:
                print("    ↓")
        print("\nMIDDLE SECTION: Parallel Branches")
        print("  ├─ Physics Engine")
        print("  ├─ Rendering Pipeline")
        print("  └─ AI/NPC System")
        print("\nBOTTOM SECTION: NAND Gate")
        print("  └─ Logical Operations & Game Rules")
        print("\nFEEDBACK LOOP: State Persistence")
        print("  └─ Memory & State History")
        print("="*60 + "\n")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'frame_count': self.frame_count,
            'entity_count': len(self.entities),
            'game_state': self.game_state.value,
            'persistent_state': self.feedback_loop.get_state()
        }
    
    def cleanup(self):
        """Clean up engine resources"""
        StackedTransformers.cleanup()
        print("✓ Engine resources cleaned up")
    
    def display_npc_info(self):
        """Display information about all NPCs in the game"""
        npcs = [entity for entity in self.entities if entity.id != 'player']
        
        if not npcs:
            print("  NPCs: None")
            return
        
        print(f"  NPCs ({len(npcs)}):")
        for npc in npcs:
            print(f"    - {npc.id}:")
            print(f"      Position: ({npc.position[0]:.2f}, {npc.position[1]:.2f})")
            print(f"      Velocity: ({npc.velocity[0]:.2f}, {npc.velocity[1]:.2f})")
            print(f"      Health: {npc.health:.1f}")
            if npc.description:
                print(f"      Description: {npc.description}")
            if npc.properties:
                print(f"      Properties: {npc.properties}")
    
    def display_rendering(self, render_data: Dict[str, Any]):
        """Display ASCII rendering output"""
        if render_data.get('ascii_map'):
            print("\n  ASCII World View:")
            print("  " + "-" * 80)
            for line in render_data['ascii_map'].split('\n'):
                print("  " + line)
            print("  " + "-" * 80)
            print("  Legend: @ = Player, N = NPC, E = Entity")


def main():
    """Example usage of the Kopis Engine"""
    # Initialize engine
    engine = KopisEngine()
    engine.visualize_circuit()
    
    # Create player entity
    player = GameEntity(
        id='player',
        position=(100.0, 100.0),
        velocity=(0.0, 0.0),
        health=100.0,
        description="Player character"
    )
    engine.add_entity(player)
    
    # Create some NPCs
    for i in range(3):
        npc = GameEntity(
            id=f'npc_{i}',
            position=(200.0 + i * 50, 200.0 + i * 50),
            velocity=(0.0, 0.0),
            health=50.0,
            description=f"NPC {i+1}"
        )
        engine.add_entity(npc)
    
    # Simulate game loop
    print("\nRunning game simulation...")
    print("="*60)
    
    for frame in range(10):
        input_data = {
            'keys': {'w': False, 'a': False, 's': False, 'd': False},
            'mouse': {'x': 0, 'y': 0, 'clicked': False}
        }
        
        result = engine.process_frame(input_data)
        
        print(f"\nFrame {frame + 1}:")
        print(f"  Game State: {result['game_state']}")
        print(f"  Input Voltage: {result['input_signal']['voltage']:.2f}V")
        print(f"  Entities: {result['parallel_branches']['physics']['entities_count']}")
        print(f"  NAND Gate Voltage: {result['nand_gate']['voltage']:.2f}V")
        print(f"  Game Continues: {result['nand_gate']['result']['game_continues']}")
        
        # Display NPC information
        engine.display_npc_info()
        
        # Display rendering output (ASCII visualization)
        if 'rendering' in result['parallel_branches']:
            engine.display_rendering(result['parallel_branches']['rendering'])
        
        time.sleep(0.1)  # Simulate frame delay
    
    # Print final stats
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    stats = engine.get_stats()
    print(json.dumps(stats, indent=2))
    print("="*60)
    
    # Cleanup resources
    engine.cleanup()


if __name__ == "__main__":
    main()
