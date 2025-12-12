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
import sys
import select
import tty
import termios

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Install with: pip install pygame")


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
    
    def __init__(self, use_transformers: bool = False, maze: Optional['Maze'] = None):
        self.branches = {
            'physics': None,
            'rendering': None,
            'ai': None
        }
        self.use_transformers = use_transformers
        self.maze = maze
        # Note: We don't actually need transformers for physics/rendering/AI
        # These are game logic systems, not NLP tasks
    
    def process_physics(self, entities: List[GameEntity], delta_time: float, maze: Optional['Maze'] = None) -> List[GameEntity]:
        """
        Process physics simulation with enhanced features:
        - Gravity
        - Friction
        - Infinite world (no boundaries)
        - Velocity-based movement
        - Maze collision detection
        """
        updated_entities = []
        gravity = 9.8 * 50  # Scaled gravity (pixels per second squared)
        friction_coefficient = 0.95  # Friction factor (0.95 = 5% velocity loss per frame)
        
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
            
            # Update position based on velocity (infinite world - no bounds)
            new_x = x + vx * delta_time
            new_y = y + vy * delta_time
            
            # Check maze collision if maze exists
            entity_radius = entity.properties.get('radius', 10.0)
            if maze:
                # Try new position
                if maze.check_collision((new_x, new_y), entity_radius):
                    # Collision detected - try moving only X or only Y
                    if not maze.check_collision((new_x, y), entity_radius):
                        new_y = y  # Keep old Y, allow X movement
                        vx = 0  # Stop X velocity
                    elif not maze.check_collision((x, new_y), entity_radius):
                        new_x = x  # Keep old X, allow Y movement
                        vy = 0  # Stop Y velocity
                    else:
                        # Can't move in either direction
                        new_x = x
                        new_y = y
                        vx = 0
                        vy = 0
            
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
    
    def process_rendering(self, entities: List[GameEntity], camera_pos: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, Any]:
        """
        Process rendering data and generate ASCII visualization
        Uses camera position for viewport rendering in infinite world
        """
        render_data = {
            'entities': [],
            'camera': {'x': camera_pos[0], 'y': camera_pos[1], 'zoom': 1.0},
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
        
        # Generate ASCII visualization with camera offset
        render_data['ascii_map'] = self._generate_ascii_map(entities, camera_pos)
        
        return render_data
    
    def _generate_ascii_map(self, entities: List[GameEntity], camera_pos: Tuple[float, float] = (0.0, 0.0), width: int = 80, height: int = 24, viewport_size: float = 800.0) -> str:
        """
        Generate ASCII text-based visualization of game world
        Uses camera position to render viewport in infinite world
        """
        # Create empty grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Calculate viewport bounds (centered on camera)
        viewport_half_width = viewport_size / 2.0
        viewport_half_height = viewport_size / 2.0
        viewport_min_x = camera_pos[0] - viewport_half_width
        viewport_max_x = camera_pos[0] + viewport_half_width
        viewport_min_y = camera_pos[1] - viewport_half_height
        viewport_max_y = camera_pos[1] + viewport_half_height
        
        # Scale factor to convert world coordinates to grid coordinates
        scale_x = width / viewport_size
        scale_y = height / viewport_size
        
        # Place entities on grid (only if within viewport)
        for entity in entities:
            entity_x, entity_y = entity.position
            
            # Check if entity is within viewport
            if (viewport_min_x <= entity_x <= viewport_max_x and 
                viewport_min_y <= entity_y <= viewport_max_y):
                
                # Convert world coordinates to grid coordinates relative to camera
                grid_x = int((entity_x - viewport_min_x) * scale_x)
                grid_y = int((entity_y - viewport_min_y) * scale_y)
                
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
    
    def process_ai(self, entities: List[GameEntity], player_entity: GameEntity, delta_time: float, maze: Optional['Maze'] = None) -> List[GameEntity]:
        """
        Process AI/NPC behavior - automatic movement with pathfinding
        NPCs use A* pathfinding to navigate through the maze towards the player
        Optimized for performance: staggered pathfinding updates and distance-based culling
        """
        updated_entities = []
        
        # Get frame count for staggering pathfinding updates
        frame_count = getattr(self, '_ai_frame_count', 0)
        self._ai_frame_count = frame_count + 1
        
        # Performance optimization: only update NPCs within reasonable distance
        MAX_UPDATE_DISTANCE = 2000.0  # Only update NPCs within 2000 pixels
        PATHFINDING_BATCH_SIZE = 3  # Update max 3 NPCs' paths per frame
        
        npcs_to_update = []
        for entity in entities:
            if entity.id != player_entity.id and 'npc' in entity.id:
                # Distance-based culling
                dx = entity.position[0] - player_entity.position[0]
                dy = entity.position[1] - player_entity.position[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > MAX_UPDATE_DISTANCE:
                    # Too far, skip update (keep current state)
                    updated_entities.append(entity)
                    continue
                
                npcs_to_update.append((entity, distance))
            else:
                # Keep non-NPC entities unchanged (player, etc.)
                updated_entities.append(entity)
        
        # Sort by distance (closest first) for priority pathfinding
        npcs_to_update.sort(key=lambda x: x[1])
        
        # Stagger pathfinding: only update a few NPCs per frame
        pathfinding_count = 0
        
        for entity, distance in npcs_to_update:
            if not maze:
                # No maze, use simple movement
                updated_entities.append(entity)
                continue
            
            entity_radius = entity.properties.get('radius', 8.0)
            speed = 50.0  # pixels per second
            
            # Get or initialize pathfinding state
            path = entity.properties.get('path', None)
            path_index = entity.properties.get('path_index', 0)
            last_path_update = entity.properties.get('last_path_update', 0.0)
            pathfinding_frame = entity.properties.get('pathfinding_frame', -1)
            current_time = time.time()
            
            # Update path less frequently and stagger across frames
            update_path = False
            if path is None or len(path) == 0:
                # Only update if we haven't exceeded batch size
                if pathfinding_count < PATHFINDING_BATCH_SIZE:
                    update_path = True
                    pathfinding_count += 1
            elif path_index >= len(path):
                if pathfinding_count < PATHFINDING_BATCH_SIZE:
                    update_path = True
                    pathfinding_count += 1
            elif current_time - last_path_update > 3.0:  # Increased to 3 seconds
                # Stagger pathfinding updates across frames
                if pathfinding_frame < 0 or (frame_count - pathfinding_frame) % max(1, len(npcs_to_update) // PATHFINDING_BATCH_SIZE) == 0:
                    if pathfinding_count < PATHFINDING_BATCH_SIZE:
                        update_path = True
                        pathfinding_count += 1
                        entity.properties['pathfinding_frame'] = frame_count
            
            # Calculate path if needed (with distance-based search radius)
            if update_path:
                # Adaptive search radius based on distance
                search_radius = min(50, max(20, int(distance / 100)))
                path = maze.find_path(entity.position, player_entity.position, max_search_radius=search_radius)
                if path:
                    path_index = 0
                    entity.properties['path'] = path
                    entity.properties['path_index'] = 0
                    entity.properties['last_path_update'] = current_time
                else:
                    # No path found, try simple movement as fallback
                    path = None
                    entity.properties['path'] = None
            
            # Get current path from properties (may have been updated above)
            path = entity.properties.get('path', None)
            path_index = entity.properties.get('path_index', 0)
            
            # Move along path (NPCs move every frame, not just when path is updated)
            if path and path_index < len(path):
                # Get current target waypoint
                target_cell = path[path_index]
                target_world = maze.cell_to_world(target_cell)
                
                # Calculate direction to waypoint
                dx = target_world[0] - entity.position[0]
                dy = target_world[1] - entity.position[1]
                distance_to_waypoint = np.sqrt(dx**2 + dy**2)
                
                # If close enough to waypoint, move to next one
                if distance_to_waypoint < maze.cell_size * 0.3:  # Within 30% of cell size
                    path_index += 1
                    if path_index < len(path):
                        target_cell = path[path_index]
                        target_world = maze.cell_to_world(target_cell)
                        dx = target_world[0] - entity.position[0]
                        dy = target_world[1] - entity.position[1]
                        distance_to_waypoint = np.sqrt(dx**2 + dy**2)
                    # Update path_index in properties
                    entity.properties['path_index'] = path_index
                
                if distance_to_waypoint > 0.1:
                    # Calculate velocity towards waypoint
                    vx = (dx / distance_to_waypoint) * speed
                    vy = (dy / distance_to_waypoint) * speed
                    
                    # Update position
                    new_x = entity.position[0] + vx * delta_time
                    new_y = entity.position[1] + vy * delta_time
                    
                    # Check collision and adjust
                    if maze.check_collision((new_x, new_y), entity_radius):
                        # Try moving only X
                        if not maze.check_collision((new_x, entity.position[1]), entity_radius):
                            new_y = entity.position[1]
                        # Try moving only Y
                        elif not maze.check_collision((entity.position[0], new_y), entity_radius):
                            new_x = entity.position[0]
                        else:
                            # Can't move, stay in place and recalculate path
                            new_x = entity.position[0]
                            new_y = entity.position[1]
                            vx = 0
                            vy = 0
                            entity.properties['path'] = None  # Force path recalculation
                else:
                    # Already at waypoint
                    new_x = entity.position[0]
                    new_y = entity.position[1]
                    vx = 0
                    vy = 0
            else:
                # No path or path exhausted, use simple fallback movement
                dx = player_entity.position[0] - entity.position[0]
                dy = player_entity.position[1] - entity.position[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 0:
                    vx = (dx / distance) * speed
                    vy = (dy / distance) * speed
                    
                    new_x = entity.position[0] + vx * delta_time
                    new_y = entity.position[1] + vy * delta_time
                    
                    # Check collision
                    if maze.check_collision((new_x, new_y), entity_radius):
                        if not maze.check_collision((new_x, entity.position[1]), entity_radius):
                            new_y = entity.position[1]
                        elif not maze.check_collision((entity.position[0], new_y), entity_radius):
                            new_x = entity.position[0]
                        else:
                            new_x = entity.position[0]
                            new_y = entity.position[1]
                            vx = 0
                            vy = 0
                else:
                    new_x = entity.position[0]
                    new_y = entity.position[1]
                    vx = 0
                    vy = 0
            
            # Create updated entity with all pathfinding properties preserved
            updated_properties = entity.properties.copy()
            # Ensure pathfinding state is preserved
            updated_properties['path'] = entity.properties.get('path', None)
            updated_properties['path_index'] = entity.properties.get('path_index', 0)
            updated_properties['last_path_update'] = entity.properties.get('last_path_update', 0.0)
            if 'pathfinding_frame' in entity.properties:
                updated_properties['pathfinding_frame'] = entity.properties['pathfinding_frame']
            
            updated_entities.append(GameEntity(
                id=entity.id,
                position=(new_x, new_y),
                velocity=(vx, vy),
                health=entity.health,
                description=entity.description,
                properties=updated_properties
            ))
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
        camera_pos = game_data.get('camera_pos', (0.0, 0.0))
        
        # Process all branches in parallel
        physics_result = self.process_physics(entities, delta_time, self.maze)
        rendering_result = self.process_rendering(entities, camera_pos)
        ai_result = self.process_ai(entities, player_entity, delta_time, self.maze) if player_entity else entities
        
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
    
    def __init__(self, use_transformers: bool = False, maze: Optional['Maze'] = None, sound_manager: Optional['SoundManager'] = None):
        """
        Initialize the Kopis Engine
        
        Args:
            use_transformers: If True, loads transformer models (may cause resource issues).
                             If False, uses simple processing (recommended for stability).
            maze: Optional maze for collision detection
            sound_manager: Optional sound manager for sound effects
        """
        print("Initializing Kopis Engine...")
        self.stacked_transformers = StackedTransformers(num_layers=3, use_transformers=use_transformers)
        self.parallel_branches = ParallelBranches(use_transformers=use_transformers, maze=maze)
        self.nand_gate = NANDGate()
        self.feedback_loop = FeedbackLoop()
        self.maze = maze
        self.sound_manager = sound_manager
        
        # Game state
        self.game_state = GameState.MENU
        self.entities = []
        self.player = None
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.last_player_pos = None
        
        # Camera system for smooth scrolling
        self.camera_pos = (0.0, 0.0)
        self.camera_smoothness = 0.1  # Lower = smoother, higher = snappier (0.1 = smooth, 1.0 = instant)
        
        # Initialize game rules
        self._initialize_rules()
        
        print("✓ Kopis Engine initialized successfully")
    
    def _initialize_rules(self):
        """Initialize game rules and conditions"""
        self.nand_gate.add_rule('win_condition', lambda state: state.get('win', False))
        self.nand_gate.add_rule('lose_condition', lambda state: state.get('lose', False))
    
    def _process_player_input(self, input_data: Dict[str, Any], delta_time: float, sound_manager: Optional['SoundManager'] = None):
        """
        Process player input for movement
        WASD keys control player movement
        """
        if not self.player:
            return
        
        keys = input_data.get('keys', {})
        player_speed = 200.0  # pixels per second
        
        # Get current velocity
        vx, vy = self.player.velocity
        was_moving = abs(vx) > 0.1 or abs(vy) > 0.1
        
        # Reset velocity based on input
        vx = 0.0
        vy = 0.0
        
        # Process movement keys (check both lowercase and uppercase)
        if keys.get('w', False):
            vy -= player_speed
        if keys.get('s', False):
            vy += player_speed
        if keys.get('a', False):
            vx -= player_speed
        if keys.get('d', False):
            vx += player_speed
        
        # Normalize diagonal movement
        if vx != 0 and vy != 0:
            vx *= 0.707  # sqrt(2)/2 for diagonal normalization
            vy *= 0.707
        
        # Play sound effects
        if sound_manager:
            is_moving = abs(vx) > 0.1 or abs(vy) > 0.1
            if is_moving and not was_moving:
                # Just started moving
                sound_manager.play('move_start', volume=0.3)
            elif is_moving:
                # Continue moving - play footstep occasionally
                import random
                if random.random() < 0.05:  # 5% chance per frame
                    sound_manager.play('footstep', volume=0.2)
        
        # Update player velocity directly on the entity
        # We need to update the entity in the list, not just a reference
        for i, entity in enumerate(self.entities):
            if entity.id == 'player':
                # Create new entity with updated velocity
                updated_entity = GameEntity(
                    id=entity.id,
                    position=entity.position,
                    velocity=(vx, vy),
                    health=entity.health,
                    description=entity.description,
                    properties=entity.properties
                )
                self.entities[i] = updated_entity
                self.player = updated_entity
                break
    
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
            - keys: dict with 'w', 'a', 's', 'd' for movement
            - mouse: dict with mouse position and click info
            
        Returns:
            Complete frame processing results
        """
        self.frame_count += 1
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Process player input for movement
        if self.player:
            self._process_player_input(input_data, delta_time, self.sound_manager)
        
        # Update camera to smoothly follow player
        if self.player:
            target_camera_x = self.player.position[0]
            target_camera_y = self.player.position[1]
            
            # Smooth camera interpolation (lerp)
            current_camera_x, current_camera_y = self.camera_pos
            lerp_factor = min(1.0, self.camera_smoothness * (1.0 + delta_time * 60))  # Adjust for frame rate
            new_camera_x = current_camera_x + (target_camera_x - current_camera_x) * lerp_factor
            new_camera_y = current_camera_y + (target_camera_y - current_camera_y) * lerp_factor
            
            self.camera_pos = (new_camera_x, new_camera_y)
        else:
            self.camera_pos = (0.0, 0.0)
        
        camera_pos = self.camera_pos
        
        # Prepare game data
        game_data = {
            'entities': self.entities,
            'player': self.player,
            'delta_time': delta_time,
            'frame': self.frame_count,
            'camera_pos': camera_pos
        }
        
        # === STACKED TRANSFORMERS (Top Section) ===
        input_signal = self.stacked_transformers.process(input_data)
        
        # === PARALLEL BRANCHES (Middle Section) ===
        parallel_signals = self.parallel_branches.process_parallel(input_signal, game_data)
        
        # Update entities from physics and AI branches
        # Physics updates all entity positions and velocities
        # AI updates NPC positions directly (overriding physics for NPCs)
        if 'physics' in parallel_signals:
            physics_entities = parallel_signals['physics'].value
            # Update all entities from physics (positions and velocities)
            physics_dict = {e.id: e for e in physics_entities}
            for i, entity in enumerate(self.entities):
                if entity.id in physics_dict:
                    # Check for collision sound (player hit wall)
                    if entity.id == 'player' and self.sound_manager and self.last_player_pos:
                        new_entity = physics_dict[entity.id]
                        # If position didn't change but velocity was set, collision occurred
                        if (abs(new_entity.position[0] - self.last_player_pos[0]) < 0.1 and
                            abs(new_entity.position[1] - self.last_player_pos[1]) < 0.1 and
                            (abs(entity.velocity[0]) > 10 or abs(entity.velocity[1]) > 10)):
                            self.sound_manager.play('collision', volume=0.4)
                    
                    # Update position and velocity from physics
                    self.entities[i] = physics_dict[entity.id]
        
        if 'ai' in parallel_signals:
            # AI updates NPC positions directly (overrides physics for NPCs)
            ai_entities = parallel_signals['ai'].value
            ai_dict = {e.id: e for e in ai_entities}
            for i, entity in enumerate(self.entities):
                if entity.id in ai_dict and entity.id != 'player':
                    # Update NPC position from AI (NPCs move automatically)
                    self.entities[i] = ai_dict[entity.id]
        
        # Store player position for collision detection
        if self.player:
            self.last_player_pos = self.player.position
        
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


class SoundManager:
    """Manages sound effects for the game"""
    
    def __init__(self):
        if not PYGAME_AVAILABLE:
            self.enabled = False
            return
        
        self.enabled = True
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Generate simple sound effects programmatically
        self.sounds = {}
        self._generate_sounds()
    
    def _generate_sounds(self):
        """Generate simple sound effects using numpy and pygame"""
        try:
            import numpy as np
            
            # Footstep sound (short beep)
            duration = 0.1
            sample_rate = 22050
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2), dtype=np.int16)
            
            # Generate a simple tone for footsteps
            max_sample = 2**(16 - 1) - 1
            for i in range(frames):
                wave = 400.0 * (i / sample_rate)  # 400 Hz
                sample = int(max_sample * 0.3 * np.sin(wave * 2 * np.pi))
                arr[i] = [sample, sample]
            
            self.sounds['footstep'] = pygame.sndarray.make_sound(arr)
            
            # Collision sound (higher pitch beep)
            duration = 0.15
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2), dtype=np.int16)
            for i in range(frames):
                wave = 600.0 * (i / sample_rate)  # 600 Hz
                sample = int(max_sample * 0.4 * np.sin(wave * 2 * np.pi))
                arr[i] = [sample, sample]
            
            self.sounds['collision'] = pygame.sndarray.make_sound(arr)
            
            # Movement start sound (low tone)
            duration = 0.05
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2), dtype=np.int16)
            for i in range(frames):
                wave = 200.0 * (i / sample_rate)  # 200 Hz
                sample = int(max_sample * 0.2 * np.sin(wave * 2 * np.pi))
                arr[i] = [sample, sample]
            
            self.sounds['move_start'] = pygame.sndarray.make_sound(arr)
            
        except Exception as e:
            print(f"Warning: Could not generate sounds: {e}")
            self.enabled = False
    
    def play(self, sound_name: str, volume: float = 0.5):
        """Play a sound effect"""
        if not self.enabled or sound_name not in self.sounds:
            return
        
        try:
            sound = self.sounds[sound_name]
            sound.set_volume(volume)
            sound.play()
        except Exception as e:
            pass  # Silently fail if sound can't play
    
    def cleanup(self):
        """Clean up sound resources"""
        if self.enabled:
            pygame.mixer.quit()


class MazeChunk:
    """A single chunk of the infinite maze"""
    
    def __init__(self, chunk_x: int, chunk_y: int, chunk_size: int = 20, cell_size: float = 50.0, adjacent_chunks: Dict[str, 'MazeChunk'] = None):
        self.chunk_x = chunk_x
        self.chunk_y = chunk_y
        self.chunk_size = chunk_size
        self.cell_size = cell_size
        self.walls = set()  # Set of (local_x, local_y) tuples
        self.paths = set()
        self.connection_points = self._get_connection_points()
        if adjacent_chunks is None:
            adjacent_chunks = {}
        self._generate_chunk(adjacent_chunks)
    
    def _get_connection_points(self):
        """Get deterministic connection points for this chunk's edges"""
        import random
        import hashlib
        
        # Create deterministic seed from chunk coordinates
        seed_str = f"{self.chunk_x}_{self.chunk_y}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        
        connections = {}
        width = self.chunk_size
        height = self.chunk_size
        
        # Determine connection points for each edge (at least 1 per edge, up to 3)
        # North edge (y = 0)
        north_count = rng.randint(1, 3)
        north_points = sorted(rng.sample(range(1, width - 1), min(north_count, width - 2)))
        connections['north'] = [(x, 0) for x in north_points]
        
        # South edge (y = height - 1)
        south_count = rng.randint(1, 3)
        south_points = sorted(rng.sample(range(1, width - 1), min(south_count, width - 2)))
        connections['south'] = [(x, height - 1) for x in south_points]
        
        # West edge (x = 0)
        west_count = rng.randint(1, 3)
        west_points = sorted(rng.sample(range(1, height - 1), min(west_count, height - 2)))
        connections['west'] = [(0, y) for y in west_points]
        
        # East edge (x = width - 1)
        east_count = rng.randint(1, 3)
        east_points = sorted(rng.sample(range(1, height - 1), min(east_count, height - 2)))
        connections['east'] = [(width - 1, y) for y in east_points]
        
        return connections
    
    def _get_matching_connection_points(self, direction: str, adjacent_chunk: 'MazeChunk') -> List[Tuple[int, int]]:
        """Get connection points that match with an adjacent chunk"""
        width = self.chunk_size
        height = self.chunk_size
        
        if direction == 'north':
            # This chunk's north edge should match adjacent chunk's south edge
            adjacent_points = adjacent_chunk.connection_points.get('south', [])
            # Convert adjacent chunk's south points to this chunk's north points
            matching = []
            for adj_x, adj_y in adjacent_points:
                # Adjacent chunk's south edge at (adj_x, height-1) matches this chunk's north at (adj_x, 0)
                matching.append((adj_x, 0))
            return matching if matching else self.connection_points.get('north', [])
        
        elif direction == 'south':
            # This chunk's south edge should match adjacent chunk's north edge
            adjacent_points = adjacent_chunk.connection_points.get('north', [])
            matching = []
            for adj_x, adj_y in adjacent_points:
                # Adjacent chunk's north edge at (adj_x, 0) matches this chunk's south at (adj_x, height-1)
                matching.append((adj_x, height - 1))
            return matching if matching else self.connection_points.get('south', [])
        
        elif direction == 'west':
            # This chunk's west edge should match adjacent chunk's east edge
            adjacent_points = adjacent_chunk.connection_points.get('east', [])
            matching = []
            for adj_x, adj_y in adjacent_points:
                # Adjacent chunk's east edge at (width-1, adj_y) matches this chunk's west at (0, adj_y)
                matching.append((0, adj_y))
            return matching if matching else self.connection_points.get('west', [])
        
        elif direction == 'east':
            # This chunk's east edge should match adjacent chunk's west edge
            adjacent_points = adjacent_chunk.connection_points.get('west', [])
            matching = []
            for adj_x, adj_y in adjacent_points:
                # Adjacent chunk's west edge at (0, adj_y) matches this chunk's east at (width-1, adj_y)
                matching.append((width - 1, adj_y))
            return matching if matching else self.connection_points.get('east', [])
        
        return []
    
    def _generate_chunk(self, adjacent_chunks: Dict[str, 'MazeChunk'] = None):
        """Generate a maze chunk using deterministic algorithm, ensuring connections to adjacent chunks"""
        import random
        import hashlib
        
        if adjacent_chunks is None:
            adjacent_chunks = {}
        
        # Create deterministic seed from chunk coordinates
        seed_str = f"{self.chunk_x}_{self.chunk_y}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        
        width = self.chunk_size
        height = self.chunk_size
        
        # Initialize all cells as walls
        grid = [[1 for _ in range(width)] for _ in range(height)]
        
        # Collect all connection points (must be paths)
        connection_cells = set()
        
        # Add connection points from adjacent chunks (if they exist)
        if 'north' in adjacent_chunks:
            matching = self._get_matching_connection_points('north', adjacent_chunks['north'])
            connection_cells.update(matching)
        else:
            connection_cells.update(self.connection_points.get('north', []))
        
        if 'south' in adjacent_chunks:
            matching = self._get_matching_connection_points('south', adjacent_chunks['south'])
            connection_cells.update(matching)
        else:
            connection_cells.update(self.connection_points.get('south', []))
        
        if 'west' in adjacent_chunks:
            matching = self._get_matching_connection_points('west', adjacent_chunks['west'])
            connection_cells.update(matching)
        else:
            connection_cells.update(self.connection_points.get('west', []))
        
        if 'east' in adjacent_chunks:
            matching = self._get_matching_connection_points('east', adjacent_chunks['east'])
            connection_cells.update(matching)
        else:
            connection_cells.update(self.connection_points.get('east', []))
        
        # Start from first connection point, or (1, 1) if no connections
        if connection_cells:
            start_pos = list(connection_cells)[0]
            stack = [start_pos]
            grid[start_pos[1]][start_pos[0]] = 0
        else:
            stack = [(1, 1)]
            grid[1][1] = 0
        
        # Ensure all connection points are paths
        for x, y in connection_cells:
            grid[y][x] = 0
        
        # Directions: up, right, down, left
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        
        # Generate maze with recursive backtracking
        while stack:
            current = stack[-1]
            x, y = current
            
            # Find unvisited neighbors
            neighbors = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < width and 0 <= ny < height and 
                    grid[ny][nx] == 1):
                    neighbors.append((nx, ny, x + dx // 2, y + dy // 2))
            
            if neighbors:
                # Choose random neighbor using deterministic RNG
                nx, ny, wall_x, wall_y = rng.choice(neighbors)
                
                # Carve path
                grid[ny][nx] = 0
                if 0 <= wall_x < width and 0 <= wall_y < height:
                    grid[wall_y][wall_x] = 0
                
                stack.append((nx, ny))
            else:
                # Backtrack
                stack.pop()
        
        # Ensure all connection points are still paths (connect them if needed)
        for conn_x, conn_y in connection_cells:
            if grid[conn_y][conn_x] == 1:
                grid[conn_y][conn_x] = 0
            # Connect to nearby path if isolated
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = conn_x + dx, conn_y + dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == 0:
                    break
            else:
                # No adjacent path, create one
                if conn_y > 0:
                    grid[conn_y - 1][conn_x] = 0
                elif conn_y < height - 1:
                    grid[conn_y + 1][conn_x] = 0
                elif conn_x > 0:
                    grid[conn_y][conn_x - 1] = 0
                elif conn_x < width - 1:
                    grid[conn_y][conn_x + 1] = 0
        
        # Convert grid to wall/path sets (local coordinates)
        for y in range(height):
            for x in range(width):
                if grid[y][x] == 1:
                    self.walls.add((x, y))
                else:
                    self.paths.add((x, y))
    
    def get_world_cell(self, local_x: int, local_y: int) -> Tuple[int, int]:
        """Convert local chunk coordinates to world cell coordinates"""
        world_x = self.chunk_x * self.chunk_size + local_x
        world_y = self.chunk_y * self.chunk_size + local_y
        return (world_x, world_y)


class Maze:
    """Infinite maze generation and collision system using chunks"""
    
    def __init__(self, chunk_size: int = 20, cell_size: float = 50.0, load_radius: int = 3):
        self.chunk_size = chunk_size
        self.cell_size = cell_size
        self.load_radius = load_radius  # How many chunks to keep loaded around player
        self.chunks: Dict[Tuple[int, int], MazeChunk] = {}  # (chunk_x, chunk_y) -> MazeChunk
        self.last_cleanup_pos = (0, 0)
    
    def _get_chunk_coords(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Get chunk coordinates from world position"""
        world_cell_x = int(world_pos[0] / self.cell_size)
        world_cell_y = int(world_pos[1] / self.cell_size)
        chunk_x = world_cell_x // self.chunk_size
        chunk_y = world_cell_y // self.chunk_size
        # Handle negative coordinates
        if world_cell_x < 0:
            chunk_x = (world_cell_x - self.chunk_size + 1) // self.chunk_size
        if world_cell_y < 0:
            chunk_y = (world_cell_y - self.chunk_size + 1) // self.chunk_size
        return (chunk_x, chunk_y)
    
    def _get_or_create_chunk(self, chunk_x: int, chunk_y: int) -> MazeChunk:
        """Get existing chunk or create new one with connections to adjacent chunks"""
        chunk_key = (chunk_x, chunk_y)
        if chunk_key not in self.chunks:
            # Get adjacent chunks if they exist
            adjacent_chunks = {}
            north_key = (chunk_x, chunk_y - 1)
            south_key = (chunk_x, chunk_y + 1)
            west_key = (chunk_x - 1, chunk_y)
            east_key = (chunk_x + 1, chunk_y)
            
            if north_key in self.chunks:
                adjacent_chunks['north'] = self.chunks[north_key]
            if south_key in self.chunks:
                adjacent_chunks['south'] = self.chunks[south_key]
            if west_key in self.chunks:
                adjacent_chunks['west'] = self.chunks[west_key]
            if east_key in self.chunks:
                adjacent_chunks['east'] = self.chunks[east_key]
            
            # Create new chunk with adjacent chunk information
            # The chunk will automatically match connection points with existing adjacent chunks
            new_chunk = MazeChunk(chunk_x, chunk_y, self.chunk_size, self.cell_size, adjacent_chunks)
            self.chunks[chunk_key] = new_chunk
        
        return self.chunks[chunk_key]
    
    def _ensure_chunks_loaded(self, world_pos: Tuple[float, float]):
        """Ensure chunks around a world position are loaded"""
        center_chunk_x, center_chunk_y = self._get_chunk_coords(world_pos)
        
        # Load chunks in radius
        for dx in range(-self.load_radius, self.load_radius + 1):
            for dy in range(-self.load_radius, self.load_radius + 1):
                chunk_x = center_chunk_x + dx
                chunk_y = center_chunk_y + dy
                self._get_or_create_chunk(chunk_x, chunk_y)
    
    def _cleanup_distant_chunks(self, world_pos: Tuple[float, float]):
        """Remove chunks that are too far from the player"""
        center_chunk_x, center_chunk_y = self._get_chunk_coords(world_pos)
        cleanup_radius = self.load_radius + 2  # Keep a bit more than load radius
        
        chunks_to_remove = []
        for chunk_key in self.chunks.keys():
            chunk_x, chunk_y = chunk_key
            dist_x = abs(chunk_x - center_chunk_x)
            dist_y = abs(chunk_y - center_chunk_y)
            if dist_x > cleanup_radius or dist_y > cleanup_radius:
                chunks_to_remove.append(chunk_key)
        
        for chunk_key in chunks_to_remove:
            del self.chunks[chunk_key]
    
    def world_to_cell(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to maze cell coordinates"""
        cell_x = int(world_pos[0] / self.cell_size)
        cell_y = int(world_pos[1] / self.cell_size)
        return (cell_x, cell_y)
    
    def cell_to_world(self, cell_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert maze cell coordinates to world coordinates (center of cell)"""
        world_x = cell_pos[0] * self.cell_size + self.cell_size / 2
        world_y = cell_pos[1] * self.cell_size + self.cell_size / 2
        return (world_x, world_y)
    
    def check_collision(self, position: Tuple[float, float], radius: float) -> bool:
        """Check if a circular entity collides with maze walls"""
        # Ensure chunks are loaded
        self._ensure_chunks_loaded(position)
        
        cell_x, cell_y = self.world_to_cell(position)
        
        # Check current cell and adjacent cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_cell = (cell_x + dx, cell_y + dy)
                
                # Get chunk for this cell
                chunk_x = check_cell[0] // self.chunk_size
                chunk_y = check_cell[1] // self.chunk_size
                if check_cell[0] < 0:
                    chunk_x = (check_cell[0] - self.chunk_size + 1) // self.chunk_size
                if check_cell[1] < 0:
                    chunk_y = (check_cell[1] - self.chunk_size + 1) // self.chunk_size
                
                chunk = self._get_or_create_chunk(chunk_x, chunk_y)
                
                # Convert to local chunk coordinates
                local_x = check_cell[0] % self.chunk_size
                local_y = check_cell[1] % self.chunk_size
                if check_cell[0] < 0:
                    local_x = (check_cell[0] % self.chunk_size + self.chunk_size) % self.chunk_size
                if check_cell[1] < 0:
                    local_y = (check_cell[1] % self.chunk_size + self.chunk_size) % self.chunk_size
                
                # Check if it's a wall in this chunk
                if (local_x, local_y) in chunk.walls:
                    # Check distance from entity center to wall cell
                    wall_world = self.cell_to_world(check_cell)
                    dist = np.sqrt((position[0] - wall_world[0])**2 + 
                                  (position[1] - wall_world[1])**2)
                    if dist < radius + self.cell_size / 2:
                        return True
        return False
    
    def is_path_cell(self, cell_pos: Tuple[int, int]) -> bool:
        """Check if a cell is a path (not a wall)"""
        cell_x, cell_y = cell_pos
        
        # Get chunk for this cell
        chunk_x = cell_x // self.chunk_size
        chunk_y = cell_y // self.chunk_size
        if cell_x < 0:
            chunk_x = (cell_x - self.chunk_size + 1) // self.chunk_size
        if cell_y < 0:
            chunk_y = (cell_y - self.chunk_size + 1) // self.chunk_size
        
        chunk = self._get_or_create_chunk(chunk_x, chunk_y)
        
        # Convert to local chunk coordinates
        local_x = cell_x % self.chunk_size
        local_y = cell_y % self.chunk_size
        if cell_x < 0:
            local_x = (cell_x % self.chunk_size + self.chunk_size) % self.chunk_size
        if cell_y < 0:
            local_y = (cell_y % self.chunk_size + self.chunk_size) % self.chunk_size
        
        return (local_x, local_y) in chunk.paths and (local_x, local_y) not in chunk.walls
    
    def find_nearest_path_cell(self, world_pos: Tuple[float, float], search_radius: int = 10) -> Optional[Tuple[int, int]]:
        """Find the nearest path cell to a world position"""
        start_cell = self.world_to_cell(world_pos)
        
        # Search in expanding radius
        for radius in range(search_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        test_cell = (start_cell[0] + dx, start_cell[1] + dy)
                        if self.is_path_cell(test_cell):
                            return test_cell
        return None
    
    def find_path(self, start_world: Tuple[float, float], target_world: Tuple[float, float], max_search_radius: int = 50) -> Optional[List[Tuple[int, int]]]:
        """
        Find a path from start to target using A* pathfinding
        Returns a list of cell coordinates, or None if no path found
        """
        # Find nearest path cells
        start_cell = self.find_nearest_path_cell(start_world, search_radius=5)
        target_cell = self.find_nearest_path_cell(target_world, search_radius=5)
        
        if not start_cell or not target_cell:
            return None
        
        # A* pathfinding
        open_set = [(0, start_cell)]  # (f_score, cell)
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: self._heuristic(start_cell, target_cell)}
        closed_set = set()
        
        while open_set:
            # Get cell with lowest f_score
            open_set.sort(key=lambda x: x[0])
            current_f, current = open_set.pop(0)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Check if we reached the target
            if current == target_cell:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_cell)
                path.reverse()
                return path
            
            # Check neighbors (4-directional)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check if neighbor is valid path cell
                if not self.is_path_cell(neighbor):
                    continue
                
                # Check if we've searched too far
                dist_from_start = abs(neighbor[0] - start_cell[0]) + abs(neighbor[1] - start_cell[1])
                if dist_from_start > max_search_radius:
                    continue
                
                tentative_g = g_score.get(current, float('inf')) + 1
                
                if neighbor in closed_set:
                    continue
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, target_cell)
                    open_set.append((f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for A*"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_wall_rects(self, camera_pos: Tuple[float, float], viewport_width: float, viewport_height: float) -> List[Tuple[float, float, float, float]]:
        """Get wall rectangles visible in viewport"""
        # Ensure chunks are loaded around camera
        self._ensure_chunks_loaded(camera_pos)
        
        # Cleanup distant chunks periodically
        if abs(camera_pos[0] - self.last_cleanup_pos[0]) > self.cell_size * self.chunk_size or \
           abs(camera_pos[1] - self.last_cleanup_pos[1]) > self.cell_size * self.chunk_size:
            self._cleanup_distant_chunks(camera_pos)
            self.last_cleanup_pos = camera_pos
        
        rects = []
        
        # Calculate viewport bounds in world coordinates
        viewport_min_x = camera_pos[0] - viewport_width / 2
        viewport_max_x = camera_pos[0] + viewport_width / 2
        viewport_min_y = camera_pos[1] - viewport_height / 2
        viewport_max_y = camera_pos[1] + viewport_height / 2
        
        # Find cells in viewport
        min_cell_x = int(viewport_min_x / self.cell_size) - 1
        max_cell_x = int(viewport_max_x / self.cell_size) + 1
        min_cell_y = int(viewport_min_y / self.cell_size) - 1
        max_cell_y = int(viewport_max_y / self.cell_size) + 1
        
        # Iterate through cells in viewport
        for cell_y in range(min_cell_y, max_cell_y + 1):
            for cell_x in range(min_cell_x, max_cell_x + 1):
                # Get chunk for this cell
                chunk_x = cell_x // self.chunk_size
                chunk_y = cell_y // self.chunk_size
                if cell_x < 0:
                    chunk_x = (cell_x - self.chunk_size + 1) // self.chunk_size
                if cell_y < 0:
                    chunk_y = (cell_y - self.chunk_size + 1) // self.chunk_size
                
                chunk = self._get_or_create_chunk(chunk_x, chunk_y)
                
                # Convert to local chunk coordinates
                local_x = cell_x % self.chunk_size
                local_y = cell_y % self.chunk_size
                if cell_x < 0:
                    local_x = (cell_x % self.chunk_size + self.chunk_size) % self.chunk_size
                if cell_y < 0:
                    local_y = (cell_y % self.chunk_size + self.chunk_size) % self.chunk_size
                
                # Check if it's a wall
                if (local_x, local_y) in chunk.walls:
                    world_x = cell_x * self.cell_size
                    world_y = cell_y * self.cell_size
                    rects.append((world_x, world_y, self.cell_size, self.cell_size))
        
        return rects


class PygameRenderer:
    """Pygame-based renderer for the Kopis Engine"""
    
    def __init__(self, width: int = 800, height: int = 600, maze: Optional['Maze'] = None):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is not available. Install with: pip install pygame")
        
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Kopis Engine")
        self.clock = pygame.time.Clock()
        self.maze = maze
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 100, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        self.WALL_COLOR = (60, 60, 80)
        self.PATH_COLOR = (20, 20, 30)
        
        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
    
    def world_to_screen(self, world_pos: Tuple[float, float], camera_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        screen_x = int(world_pos[0] - camera_pos[0] + self.width / 2)
        screen_y = int(world_pos[1] - camera_pos[1] + self.height / 2)
        return (screen_x, screen_y)
    
    def render(self, entities: List[GameEntity], camera_pos: Tuple[float, float], frame_count: int, fps: float):
        """Render all entities to the screen"""
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Draw maze if available
        if self.maze:
            wall_rects = self.maze.get_wall_rects(camera_pos, self.width, self.height)
            for wall_x, wall_y, wall_w, wall_h in wall_rects:
                screen_x, screen_y = self.world_to_screen((wall_x, wall_y), camera_pos)
                # Draw wall rectangle
                pygame.draw.rect(self.screen, self.WALL_COLOR, 
                               (screen_x, screen_y, int(wall_w), int(wall_h)))
                # Draw wall border
                pygame.draw.rect(self.screen, (40, 40, 50), 
                               (screen_x, screen_y, int(wall_w), int(wall_h)), 2)
        else:
            # Draw grid (optional - helps visualize infinite world)
            grid_size = 50
            viewport_min_x = camera_pos[0] - self.width / 2
            viewport_max_x = camera_pos[0] + self.width / 2
            viewport_min_y = camera_pos[1] - self.height / 2
            viewport_max_y = camera_pos[1] + self.height / 2
            
            # Draw grid lines
            start_x = int(viewport_min_x // grid_size) * grid_size
            start_y = int(viewport_min_y // grid_size) * grid_size
            
            for x in range(start_x, int(viewport_max_x) + grid_size, grid_size):
                screen_x, _ = self.world_to_screen((x, 0), camera_pos)
                if 0 <= screen_x <= self.width:
                    pygame.draw.line(self.screen, (20, 20, 20), (screen_x, 0), (screen_x, self.height))
            
            for y in range(start_y, int(viewport_max_y) + grid_size, grid_size):
                _, screen_y = self.world_to_screen((0, y), camera_pos)
                if 0 <= screen_y <= self.height:
                    pygame.draw.line(self.screen, (20, 20, 20), (0, screen_y), (self.width, screen_y))
        
        # Draw entities
        for entity in entities:
            screen_pos = self.world_to_screen(entity.position, camera_pos)
            
            # Only draw if on screen
            if 0 <= screen_pos[0] <= self.width and 0 <= screen_pos[1] <= self.height:
                # Choose color and size based on entity type
                if entity.id == 'player':
                    color = self.BLUE
                    radius = 15
                elif 'npc' in entity.id:
                    color = self.RED
                    radius = 10
                else:
                    color = self.GREEN
                    radius = 8
                
                # Draw entity circle
                pygame.draw.circle(self.screen, color, screen_pos, radius)
                pygame.draw.circle(self.screen, self.WHITE, screen_pos, radius, 2)
                
                # Draw entity ID label
                label = self.small_font.render(entity.id, True, self.WHITE)
                label_rect = label.get_rect(center=(screen_pos[0], screen_pos[1] - radius - 15))
                self.screen.blit(label, label_rect)
                
                # Draw health bar for entities with health
                if entity.health < 100:
                    bar_width = 30
                    bar_height = 4
                    bar_x = screen_pos[0] - bar_width // 2
                    bar_y = screen_pos[1] + radius + 5
                    
                    # Background
                    pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
                    # Health
                    health_width = int(bar_width * (entity.health / 100.0))
                    health_color = self.GREEN if entity.health > 50 else self.YELLOW if entity.health > 25 else self.RED
                    pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, health_width, bar_height))
        
        # Draw UI overlay
        self._draw_ui(camera_pos, frame_count, fps, len(entities))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
    
    def _draw_ui(self, camera_pos: Tuple[float, float], frame_count: int, fps: float, entity_count: int):
        """Draw UI overlay with game information"""
        # Background for UI
        ui_surface = pygame.Surface((200, 100))
        ui_surface.set_alpha(180)
        ui_surface.fill((0, 0, 0))
        self.screen.blit(ui_surface, (10, 10))
        
        # Draw text
        texts = [
            f"Frame: {frame_count}",
            f"FPS: {fps:.1f}",
            f"Entities: {entity_count}",
            f"Camera: ({camera_pos[0]:.0f}, {camera_pos[1]:.0f})"
        ]
        
        y_offset = 15
        for text in texts:
            text_surface = self.small_font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (15, y_offset))
            y_offset += 20
    
    def get_input(self) -> Dict[str, Any]:
        """Get input from pygame events"""
        keys_pressed = {
            'w': False, 'a': False, 's': False, 'd': False
        }
        
        mouse_pos = (0, 0)
        mouse_clicked = False
        quit_requested = False
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            
            if event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_clicked = True
        
        # Check for held keys using pygame.key.get_pressed() - this is the reliable way
        keys = pygame.key.get_pressed()
        keys_pressed['w'] = keys[pygame.K_w]
        keys_pressed['a'] = keys[pygame.K_a]
        keys_pressed['s'] = keys[pygame.K_s]
        keys_pressed['d'] = keys[pygame.K_d]
        
        return {
            'quit': quit_requested,
            'keys': keys_pressed,
            'mouse': {'x': mouse_pos[0], 'y': mouse_pos[1], 'clicked': mouse_clicked}
        }
    
    def cleanup(self):
        """Clean up pygame resources"""
        if PYGAME_AVAILABLE:
            pygame.quit()


def main():
    """Interactive game loop with pygame visualization, maze, and sound effects"""
    if not PYGAME_AVAILABLE:
        print("Error: pygame is required for visualization.")
        print("Install with: pip install pygame")
        return
    
    # Create infinite maze
    print("Initializing infinite maze system...")
    maze = Maze(chunk_size=20, cell_size=50.0, load_radius=3)
    print(f"✓ Infinite maze system initialized (chunks loaded on demand)")
    
    # Initialize sound manager
    print("Initializing sound system...")
    sound_manager = SoundManager()
    if sound_manager.enabled:
        print("✓ Sound system initialized")
    else:
        print("⚠ Sound system unavailable")
    
    # Initialize engine with maze and sound
    engine = KopisEngine(maze=maze, sound_manager=sound_manager)
    engine.visualize_circuit()
    
    # Initialize pygame renderer with maze
    renderer = PygameRenderer(width=800, height=600, maze=maze)
    
    # Find a valid starting position in the maze (on a path) - RANDOM
    import random
    
    # First, ensure chunks are loaded around origin
    maze._ensure_chunks_loaded((0.0, 0.0))
    
    # Collect all valid starting positions first
    valid_positions = []
    search_radius = 20
    
    for cell_x in range(-search_radius, search_radius + 1):
        for cell_y in range(-search_radius, search_radius + 1):
            # Test cell center position
            test_world = maze.cell_to_world((cell_x, cell_y))
            test_x, test_y = test_world
            
            # Check collision with proper margin
            if not maze.check_collision((test_x, test_y), 12.0 + 2):  # Player radius + margin
                # Double-check by verifying it's actually a path cell
                chunk_x = cell_x // maze.chunk_size
                chunk_y = cell_y // maze.chunk_size
                if cell_x < 0:
                    chunk_x = (cell_x - maze.chunk_size + 1) // maze.chunk_size
                if cell_y < 0:
                    chunk_y = (cell_y - maze.chunk_size + 1) // maze.chunk_size
                
                chunk = maze._get_or_create_chunk(chunk_x, chunk_y)
                local_x = cell_x % maze.chunk_size
                local_y = cell_y % maze.chunk_size
                if cell_x < 0:
                    local_x = (cell_x % maze.chunk_size + maze.chunk_size) % maze.chunk_size
                if cell_y < 0:
                    local_y = (cell_y % maze.chunk_size + maze.chunk_size) % maze.chunk_size
                
                # Verify it's a path, not a wall
                if (local_x, local_y) in chunk.paths and (local_x, local_y) not in chunk.walls:
                    valid_positions.append((test_x, test_y))
    
    # Randomly select from valid positions
    if valid_positions:
        start_world = random.choice(valid_positions)
        start_cell = maze.world_to_cell(start_world)
        print(f"✓ Found {len(valid_positions)} valid starting positions, selected random position")
    else:
        # Fallback: try more aggressive search
        print("⚠ No valid positions found in initial search, trying fallback...")
        fallback_positions = []
        for cell_x in range(-30, 31):
            for cell_y in range(-30, 31):
                test_world = maze.cell_to_world((cell_x, cell_y))
                test_x, test_y = test_world
                if not maze.check_collision((test_x, test_y), 12.0 + 5):
                    fallback_positions.append((test_x, test_y))
        
        if fallback_positions:
            start_world = random.choice(fallback_positions)
            start_cell = maze.world_to_cell(start_world)
            print(f"✓ Found {len(fallback_positions)} fallback positions, selected random position")
        else:
            # Final safety: try random positions
            print("⚠ Using random position search as last resort")
            start_world = (0.0, 0.0)
            for attempts in range(100):
                random_x = (random.random() - 0.5) * 1000
                random_y = (random.random() - 0.5) * 1000
                if not maze.check_collision((random_x, random_y), 12.0 + 10):
                    start_world = (random_x, random_y)
                    break
            start_cell = maze.world_to_cell(start_world)
    
    # Final verification
    if maze.check_collision(start_world, 12.0):
        print("⚠ Warning: Starting position may be in wall, attempting to find safe position...")
        # Try to find any safe position
        for attempts in range(200):
            test_x = (random.random() - 0.5) * 2000
            test_y = (random.random() - 0.5) * 2000
            if not maze.check_collision((test_x, test_y), 12.0 + 5):
                start_world = (test_x, test_y)
                start_cell = maze.world_to_cell(start_world)
                print(f"✓ Found safe position at ({test_x:.1f}, {test_y:.1f})")
                break
    
    # Create player entity at valid maze position
    player = GameEntity(
        id='player',
        position=start_world,
        velocity=(0.0, 0.0),
        health=100.0,
        description="Player character",
        properties={
            'affected_by_gravity': False,  # Player not affected by gravity
            'radius': 12.0  # Player collision radius
        }
    )
    engine.add_entity(player)
    
    # Create many NPCs at different valid positions near the start
    # Performance optimized: can handle many NPCs with staggered pathfinding
    MAX_NPCS = 50  # Increased from 5 to 50
    for i in range(MAX_NPCS):
        # Find a nearby path cell for NPC
        npc_offset = (i + 1) * 3  # Space NPCs out
        npc_cell = (start_cell[0] + npc_offset, start_cell[1])
        
        # Ensure chunk is loaded
        npc_world_temp = maze.cell_to_world(npc_cell)
        maze._ensure_chunks_loaded(npc_world_temp)
        
        # Verify it's a path, if not find nearby
        chunk_x = npc_cell[0] // maze.chunk_size
        chunk_y = npc_cell[1] // maze.chunk_size
        if npc_cell[0] < 0:
            chunk_x = (npc_cell[0] - maze.chunk_size + 1) // maze.chunk_size
        if npc_cell[1] < 0:
            chunk_y = (npc_cell[1] - maze.chunk_size + 1) // maze.chunk_size
        
        chunk = maze._get_or_create_chunk(chunk_x, chunk_y)
        local_x = npc_cell[0] % maze.chunk_size
        local_y = npc_cell[1] % maze.chunk_size
        if npc_cell[0] < 0:
            local_x = (npc_cell[0] % maze.chunk_size + maze.chunk_size) % maze.chunk_size
        if npc_cell[1] < 0:
            local_y = (npc_cell[1] % maze.chunk_size + maze.chunk_size) % maze.chunk_size
        
        if (local_x, local_y) not in chunk.paths:
            # Try nearby cells
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                test_cell = (npc_cell[0] + dx, npc_cell[1] + dy)
                test_chunk_x = test_cell[0] // maze.chunk_size
                test_chunk_y = test_cell[1] // maze.chunk_size
                if test_cell[0] < 0:
                    test_chunk_x = (test_cell[0] - maze.chunk_size + 1) // maze.chunk_size
                if test_cell[1] < 0:
                    test_chunk_y = (test_cell[1] - maze.chunk_size + 1) // maze.chunk_size
                test_chunk = maze._get_or_create_chunk(test_chunk_x, test_chunk_y)
                test_local_x = test_cell[0] % maze.chunk_size
                test_local_y = test_cell[1] % maze.chunk_size
                if test_cell[0] < 0:
                    test_local_x = (test_cell[0] % maze.chunk_size + maze.chunk_size) % maze.chunk_size
                if test_cell[1] < 0:
                    test_local_y = (test_cell[1] % maze.chunk_size + maze.chunk_size) % maze.chunk_size
                if (test_local_x, test_local_y) in test_chunk.paths:
                    npc_cell = test_cell
                    break
        
        npc_world = maze.cell_to_world(npc_cell)
        npc = GameEntity(
            id=f'npc_{i}',
            position=npc_world,
            velocity=(0.0, 0.0),
            health=50.0,
            description=f"NPC {i+1}",
            properties={
                'affected_by_gravity': False,  # NPCs not affected by gravity
                'radius': 8.0,  # NPC collision radius
                'path': None,  # Pathfinding path
                'path_index': 0,  # Current waypoint index
                'last_path_update': 0.0  # Last time path was calculated
            }
        )
        engine.add_entity(npc)
    
    print("\n" + "="*60)
    print("KOPIS ENGINE - Pygame Visualization with Maze")
    print("="*60)
    print("Controls: W/A/S/D to move, ESC or close window to quit")
    print("NPCs will automatically move towards the player")
    print("Navigate through the maze - avoid walls!")
    print("Sound effects: footsteps and collisions")
    print("="*60)
    
    running = True
    frame = 0
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 60.0
    
    try:
        while running:
            # Get input from pygame
            input_data = renderer.get_input()
            
            if input_data.get('quit', False):
                running = False
                break
            
            # Process frame
            result = engine.process_frame(input_data)
            frame += 1
            fps_counter += 1
            
            # Calculate FPS
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()
            
            # Get camera position from engine (smoothly follows player)
            camera_pos = engine.camera_pos
            
            # Render with pygame
            renderer.render(engine.entities, camera_pos, frame, current_fps)
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\n\nGame interrupted")
    
    finally:
        # Cleanup
        renderer.cleanup()
        if sound_manager:
            sound_manager.cleanup()
        engine.cleanup()
        
        # Print final stats
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        stats = engine.get_stats()
        print(json.dumps(stats, indent=2))
        print("="*60)


if __name__ == "__main__":
    main()
