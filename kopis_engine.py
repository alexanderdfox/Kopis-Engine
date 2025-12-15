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
    """Represents a game entity (player, NPC, object) - 3D coordinates"""
    id: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, z) - 3D position
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (vx, vy, vz) - 3D velocity
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
        Process physics simulation with enhanced features (3D):
        - Gravity (Z axis)
        - Friction
        - Infinite world (no boundaries)
        - Velocity-based movement
        - Maze collision detection (X/Y plane)
        - Ground collision (Z axis)
        """
        updated_entities = []
        # Realistic gravity for a 10-foot room
        # Scale: 1 foot = 30 units (so 10 feet = 300 units)
        # Real gravity: 32.2 ft/s² = 9.8 m/s²
        # In game units: 32.2 ft/s² * 30 units/ft = 966 units/s²
        gravity = 32.2 * 30  # Realistic gravity: 966 units/s² (based on 1 foot = 30 units)
        friction_coefficient = 0.95  # Friction factor (0.95 = 5% velocity loss per frame)
        ground_level = 0.0  # Ground is at Z = 0 (positive Z is up in the air)
        ceiling_level = 300.0  # Ceiling is at Z = 300 units (10 feet = 300 units at 30 units/foot)
        
        for entity in entities:
            # Get current state (3D)
            x, y, z = entity.position
            vx, vy, vz = entity.velocity
            
            # Apply gravity (if entity is affected by gravity) - affects Z axis
            # Positive Z is up, so gravity pulls down (decreases upward velocity)
            # When vz is positive (moving up), gravity reduces it
            # When vz is negative (falling), gravity makes it more negative (faster fall)
            if entity.properties.get('affected_by_gravity', True):
                # Always apply gravity - this ensures entities fall back down after jumping
                # Ground collision below will stop downward velocity when hitting the ground
                vz -= gravity * delta_time  # Gravity always pulls down (decreases vz)
            
            # Apply friction (only to X and Y, not Z)
            # Doom-style: Player has no friction (instant movement), NPCs have friction
            is_player = entity.id == 'player'
            if not is_player:
                # Apply friction to NPCs only (not player - Doom-style instant movement)
                vx *= friction_coefficient
                vy *= friction_coefficient
                # Stop very small velocities to prevent jitter
                if abs(vx) < 1.0:
                    vx = 0.0
                if abs(vy) < 1.0:
                    vy = 0.0
            else:
                # Player: Doom-style - stop immediately when not moving (no friction, but instant stop)
                # Velocity is set directly by input, so if it's very small, just stop
                if abs(vx) < 0.1:
                    vx = 0.0
                if abs(vy) < 0.1:
                    vy = 0.0
            
            # Update position based on velocity (infinite world - no bounds)
            new_x = x + vx * delta_time
            new_y = y + vy * delta_time
            new_z = z + vz * delta_time
            
            # Ground collision (Z axis) - prevent going below ground
            if new_z < ground_level:
                new_z = ground_level
                if vz < 0:  # Only stop downward velocity if hitting ground
                    vz = 0
                # Mark as on ground
                entity.properties['on_ground'] = True
            else:
                # In the air
                entity.properties['on_ground'] = False
            
            # Ceiling collision (Z axis) - prevent going above ceiling and make player fall
            if new_z > ceiling_level:
                new_z = ceiling_level
                if vz > 0:  # Only stop upward velocity if hitting ceiling
                    vz = 0  # Stop upward velocity, gravity will pull down
                # Mark as hitting ceiling (will fall due to gravity)
                entity.properties['on_ground'] = False
            
            # Check maze collision if maze exists (X/Y plane only)
            entity_radius = entity.properties.get('radius', 10.0)
            if maze:
                # First, check if entity is currently stuck inside a wall (safety check)
                if maze.check_collision((x, y), entity_radius):
                    # Entity is stuck inside a wall - find nearest safe position
                    safe_position = maze.find_nearest_safe_position((x, y), entity_radius, max_search_radius=10)
                    if safe_position:
                        new_x, new_y = safe_position
                        vx = 0.0
                        vy = 0.0
                        if entity.id == 'player':
                            print(f"⚠ Player was stuck in wall, moved to safe position")
                    else:
                        # Couldn't find safe position - try to push out in any direction
                        for check_dx, check_dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                            push_distance = entity_radius * 2.0
                            check_x = x + check_dx * push_distance
                            check_y = y + check_dy * push_distance
                            if not maze.check_collision((check_x, check_y), entity_radius):
                                new_x = check_x
                                new_y = check_y
                                vx = 0.0
                                vy = 0.0
                                if entity.id == 'player':
                                    print(f"⚠ Player pushed out of wall")
                                break
                
                # Continuous collision detection - check along movement path
                # Use multiple steps to prevent passing through walls
                steps = max(1, int(abs(vx * delta_time) / (entity_radius * 0.5)) + int(abs(vy * delta_time) / (entity_radius * 0.5)))
                steps = min(steps, 10)  # Limit steps for performance
                
                if steps > 1:
                    step_dx = (new_x - x) / steps
                    step_dy = (new_y - y) / steps
                    last_valid_x, last_valid_y = x, y
                    
                    for step in range(1, steps + 1):
                        test_x = x + step_dx * step
                        test_y = y + step_dy * step
                        if not maze.check_collision((test_x, test_y), entity_radius):
                            last_valid_x, last_valid_y = test_x, test_y
                        else:
                            # Hit a wall, stop at last valid position
                            new_x = last_valid_x
                            new_y = last_valid_y
                            # Reduce velocity to prevent bouncing
                            vx *= 0.3
                            vy *= 0.3
                            break
                
                # Final collision check at new position
                if maze.check_collision((new_x, new_y), entity_radius):
                    # Collision detected - try moving only X or only Y
                    if not maze.check_collision((new_x, y), entity_radius):
                        new_y = y  # Keep old Y, allow X movement
                        vx *= 0.5  # Reduce velocity instead of stopping
                    elif not maze.check_collision((x, new_y), entity_radius):
                        new_x = x  # Keep old X, allow Y movement
                        vy *= 0.5  # Reduce velocity instead of stopping
                    else:
                        # Can't move in either direction - try to slide along wall
                        # Try perpendicular movement
                        perp_x = -vy * 0.1
                        perp_y = vx * 0.1
                        if not maze.check_collision((x + perp_x, y + perp_y), entity_radius):
                            new_x = x + perp_x
                            new_y = y + perp_y
                            vx *= 0.3
                            vy *= 0.3
                        else:
                            # Really stuck - find nearest safe position
                            safe_position = maze.find_nearest_safe_position((x, y), entity_radius, max_search_radius=5)
                            if safe_position:
                                new_x, new_y = safe_position
                                vx = 0.0
                                vy = 0.0
                            else:
                                # Last resort - stay in place and stop
                                new_x = x
                                new_y = y
                                vx = 0.0
                                vy = 0.0
            
            # Create updated entity
            updated_entities.append(GameEntity(
                id=entity.id,
                position=(new_x, new_y, new_z),
                velocity=(vx, vy, vz),
                health=entity.health,
                description=entity.description,
                properties=entity.properties
            ))
        return updated_entities
    
    def process_rendering(self, entities: List[GameEntity], camera_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Dict[str, Any]:
        """
        Process rendering data and generate ASCII visualization (3D)
        Uses camera position for viewport rendering in infinite world
        """
        render_data = {
            'entities': [],
            'camera': {'x': camera_pos[0], 'y': camera_pos[1], 'z': camera_pos[2], 'zoom': 1.0},
            'ascii_map': None
        }
        
        # Collect entity data (3D)
        for entity in entities:
            render_data['entities'].append({
                'id': entity.id,
                'x': entity.position[0],
                'y': entity.position[1],
                'z': entity.position[2],
                'health': entity.health
            })
        
        # Generate ASCII visualization with camera offset (project to 2D for ASCII)
        render_data['ascii_map'] = self._generate_ascii_map(entities, camera_pos)
        
        return render_data
    
    def _generate_ascii_map(self, entities: List[GameEntity], camera_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0), width: int = 80, height: int = 24, viewport_size: float = 800.0) -> str:
        """
        Generate ASCII text-based visualization of game world (3D projected to 2D)
        Uses camera position to render viewport in infinite world
        Projects 3D positions to 2D for ASCII display
        """
        # Create empty grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Calculate viewport bounds (centered on camera, X/Y plane)
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
        # Sort by Z to show higher entities on top
        sorted_entities = sorted(entities, key=lambda e: e.position[2])
        
        for entity in sorted_entities:
            entity_x, entity_y, entity_z = entity.position
            
            # Check if entity is within viewport (X/Y plane)
            if (viewport_min_x <= entity_x <= viewport_max_x and 
                viewport_min_y <= entity_y <= viewport_max_y):
                
                # Convert world coordinates to grid coordinates relative to camera
                grid_x = int((entity_x - viewport_min_x) * scale_x)
                grid_y = int((entity_y - viewport_min_y) * scale_y)
                
                # Clamp to grid bounds
                grid_x = max(0, min(width - 1, grid_x))
                grid_y = max(0, min(height - 1, grid_y))
                
                # Choose symbol based on entity type and height
                if entity.id == 'player':
                    symbol = '@' if entity_z < 10 else '^'  # Show ^ if jumping high
                elif 'npc' in entity.id:
                    symbol = 'N' if entity_z < 10 else '^'
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
        MAX_UPDATE_DISTANCE = 400.0  # Only update NPCs within 400 pixels (reduced for better FPS)
        PATHFINDING_BATCH_SIZE = 2  # Update max 2 NPCs' paths per frame (reduced for better FPS)
        
        npcs_to_update = []
        for entity in entities:
            if entity.id != player_entity.id and 'npc' in entity.id:
                # Distance-based culling (3D distance, but use X/Y for pathfinding)
                dx = entity.position[0] - player_entity.position[0]
                dy = entity.position[1] - player_entity.position[1]
                dz = entity.position[2] - player_entity.position[2] if len(entity.position) == 3 else 0.0
                distance = np.sqrt(dx**2 + dy**2 + dz**2)  # 3D distance for culling
                
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
            # Pathfinding is 2D (X/Y plane only), Z is handled separately
            if update_path:
                # Adaptive search radius based on distance
                search_radius = min(50, max(20, int(distance / 100)))
                # Extract X/Y coordinates for pathfinding (ignore Z)
                entity_xy = (entity.position[0], entity.position[1])
                player_xy = (player_entity.position[0], player_entity.position[1])
                path = maze.find_path(entity_xy, player_xy, max_search_radius=search_radius)
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
                
                # Calculate direction to waypoint (X/Y plane only, Z handled by physics)
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
                    # Calculate velocity towards waypoint (X/Y only, Z handled by physics)
                    vx = (dx / distance_to_waypoint) * speed
                    vy = (dy / distance_to_waypoint) * speed
                    vz = entity.velocity[2]  # Preserve Z velocity from physics
                    
                    # Update position (preserve Z, it's handled by physics)
                    new_x = entity.position[0] + vx * delta_time
                    new_y = entity.position[1] + vy * delta_time
                    new_z = entity.position[2]  # Z will be updated by physics
                    
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
                    new_z = entity.position[2]
                    vx = 0
                    vy = 0
                    vz = entity.velocity[2]  # Preserve Z velocity
            else:
                # No path or path exhausted, use simple fallback movement
                dx = player_entity.position[0] - entity.position[0]
                dy = player_entity.position[1] - entity.position[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 0:
                    vx = (dx / distance) * speed
                    vy = (dy / distance) * speed
                    vz = entity.velocity[2]  # Preserve Z velocity
                    
                    new_x = entity.position[0] + vx * delta_time
                    new_y = entity.position[1] + vy * delta_time
                    new_z = entity.position[2]  # Z handled by physics
                    
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
                    new_z = entity.position[2]
                    vx = 0
                    vy = 0
                    vz = entity.velocity[2]  # Preserve Z velocity
            
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
                position=(new_x, new_y, new_z),
                velocity=(vx, vy, vz),
                health=entity.health,
                description=entity.description,
                properties=updated_properties
            ))
        return updated_entities
    
    def process_parallel(self, signal: Signal, game_data: Dict[str, Any]) -> Dict[str, Signal]:
        """
        Process through all parallel branches simultaneously (3D)
        
        Returns:
            Dictionary of signals from each branch
        """
        entities = game_data.get('entities', [])
        delta_time = game_data.get('delta_time', 0.016)
        player_entity = game_data.get('player', None)
        camera_pos = game_data.get('camera_pos', (0.0, 0.0, 0.0))  # 3D camera
        
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
        
        # Camera system for smooth scrolling (3D) with FPV support and 6DOF
        self.camera_pos = (0.0, 0.0, 0.0)  # (x, y, z) - 3D camera position
        self.camera_smoothness = 0.1  # Lower = smoother, higher = snappier (0.1 = smooth, 1.0 = instant)
        self.camera_yaw = 0.0  # Horizontal rotation (left/right) in degrees
        self.camera_pitch = 0.0  # Vertical rotation (up/down) in degrees
        self.camera_roll = 0.0  # Roll rotation (tilt left/right) in degrees - 6DOF
        self.fpv_mode = True  # First-person view mode
        # Camera height for 10-foot room: standard FPS eye level is ~5 feet = 150 units
        # Scale: 1 foot = 30 units, so 5 feet = 150 units
        self.camera_height = 150.0  # Camera height above ground (eye level at ~5 feet)
        
        # Initialize game rules
        self._initialize_rules()
        
        print("✓ Kopis Engine initialized successfully")
    
    def _initialize_rules(self):
        """Initialize game rules and conditions"""
        self.nand_gate.add_rule('win_condition', lambda state: state.get('win', False))
        self.nand_gate.add_rule('lose_condition', lambda state: state.get('lose', False))
    
    def _process_player_input(self, input_data: Dict[str, Any], delta_time: float, sound_manager: Optional['SoundManager'] = None, screen_width: int = 800, screen_height: int = 600):
        """
        Process player input for movement (3D FPV) - Doom-style
        WASD keys control player movement relative to camera yaw (where you're looking)
        Space bar controls jumping (Z axis)
        Movement is instant (no acceleration/deceleration) like classic Doom
        """
        if not self.player:
            return
        
        import math
        
        keys = input_data.get('keys', {})
        player_speed = 200.0  # pixels per second (Doom-style fast movement)
        
        # Calculate jump force for realistic 10-foot room
        # Scale: 1 foot = 30 units, so 10 feet = 300 units
        # Gravity: 966 units/s² (32.2 ft/s² * 30 units/ft)
        # Target jump height: 7 feet = 210 units (leaving 3 feet headroom)
        # Using physics: max_height = v₀² / (2 * g)
        # 210 = jump_force² / (2 * 966)
        # jump_force² = 210 * 2 * 966 = 405,720
        # jump_force = √405,720 ≈ 637
        jump_force = 637.0  # Jump velocity to reach 7 feet (210 units) in a 10-foot room
        
        # Get current velocity (3D) - handle both 2D and 3D for compatibility
        if len(self.player.velocity) == 3:
            vx, vy, vz = self.player.velocity
        else:
            vx, vy = self.player.velocity
            vz = 0.0  # Default Z velocity
        was_moving = abs(vx) > 0.1 or abs(vy) > 0.1
        
        # Check if player is on ground
        on_ground = self.player.properties.get('on_ground', False)
        
        # Doom-style: Reset X/Y velocity based on input (instant velocity changes, no acceleration)
        vx = 0.0
        vy = 0.0
        # vz is preserved from above (will be updated if jumping, otherwise kept for physics)
        
        # Calculate movement direction based on camera yaw (Doom-style)
        # Camera yaw: 0 = looking along -Y (north), 90 = looking along +X (east), etc.
        camera_yaw = self.camera_yaw
        yaw_rad = math.radians(camera_yaw)
        
        # Forward direction: sin(yaw) for X, -cos(yaw) for Y
        # This makes yaw=0 point along -Y (forward/north)
        forward_x = math.sin(yaw_rad)
        forward_y = -math.cos(yaw_rad)
        
        # Right direction: cos(yaw) for X, sin(yaw) for Y
        # This is 90 degrees clockwise from forward
        right_x = math.cos(yaw_rad)
        right_y = math.sin(yaw_rad)
        
        # Process movement keys relative to camera direction (Doom-style)
        move_forward = 0.0
        move_right = 0.0
        
        if keys.get('w', False):
            move_forward += 1.0  # Move forward
        if keys.get('s', False):
            move_forward -= 1.0  # Move backward
        if keys.get('a', False):
            move_right -= 1.0  # Move left (strafe left)
        if keys.get('d', False):
            move_right += 1.0  # Move right (strafe right)
        
        # Calculate velocity in world space based on camera-relative movement (Doom-style)
        if move_forward != 0.0 or move_right != 0.0:
            # Normalize diagonal movement to prevent faster diagonal movement
            move_magnitude = math.sqrt(move_forward**2 + move_right**2)
            if move_magnitude > 0.0:
                move_forward /= move_magnitude
                move_right /= move_magnitude
                # Apply diagonal normalization factor (Doom-style: diagonal movement is slower)
                if abs(move_forward) > 0.1 and abs(move_right) > 0.1:
                    move_forward *= 0.707  # sqrt(2)/2 - normalize diagonal
                    move_right *= 0.707
            
            # Calculate world-space velocity (instant, no acceleration - Doom-style)
            vx = (forward_x * move_forward + right_x * move_right) * player_speed
            vy = (forward_y * move_forward + right_y * move_right) * player_speed
        
        # Process jumping (Space bar)
        # Note: vz is already set from the velocity unpacking above
        if keys.get('space', False) and on_ground:
            vz = jump_force  # Jump upward (positive Z is up in our system)
            self.player.properties['on_ground'] = False
            if sound_manager:
                sound_manager.play('move_start', volume=0.4)  # Jump sound
        # If not jumping, vz is preserved from the unpacking above (physics will handle it)
        
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
                # Create new entity with updated velocity (3D)
                updated_entity = GameEntity(
                    id=entity.id,
                    position=entity.position,
                    velocity=(vx, vy, vz),
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
            # Get screen dimensions from renderer if available, otherwise use defaults
            screen_width = getattr(self, 'screen_width', 800)
            screen_height = getattr(self, 'screen_height', 600)
            self._process_player_input(input_data, delta_time, self.sound_manager, screen_width, screen_height)
        
        # Update camera rotation from mouse input (FPV) - 6DOF with full rotation
        if self.fpv_mode and input_data.get('mouse'):
            mouse_dx = input_data['mouse'].get('dx', 0)
            mouse_dy = input_data['mouse'].get('dy', 0)
            # Update camera yaw (horizontal rotation) - unlimited 360 degrees
            self.camera_yaw += mouse_dx * 0.1  # Sensitivity
            self.camera_yaw = self.camera_yaw % 360.0  # Normalize to 0-360
            
            # Update camera pitch (vertical rotation) - unlimited 360 degrees (full 6DOF)
            self.camera_pitch += mouse_dy * 0.1  # Sensitivity
            self.camera_pitch = self.camera_pitch % 360.0  # Normalize to 0-360
            
            # Roll can be controlled with Q/E keys (optional 6DOF feature)
            keys = input_data.get('keys', {})
            roll_speed = 2.0  # degrees per frame
            if keys.get('q', False):
                self.camera_roll -= roll_speed
            if keys.get('e', False):
                self.camera_roll += roll_speed
            self.camera_roll = self.camera_roll % 360.0  # Normalize to 0-360
        
        # Update camera to smoothly follow player (3D) with FPV
        if self.player:
            target_camera_x = self.player.position[0]
            target_camera_y = self.player.position[1]
            # In FPV, camera is at eye level above player's feet
            if self.fpv_mode:
                target_camera_z = self.player.position[2] + self.camera_height
            else:
                target_camera_z = self.player.position[2]  # Follow player's Z position
            
            # Smooth camera interpolation (lerp)
            current_camera_x, current_camera_y, current_camera_z = self.camera_pos
            lerp_factor = min(1.0, self.camera_smoothness * (1.0 + delta_time * 60))  # Adjust for frame rate
            new_camera_x = current_camera_x + (target_camera_x - current_camera_x) * lerp_factor
            new_camera_y = current_camera_y + (target_camera_y - current_camera_y) * lerp_factor
            new_camera_z = current_camera_z + (target_camera_z - current_camera_z) * lerp_factor
            
            self.camera_pos = (new_camera_x, new_camera_y, new_camera_z)
        else:
            self.camera_pos = (0.0, 0.0, 0.0)
        
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
    
    def _ensure_chunks_loaded(self, world_pos):
        """Ensure chunks around a world position are loaded (handles 2D or 3D positions)"""
        # Extract X/Y coordinates (ignore Z for 2D maze)
        if len(world_pos) == 3:
            world_pos_2d = (world_pos[0], world_pos[1])
        else:
            world_pos_2d = world_pos
        center_chunk_x, center_chunk_y = self._get_chunk_coords(world_pos_2d)
        
        # Load chunks in radius
        for dx in range(-self.load_radius, self.load_radius + 1):
            for dy in range(-self.load_radius, self.load_radius + 1):
                chunk_x = center_chunk_x + dx
                chunk_y = center_chunk_y + dy
                self._get_or_create_chunk(chunk_x, chunk_y)
    
    def _cleanup_distant_chunks(self, world_pos):
        """Remove chunks that are too far from the player (handles 2D or 3D positions)"""
        # Extract X/Y coordinates (ignore Z for 2D maze)
        if len(world_pos) == 3:
            world_pos_2d = (world_pos[0], world_pos[1])
        else:
            world_pos_2d = world_pos
        center_chunk_x, center_chunk_y = self._get_chunk_coords(world_pos_2d)
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
    
    def check_collision(self, position, radius: float) -> bool:
        """Check if a circular entity collides with maze walls (handles 2D or 3D positions)"""
        # Extract X/Y coordinates (ignore Z for 2D maze collision)
        if len(position) == 3:
            position_2d = (position[0], position[1])
        else:
            position_2d = position
        # Ensure chunks are loaded
        self._ensure_chunks_loaded(position_2d)
        
        cell_x, cell_y = self.world_to_cell(position_2d)
        
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
                    # Check distance from entity center to wall cell (X/Y plane only)
                    wall_world = self.cell_to_world(check_cell)
                    dist = np.sqrt((position_2d[0] - wall_world[0])**2 + 
                                  (position_2d[1] - wall_world[1])**2)
                    # Use a slightly smaller collision radius to allow movement near walls
                    collision_margin = radius + self.cell_size / 2 - 2.0  # Small margin for movement
                    if dist < collision_margin:
                        return True
        return False
    
    def find_nearest_safe_position(self, position, radius: float, max_search_radius: int = 10) -> Optional[Tuple[float, float]]:
        """Find the nearest safe (non-colliding) position from the given position"""
        # Extract X/Y coordinates (ignore Z for 2D maze)
        if len(position) == 3:
            position_2d = (position[0], position[1])
        else:
            position_2d = position
        
        # Ensure chunks are loaded
        self._ensure_chunks_loaded(position_2d)
        
        # Search in expanding circles around the position
        for search_radius in range(1, max_search_radius + 1):
            # Check positions in a grid pattern around the current position
            step_size = self.cell_size * 0.5
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    # Skip if outside search radius
                    if abs(dx) + abs(dy) > search_radius:
                        continue
                    
                    test_x = position_2d[0] + dx * step_size
                    test_y = position_2d[1] + dy * step_size
                    
                    # Check if this position is safe
                    if not self.check_collision((test_x, test_y), radius):
                        return (test_x, test_y)
        
        # No safe position found
        return None
    
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
    
    def find_nearest_path_cell(self, world_pos, search_radius: int = 10) -> Optional[Tuple[int, int]]:
        """Find the nearest path cell to a world position (handles 2D or 3D positions)"""
        # Extract X/Y coordinates (ignore Z for 2D maze)
        if len(world_pos) == 3:
            world_pos_2d = (world_pos[0], world_pos[1])
        else:
            world_pos_2d = world_pos
        start_cell = self.world_to_cell(world_pos_2d)
        
        # Search in expanding radius
        for radius in range(search_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        test_cell = (start_cell[0] + dx, start_cell[1] + dy)
                        if self.is_path_cell(test_cell):
                            return test_cell
        return None
    
    def find_path(self, start_world, target_world, max_search_radius: int = 50) -> Optional[List[Tuple[int, int]]]:
        """
        Find a path from start to target using A* pathfinding (handles 2D or 3D positions)
        Returns a list of cell coordinates, or None if no path found
        """
        # Extract X/Y coordinates (ignore Z for 2D maze pathfinding)
        if len(start_world) == 3:
            start_world_2d = (start_world[0], start_world[1])
        else:
            start_world_2d = start_world
        if len(target_world) == 3:
            target_world_2d = (target_world[0], target_world[1])
        else:
            target_world_2d = target_world
        
        # Find nearest path cells
        start_cell = self.find_nearest_path_cell(start_world_2d, search_radius=5)
        target_cell = self.find_nearest_path_cell(target_world_2d, search_radius=5)
        
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
    
    def get_wall_rects(self, camera_pos, viewport_width: float, viewport_height: float) -> List[Tuple[float, float, float, float]]:
        """Get wall rectangles visible in viewport (handles 2D or 3D camera position)"""
        # Extract X/Y coordinates (ignore Z for 2D maze)
        if len(camera_pos) == 3:
            camera_pos_2d = (camera_pos[0], camera_pos[1])
        else:
            camera_pos_2d = camera_pos
        # Ensure chunks are loaded around camera
        self._ensure_chunks_loaded(camera_pos_2d)
        
        # Cleanup distant chunks periodically
        if abs(camera_pos_2d[0] - self.last_cleanup_pos[0]) > self.cell_size * self.chunk_size or \
           abs(camera_pos_2d[1] - self.last_cleanup_pos[1]) > self.cell_size * self.chunk_size:
            self._cleanup_distant_chunks(camera_pos_2d)
            self.last_cleanup_pos = camera_pos_2d
        
        rects = []
        
        # Calculate viewport bounds in world coordinates (use 2D camera position)
        viewport_min_x = camera_pos_2d[0] - viewport_width / 2
        viewport_max_x = camera_pos_2d[0] + viewport_width / 2
        viewport_min_y = camera_pos_2d[1] - viewport_height / 2
        viewport_max_y = camera_pos_2d[1] + viewport_height / 2
        
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
    
    def get_wall_boxes_3d(self, camera_pos, view_distance: float = 1000.0, wall_height: float = 100.0) -> List[Tuple[float, float, float, float, float, float, Tuple[int, int, int]]]:
        """Get 3D wall boxes (x, y, z, width, height, depth, color) visible from camera with random colors"""
        import hashlib
        import random
        
        # Extract X/Y coordinates (ignore Z for 2D maze)
        if len(camera_pos) == 3:
            camera_pos_2d = (camera_pos[0], camera_pos[1])
        else:
            camera_pos_2d = camera_pos
        # Ensure chunks are loaded around camera
        self._ensure_chunks_loaded(camera_pos_2d)
        
        boxes = []
        # Get walls in a radius around camera
        search_radius = int(view_distance / self.cell_size) + 2
        
        center_cell = self.world_to_cell(camera_pos_2d)
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                cell_x = center_cell[0] + dx
                cell_y = center_cell[1] + dy
                
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
                    world_z = 0.0  # Ground level
                    
                    # Generate deterministic random color based on cell position
                    # Doom-style wall colors: dark grays, browns, and muted tones
                    seed_str = f"{cell_x}_{cell_y}"
                    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
                    rng = random.Random(seed)
                    # Doom palette: dark grays (30-80), browns (40-100), muted tones
                    # Create variation but keep it dark and moody
                    base_r = rng.randint(30, 80)  # Dark red-brown tones
                    base_g = rng.randint(25, 60)  # Muted greens
                    base_b = rng.randint(20, 50)  # Dark blues
                    wall_color = (base_r, base_g, base_b)
                    
                    # Return as 3D box: (x, y, z, width, height, depth, color)
                    boxes.append((world_x, world_y, world_z, self.cell_size, wall_height, self.cell_size, wall_color))
        
        return boxes


class GameOfLife:
    """Conway's Game of Life for a single wall (100x100 grid)"""
    
    def __init__(self, seed: int = None):
        import random
        import numpy as np
        
        self.width = 100
        self.height = 100
        self.grid = np.zeros((self.height, self.width), dtype=bool)
        self.frame_count = 0  # Track frames for deterministic gravity
        
        # Initialize with random pattern
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        # Initialize with blood-dripping pattern (more cells at top, less at bottom)
        # This creates a dripping effect that looks like blood on walls
        for y in range(self.height):
            for x in range(self.width):
                # Higher probability at top (blood starts dripping from top)
                # Probability decreases as we go down (gravity effect)
                top_probability = 0.4  # 40% at top
                bottom_probability = 0.1  # 10% at bottom
                probability = top_probability - (top_probability - bottom_probability) * (y / self.height)
                
                # Add some randomness for natural blood patterns
                if rng.random() < probability:
                    self.grid[y, x] = True
    
    def update(self):
        """Update the grid according to Conway's Game of Life rules (optimized with numpy)"""
        import numpy as np
        
        # Optimized version using numpy vectorized operations
        # Count neighbors using numpy roll operations (much faster than loops)
        neighbor_count = np.zeros_like(self.grid, dtype=int)
        
        # Sum all 8 neighbors using numpy roll
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                # Roll the grid and add to neighbor count
                rolled = np.roll(np.roll(self.grid.astype(int), dy, axis=0), dx, axis=1)
                neighbor_count += rolled
        
        # Apply Conway's rules using vectorized operations (much faster)
        # Rule 1: Live cell with 2-3 neighbors survives
        survives = (self.grid) & ((neighbor_count == 2) | (neighbor_count == 3))
        # Rule 2: Dead cell with 3 neighbors becomes alive
        born = (~self.grid) & (neighbor_count == 3)
        
        # Combine rules
        new_grid = survives | born
        
        # Add gravity effect: cells tend to "fall" down (blood dripping effect)
        # Shift cells down slightly to simulate dripping (simplified for performance)
        gravity_shift = np.roll(new_grid, 1, axis=0)  # Shift down by 1
        # Some cells fall down (blood dripping) - use deterministic pattern based on frame
        # Use frame count to create time-based falling pattern (10% chance, changes over time)
        self.frame_count += 1
        fall_mask = ((np.arange(self.height * self.width).reshape(self.height, self.width) + self.frame_count) % 10) < 1
        # Cells that fall replace empty cells below
        new_grid = new_grid | (gravity_shift & fall_mask & ~new_grid)
        
        self.grid = new_grid
    
    def get_pattern(self) -> np.ndarray:
        """Get the current grid pattern"""
        return self.grid


class PygameRenderer:
    """Pygame-based renderer for the Kopis Engine"""
    
    def __init__(self, width: int = 800, height: int = 600, maze: Optional['Maze'] = None):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is not available. Install with: pip install pygame")
        
        pygame.init()
        self.fullscreen = True  # Start in fullscreen mode
        if self.fullscreen:
            # Get desktop resolution for fullscreen
            info = pygame.display.Info()
            self.width = info.current_w
            self.height = info.current_h
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        else:
            self.width = width
            self.height = height
            self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Kopis Engine - Press F11 to toggle fullscreen")
        self.clock = pygame.time.Clock()
        self.maze = maze
        self.mouse_captured = False  # Track mouse capture state
        
        # Single shared Game of Life instance for all walls
        self.shared_game_of_life = GameOfLife(seed=42)  # Fixed seed so all walls use same pattern
        self.game_of_life_update_counter = 0  # Update Game of Life every N frames
        self.game_of_life_surface_cache = None  # Cache rendered pattern to avoid redrawing
        self.game_of_life_cache_frame = -1  # Track when cache was created
        
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
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            # Get desktop resolution for fullscreen
            import pygame
            info = pygame.display.Info()
            self.width = info.current_w
            self.height = info.current_h
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        else:
            # Return to windowed mode
            self.width = 800
            self.height = 600
            self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Kopis Engine - Press F11 for fullscreen")
    
    def world_to_screen(self, world_pos, camera_pos, camera_yaw: float = 0.0, camera_pitch: float = 0.0, camera_roll: float = 0.0, fpv_mode: bool = False) -> Tuple[int, int]:
        """Convert 3D world coordinates to screen coordinates with perspective projection (6DOF)"""
        # Extract coordinates
        if len(world_pos) == 3:
            wx, wy, wz = world_pos
        else:
            wx, wy = world_pos
            wz = 0.0
        
        if len(camera_pos) == 3:
            cx, cy, cz = camera_pos
        else:
            cx, cy = camera_pos
            cz = 0.0
        
        # Calculate relative position
        rel_x = wx - cx
        rel_y = wy - cy
        rel_z = wz - cz
        
        if fpv_mode:
            # First-person view projection with 6DOF (yaw, pitch, roll)
            import math
            
            # Step 1: Rotate by camera yaw (horizontal rotation around Z axis)
            yaw_rad = math.radians(camera_yaw)
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)
            # Rotate around Z axis (yaw)
            temp_x = rel_x * cos_yaw - rel_y * sin_yaw
            temp_y = rel_x * sin_yaw + rel_y * cos_yaw
            temp_z = rel_z
            
            # Step 2: Rotate by camera pitch (vertical rotation around X axis)
            pitch_rad = math.radians(camera_pitch)
            cos_pitch = math.cos(pitch_rad)
            sin_pitch = math.sin(pitch_rad)
            # Rotate around X axis (pitch)
            temp_y2 = temp_y * cos_pitch - temp_z * sin_pitch
            temp_z2 = temp_y * sin_pitch + temp_z * cos_pitch
            temp_x2 = temp_x
            
            # Step 3: Rotate by camera roll (tilt rotation around Y axis) - 6DOF
            roll_rad = math.radians(camera_roll)
            cos_roll = math.cos(roll_rad)
            sin_roll = math.sin(roll_rad)
            # Rotate around Y axis (roll)
            final_x = temp_x2 * cos_roll + temp_z2 * sin_roll
            final_y = temp_y2
            final_z = -temp_x2 * sin_roll + temp_z2 * cos_roll
            
            # Perspective projection (camera looks down +Z axis in camera space)
            # Distance from camera
            distance = math.sqrt(final_x**2 + final_y**2 + final_z**2)
            if distance > 0.1:  # Avoid division by zero
                # Field of view
                fov = 70.0  # degrees
                fov_scale = 1.0 / math.tan(math.radians(fov / 2.0))
                
                # Project to screen (camera looks along +Z, so we use -final_z as depth)
                depth = max(0.1, -final_z)  # Distance along view direction
                screen_x = int(self.width / 2 + final_x * fov_scale * (self.height / depth))
                screen_y = int(self.height / 2 - final_y * fov_scale * (self.height / depth))  # Flip Y
            else:
                screen_x = int(self.width / 2)
                screen_y = int(self.height / 2)
        else:
            # Isometric-style projection (original)
            perspective_scale = 1.0 / (1.0 + rel_z * 0.001)
            screen_x = int(rel_x * perspective_scale + self.width / 2)
            screen_y = int(rel_y * perspective_scale - rel_z * 0.5 + self.height / 2)
        
        return (screen_x, screen_y)
    
    def _render_game_of_life_on_face(self, face_points: List[Tuple[int, int]], game_of_life: 'GameOfLife', base_color: Tuple[int, int, int]):
        """Render Conway's Game of Life pattern on a wall face (optimized)"""
        if len(face_points) < 4:
            return
        
        import numpy as np
        
        # Get bounding box of the face
        xs = [p[0] for p in face_points]
        ys = [p[1] for p in face_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        face_width = max_x - min_x
        face_height = max_y - min_y
        
        # Skip rendering if face is too small or off-screen
        if face_width < 10 or face_height < 10:
            return
        if max_x < 0 or min_x > self.width or max_y < 0 or min_y > self.height:
            return  # Face is completely off-screen
        
        # Get Game of Life pattern
        pattern = game_of_life.get_pattern()
        gol_width = game_of_life.width
        gol_height = game_of_life.height
        
        # Calculate cell size for rendering (skip if cells are too small)
        cell_w = max(1, face_width // gol_width)
        cell_h = max(1, face_height // gol_height)
        
        # Skip rendering if cells are too small (performance optimization)
        if cell_w < 2 or cell_h < 2:
            return  # Cells too small to render (increased threshold for performance)
        
        # Doom-style blood colors: dark red, blood red, dark brown
        # Vary color based on position to create dripping/blood effect
        
        # Calculate visible range
        start_x = max(0, (0 - min_x) // cell_w)
        end_x = min(gol_width, ((self.width - min_x) // cell_w) + 1)
        start_y = max(0, (0 - min_y) // cell_h)
        end_y = min(gol_height, ((self.height - min_y) // cell_h) + 1)
        
        # Render only visible alive cells with Doom blood colors
        # Skip every other cell if cells are small for performance
        skip_cells = 1 if cell_w < 3 or cell_h < 3 else 0
        
        for y in range(start_y, end_y, 1 + skip_cells):
            for x in range(start_x, end_x, 1 + skip_cells):
                if pattern[y, x]:  # Cell is alive (blood)
                    screen_x = min_x + (x * face_width) // gol_width
                    screen_y = min_y + (y * face_height) // gol_height
                    
                    # Only draw if on screen
                    if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                        # Doom-style blood colors: vary intensity for dripping effect
                        # Cells at top are brighter (fresh blood), cells at bottom are darker (dried blood)
                        intensity = 1.0 - (y / gol_height) * 0.4  # Darker as we go down
                        
                        # Doom color palette: dark reds and browns
                        # Base blood red: (139, 0, 0) to (178, 34, 34) to (101, 67, 33)
                        blood_red = int(139 + (y / gol_height) * 39)  # 139-178 (bright to darker red)
                        blood_green = int((y / gol_height) * 34)  # 0-34 (red to brown)
                        blood_blue = int((y / gol_height) * 33)  # 0-33 (red to brown)
                        
                        # Simplified color calculation for performance (no expensive hash)
                        # Use simple modulo for variation instead of MD5 hash
                        variation = ((x * 7 + y * 11) % 31) - 15  # Deterministic variation
                        
                        # Doom blood red colors: bright red at top, dark brown/red at bottom
                        blood_red = max(120, min(200, blood_red + variation))
                        blood_green = max(0, min(60, blood_green + variation // 2))
                        blood_blue = max(0, min(50, blood_blue + variation // 3))
                        
                        blood_color = (blood_red, blood_green, blood_blue)
                        
                        # Draw blood cell (Doom-style dripping blood)
                        pygame.draw.rect(self.screen, blood_color, 
                                       (screen_x, screen_y, cell_w, cell_h))
                        
                        # Add darker outline/shadow for depth (coagulated blood effect)
                        # Only draw outline on larger cells for performance
                        if cell_w > 3 and cell_h > 3:
                            # Darker blood for outline (dried blood look)
                            darker_blood = (max(80, blood_red - 40), max(0, blood_green - 15), max(0, blood_blue - 15))
                            pygame.draw.rect(self.screen, darker_blood, 
                                           (screen_x, screen_y, cell_w, cell_h), 1)
    
    def _is_face_visible(self, face_points: List[Tuple[int, int]]) -> bool:
        """Check if a face is visible (simple back-face culling using winding order)"""
        if len(face_points) < 3:
            return False
        
        # Check if any point is on screen
        on_screen = any(0 <= x < self.width and 0 <= y < self.height for x, y in face_points)
        if not on_screen:
            return False
        
        # Simple check: if the face has a reasonable area, it's likely visible
        # Calculate signed area (winding order)
        area = 0
        for i in range(len(face_points)):
            x1, y1 = face_points[i]
            x2, y2 = face_points[(i + 1) % len(face_points)]
            area += (x2 - x1) * (y2 + y1)
        
        # If area is positive, face is likely front-facing (simplified check)
        return abs(area) > 100  # Minimum area threshold
    
    def _raycast_wall(self, cam_x: float, cam_y: float, ray_dir_x: float, ray_dir_y: float, 
                      max_depth: float = 1000.0) -> Optional[Tuple[float, int]]:
        """
        Raycast to find wall intersection using DDA algorithm (Doom-style).
        Returns (perpendicular_distance, side) where side is 0 for X-side, 1 for Y-side.
        Returns None if no wall found.
        """
        import math
        
        # Avoid division by zero
        if abs(ray_dir_x) < 0.0001:
            ray_dir_x = -0.0001 if ray_dir_x < 0 else 0.0001
        if abs(ray_dir_y) < 0.0001:
            ray_dir_y = -0.0001 if ray_dir_y < 0 else 0.0001
        
        # Get cell size from maze
        cell_size = self.maze.cell_size if self.maze else 50.0
        
        # DDA algorithm
        map_x = int(cam_x / cell_size)
        map_y = int(cam_y / cell_size)
        
        delta_dist_x = abs(1.0 / ray_dir_x)
        delta_dist_y = abs(1.0 / ray_dir_y)
        
        # Determine step direction and initial side distances
        if ray_dir_x < 0:
            step_x = -1
            side_dist_x = (cam_x / cell_size - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - cam_x / cell_size) * delta_dist_x
        
        if ray_dir_y < 0:
            step_y = -1
            side_dist_y = (cam_y / cell_size - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - cam_y / cell_size) * delta_dist_y
        
        # Perform DDA
        hit = False
        side = 0
        perp_wall_dist = 0.0
        steps = 0
        max_steps = int(max_depth / cell_size) + 1
        
        while not hit and steps < max_steps:
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1
            steps += 1
            
            # Check if we hit a wall
            if self.maze:
                chunk_x, chunk_y = self.maze._get_chunk_coords((map_x * cell_size, map_y * cell_size))
                chunk = self.maze._get_or_create_chunk(chunk_x, chunk_y)
                local_x = ((map_x % self.maze.chunk_size) + self.maze.chunk_size) % self.maze.chunk_size
                local_y = ((map_y % self.maze.chunk_size) + self.maze.chunk_size) % self.maze.chunk_size
                
                if (local_x, local_y) in chunk.walls:
                    hit = True
                    if side == 0:
                        perp_wall_dist = (map_x - cam_x / cell_size + (1 - step_x) / 2.0) / ray_dir_x
                    else:
                        perp_wall_dist = (map_y - cam_y / cell_size + (1 - step_y) / 2.0) / ray_dir_y
        
        if hit:
            return (perp_wall_dist, side)
        return None
    
    def render(self, entities: List[GameEntity], camera_pos, frame_count: int, fps: float, 
               camera_yaw: float = 0.0, camera_pitch: float = 0.0, camera_roll: float = 0.0, fpv_mode: bool = False):
        """Render all entities to the screen (3D with perspective, FPV support)"""
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Extract camera X/Y for 2D maze rendering
        if len(camera_pos) == 3:
            camera_pos_2d = (camera_pos[0], camera_pos[1])
        else:
            camera_pos_2d = camera_pos
        
        # Draw maze if available
        if self.maze:
            if fpv_mode:
                # Doom-style raycasting renderer
                import math
                
                FOV = 60.0  # Field of view in degrees
                MAX_DEPTH = 1000.0  # Maximum ray distance
                CEILING_LEVEL = 300.0  # Ceiling height
                CAMERA_HEIGHT_OFFSET = 150.0  # Eye level
                
                # Extract camera position
                if len(camera_pos) == 3:
                    cam_x, cam_y, cam_z = camera_pos
                else:
                    cam_x, cam_y = camera_pos
                    cam_z = 0.0
                
                # Pre-calculate direction vectors
                yaw_rad = math.radians(camera_yaw)
                pitch_rad = math.radians(camera_pitch)
                
                forward_x = math.sin(yaw_rad)
                forward_y = -math.cos(yaw_rad)
                right_x = math.cos(yaw_rad)
                right_y = math.sin(yaw_rad)
                
                # Draw floor and ceiling first (Doom-style)
                half_height = self.height / 2
                horizon_y = half_height - math.tan(pitch_rad) * (self.height / 2)
                
                # Ceiling
                ceiling_top = 0
                ceiling_bottom = max(0, int(horizon_y))
                pygame.draw.rect(self.screen, (26, 26, 26), (0, ceiling_top, self.width, ceiling_bottom - ceiling_top))
                
                # Floor
                floor_top = min(self.height, int(horizon_y))
                floor_bottom = self.height
                pygame.draw.rect(self.screen, (42, 42, 42), (0, floor_top, self.width, floor_bottom - floor_top))
                
                # Render Game of Life blood pattern on floor and ceiling
                # Adaptive sample rate based on FPS for better performance
                if self.shared_game_of_life:
                    pattern = self.shared_game_of_life.get_pattern()
                    gol_height = self.shared_game_of_life.height
                    gol_width = self.shared_game_of_life.width
                    
                    # Adaptive sample rate based on FPS
                    ceiling_sample_rate = 2 if fps < 60 else (3 if fps < 100 else 4)
                    floor_sample_rate = 2 if fps < 60 else (3 if fps < 100 else 4)
                    
                    # Render blood on ceiling (inverted pattern - blood drips down)
                    if ceiling_bottom > ceiling_top:
                        ceiling_height = ceiling_bottom - ceiling_top
                        cell_h = max(1, ceiling_height / gol_height)
                        cell_w = max(1, self.width / gol_width)
                        
                        if cell_h >= 1 and cell_w >= 1:
                            # Sample every N cells for performance (adaptive based on FPS)
                            for y in range(0, gol_height, ceiling_sample_rate):
                                for x in range(0, gol_width, ceiling_sample_rate):
                                    if pattern[y, x]:
                                        screen_x = int(x * self.width / gol_width)
                                        screen_y = ceiling_top + int((gol_height - 1 - y) * ceiling_height / gol_height)  # Inverted for ceiling
                                        if ceiling_top <= screen_y < ceiling_bottom and 0 <= screen_x < self.width:
                                            variation = ((x * 7 + y * 11) % 31) - 15
                                            # Ceiling blood: darker, more dried (inverted y for dripping effect)
                                            blood_red = max(100, min(180, int(139 + ((gol_height - 1 - y) / gol_height) * 39 + variation)))
                                            blood_green = max(0, min(60, int(((gol_height - 1 - y) / gol_height) * 34 + variation // 2)))
                                            blood_blue = max(0, min(50, int(((gol_height - 1 - y) / gol_height) * 33 + variation // 3)))
                                            
                                            # Draw with sample rate width/height to fill gaps
                                            pygame.draw.rect(self.screen, (blood_red, blood_green, blood_blue),
                                                           (screen_x, screen_y, max(1, int(cell_w * ceiling_sample_rate)), 
                                                            max(1, int(cell_h * ceiling_sample_rate))))
                    
                    # Render blood on floor
                    if floor_bottom > floor_top:
                        floor_height = floor_bottom - floor_top
                        cell_h = max(1, floor_height / gol_height)
                        cell_w = max(1, self.width / gol_width)
                        
                        if cell_h >= 1 and cell_w >= 1:
                            # Sample every N cells for performance (adaptive based on FPS)
                            for y in range(0, gol_height, floor_sample_rate):
                                for x in range(0, gol_width, floor_sample_rate):
                                    if pattern[y, x]:
                                        screen_x = int(x * self.width / gol_width)
                                        screen_y = floor_top + int(y * floor_height / gol_height)
                                        if floor_top <= screen_y < floor_bottom and 0 <= screen_x < self.width:
                                            variation = ((x * 7 + y * 11) % 31) - 15
                                            # Floor blood: brighter at top (fresh), darker at bottom (dried)
                                            blood_red = max(120, min(200, int(139 + (y / gol_height) * 39 + variation)))
                                            blood_green = max(0, min(60, int((y / gol_height) * 34 + variation // 2)))
                                            blood_blue = max(0, min(50, int((y / gol_height) * 33 + variation // 3)))
                                            
                                            # Draw with sample rate width/height to fill gaps
                                            pygame.draw.rect(self.screen, (blood_red, blood_green, blood_blue),
                                                           (screen_x, screen_y, max(1, int(cell_w * floor_sample_rate)), 
                                                            max(1, int(cell_h * floor_sample_rate))))
                
                # Raycast for each column
                # Adaptive quality: skip columns for higher FPS
                raycast_skip = 1 if fps < 60 else (2 if fps < 100 else 3)  # Skip 1-3 columns based on FPS
                cell_size = self.maze.cell_size
                ray_results = []
                
                for x in range(0, self.width, raycast_skip):
                    # Calculate ray angle (screen space to world space)
                    camera_x = 2.0 * x / self.width - 1.0  # -1 to 1
                    ray_dir_x = forward_x + right_x * camera_x * math.tan(math.radians(FOV / 2.0))
                    ray_dir_y = forward_y + right_y * camera_x * math.tan(math.radians(FOV / 2.0))
                    
                    # Raycast to find wall
                    result = self._raycast_wall(cam_x, cam_y, ray_dir_x, ray_dir_y, MAX_DEPTH)
                    
                    if result:
                        perp_wall_dist, side = result
                        distance = perp_wall_dist * cell_size
                        
                        # Calculate wall height
                        line_height = abs(self.height / perp_wall_dist) if perp_wall_dist > 0 else self.height
                        draw_start = -line_height / 2 + self.height / 2 + math.tan(pitch_rad) * (self.height / 2)
                        draw_end = line_height / 2 + self.height / 2 + math.tan(pitch_rad) * (self.height / 2)
                        
                        # Calculate wall color based on distance and side
                        brightness = max(0.3, min(1.0, 1.0 - distance / 500.0))
                        
                        # Get wall color (deterministic based on map position)
                        map_x = int((cam_x + ray_dir_x * perp_wall_dist * cell_size) / cell_size)
                        map_y = int((cam_y + ray_dir_y * perp_wall_dist * cell_size) / cell_size)
                        
                        # Generate deterministic color
                        seed = hash(f"{map_x}_{map_y}") % 1000000
                        rng = (seed * 9301 + 49297) % 233280 / 233280
                        base_r = int(30 + rng * 50)
                        base_g = int(25 + rng * 35)
                        base_b = int(20 + rng * 30)
                        
                        # Darker on sides for depth
                        wall_r = int(base_r * brightness * (0.8 if side == 1 else 1.0))
                        wall_g = int(base_g * brightness * (0.8 if side == 1 else 1.0))
                        wall_b = int(base_b * brightness * (0.8 if side == 1 else 1.0))
                        
                        # Draw wall column (wider if skipping columns for performance)
                        wall_width = raycast_skip
                        pygame.draw.rect(self.screen, (wall_r, wall_g, wall_b),
                                       (x, max(0, int(draw_start)), wall_width, 
                                        min(self.height, int(draw_end)) - max(0, int(draw_start))))
                        
                        # Store ray result for sprite rendering
                        ray_results.append({
                            'x': x,
                            'distance': distance,
                            'wall_start': draw_start,
                            'wall_end': draw_end
                        })
                        
                        # Render Game of Life blood pattern on close walls
                        # Adaptive sampling based on FPS
                        blood_sample_rate = 2 if fps < 60 else (3 if fps < 100 else 4)
                        if distance < 200 and x % blood_sample_rate == 0:
                            if self.shared_game_of_life:
                                pattern = self.shared_game_of_life.get_pattern()
                                gol_height = self.shared_game_of_life.height
                                wall_height = draw_end - draw_start
                                cell_h = max(1, wall_height / gol_height)
                                
                                if cell_h >= 1:
                                    gol_width = self.shared_game_of_life.width
                                    pattern_x = int(x / self.width * gol_width)
                                    
                                    for y in range(gol_height):
                                        if pattern[y, pattern_x]:
                                            screen_y = draw_start + (y * wall_height) / gol_height
                                            if draw_start <= screen_y < draw_end:
                                                variation = ((x * 7 + y * 11) % 31) - 15
                                                blood_red = max(120, min(200, int(139 + (y / gol_height) * 39 + variation)))
                                                blood_green = max(0, min(60, int((y / gol_height) * 34 + variation // 2)))
                                                blood_blue = max(0, min(50, int((y / gol_height) * 33 + variation // 3)))
                                                
                                                # Draw blood pixel with width matching raycast skip
                                                pygame.draw.rect(self.screen, (blood_red, blood_green, blood_blue),
                                                               (x, int(screen_y), raycast_skip, max(1, int(cell_h))))
                    else:
                        ray_results.append({'x': x, 'distance': MAX_DEPTH, 'wall_start': 0, 'wall_end': 0})
                
                # Store ray results for sprite rendering
                self.ray_results = ray_results
                
                # Update shared Game of Life (every 10 frames for better performance)
                self.game_of_life_update_counter += 1
                if self.game_of_life_update_counter >= 10:
                    self.game_of_life_update_counter = 0
                    if self.shared_game_of_life:
                        self.shared_game_of_life.update()
                    self.game_of_life_surface_cache = None
                
                # Raycasting complete - old polygon rendering removed
            else:
                # Original 2D wall rendering
                wall_rects = self.maze.get_wall_rects(camera_pos, self.width, self.height)
                for wall_x, wall_y, wall_w, wall_h in wall_rects:
                    # Draw walls at ground level (Z=0)
                    screen_x, screen_y = self.world_to_screen((wall_x, wall_y, 0.0), camera_pos, camera_yaw, camera_pitch, camera_roll, fpv_mode)
                    # Scale wall size based on perspective
                    if len(camera_pos) == 3:
                        cz = camera_pos[2]
                    else:
                        cz = 0.0
                    perspective_scale = 1.0 / (1.0 + abs(cz) * 0.001)
                    wall_w_scaled = int(wall_w * perspective_scale)
                    wall_h_scaled = int(wall_h * perspective_scale)
                    # Draw wall rectangle
                    pygame.draw.rect(self.screen, self.WALL_COLOR, 
                                   (screen_x, screen_y, wall_w_scaled, wall_h_scaled))
                    # Draw wall border
                    pygame.draw.rect(self.screen, (40, 40, 50), 
                                   (screen_x, screen_y, wall_w_scaled, wall_h_scaled), 2)
        else:
            # Draw grid (optional - helps visualize infinite world)
            grid_size = 50
            viewport_min_x = camera_pos_2d[0] - self.width / 2
            viewport_max_x = camera_pos_2d[0] + self.width / 2
            viewport_min_y = camera_pos_2d[1] - self.height / 2
            viewport_max_y = camera_pos_2d[1] + self.height / 2
            
            # Draw grid lines
            start_x = int(viewport_min_x // grid_size) * grid_size
            start_y = int(viewport_min_y // grid_size) * grid_size
            
            for x in range(start_x, int(viewport_max_x) + grid_size, grid_size):
                screen_x, _ = self.world_to_screen((x, 0, 0.0), camera_pos, camera_yaw, camera_pitch, camera_roll, fpv_mode)
                if 0 <= screen_x <= self.width:
                    pygame.draw.line(self.screen, (20, 20, 20), (screen_x, 0), (screen_x, self.height))
            
            for y in range(start_y, int(viewport_max_y) + grid_size, grid_size):
                _, screen_y = self.world_to_screen((0, y, 0.0), camera_pos, camera_yaw, camera_pitch, camera_roll, fpv_mode)
                if 0 <= screen_y <= self.height:
                    pygame.draw.line(self.screen, (20, 20, 20), (0, screen_y), (self.width, screen_y))
        
        # Draw entities as Doom-style sprites (billboards)
        if fpv_mode:
            # Don't render player in FPV mode
            entities_to_render = [e for e in entities if e.id != 'player']
            
            # Calculate sprite positions and distances
            import math
            sprites = []
            for entity in entities_to_render:
                if not entity or len(entity.position) < 2:
                    continue
                
                # Extract positions
                ex, ey = entity.position[0], entity.position[1]
                ez = entity.position[2] if len(entity.position) == 3 else 0.0
                
                if len(camera_pos) == 3:
                    cx, cy, cz = camera_pos
                else:
                    cx, cy = camera_pos
                    cz = 0.0
                
                dx = ex - cx
                dy = ey - cy
                dz = ez - cz
                
                # Transform to camera space
                yaw_rad = math.radians(camera_yaw)
                pitch_rad = math.radians(camera_pitch)
                
                # Rotate by yaw
                temp_x = dx * math.cos(yaw_rad) - dy * math.sin(yaw_rad)
                temp_y = dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
                temp_z = dz
                
                # Rotate by pitch
                final_y = temp_y * math.cos(pitch_rad) - temp_z * math.sin(pitch_rad)
                final_z = temp_y * math.sin(pitch_rad) + temp_z * math.cos(pitch_rad)
                
                # Project to screen
                FOV = 60.0
                fov_scale = 1.0 / math.tan(math.radians(FOV / 2.0))
                depth = max(0.1, -temp_x)
                
                if depth <= 0 or temp_x < 0:
                    continue  # Behind camera
                
                screen_x = self.width / 2 + temp_y * fov_scale * (self.height / depth)
                screen_y = self.height / 2 - final_y * fov_scale * (self.height / depth)
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                
                sprites.append({
                    'entity': entity,
                    'screen_x': screen_x,
                    'screen_y': screen_y,
                    'distance': distance,
                    'depth': depth
                })
            
            # Sort by distance (farthest first for proper depth)
            sprites.sort(key=lambda s: s['distance'], reverse=True)
            
            # Render sprites
            for sprite in sprites:
                entity = sprite['entity']
                is_player = entity.id == 'player'
                color = self.BLUE if is_player else (self.RED if 'npc' in entity.id else self.GREEN)
                base_radius = 15 if is_player else (10 if 'npc' in entity.id else 8)
                
                # Calculate sprite size based on distance
                sprite_size = max(5, min(50, base_radius * 2 * (self.height / sprite['depth'])))
                sprite_x = sprite['screen_x'] - sprite_size / 2
                sprite_y = sprite['screen_y'] - sprite_size / 2
                
                # Check if sprite is on screen
                if sprite_x + sprite_size < 0 or sprite_x > self.width or \
                   sprite_y + sprite_size < 0 or sprite_y > self.height:
                    continue
                
                # Check if sprite is behind a wall (simple depth check)
                screen_x_int = int(sprite['screen_x'])
                if 0 <= screen_x_int < self.width and hasattr(self, 'ray_results'):
                    ray = next((r for r in self.ray_results if r['x'] == screen_x_int), None)
                    if ray and sprite['distance'] > ray['distance']:
                        continue  # Behind wall
                
                # Brightness based on distance
                brightness = max(0.5, min(1.0, 1.0 - sprite['distance'] / 500.0))
                
                # Draw sprite as circle (Doom-style)
                alpha = int(255 * brightness)
                sprite_surface = pygame.Surface((int(sprite_size * 2), int(sprite_size * 2)), pygame.SRCALPHA)
                
                # Draw glow
                pygame.draw.circle(sprite_surface, (*color, alpha // 2), 
                                 (int(sprite_size), int(sprite_size)), int(sprite_size))
                
                # Draw sprite body
                pygame.draw.circle(sprite_surface, (*color, alpha), 
                                 (int(sprite_size), int(sprite_size)), int(sprite_size // 2))
                pygame.draw.circle(sprite_surface, (255, 255, 255, alpha), 
                                 (int(sprite_size), int(sprite_size)), int(sprite_size // 2), 2)
                
                self.screen.blit(sprite_surface, (int(sprite_x), int(sprite_y)))
        else:
            # 2D fallback rendering
            sorted_entities = sorted(entities, key=lambda e: e.position[2] if len(e.position) == 3 else 0.0)
            
            for entity in sorted_entities:
                screen_pos = self.world_to_screen(entity.position, camera_pos, camera_yaw, camera_pitch, camera_roll, fpv_mode)
                
                if 0 <= screen_pos[0] <= self.width and 0 <= screen_pos[1] <= self.height:
                    if entity.id == 'player':
                        color = self.BLUE
                        base_radius = 15
                    elif 'npc' in entity.id:
                        color = self.RED
                        base_radius = 10
                    else:
                        color = self.GREEN
                        base_radius = 8
                    
                    if len(entity.position) == 3:
                        ez = entity.position[2]
                    else:
                        ez = 0.0
                    if len(camera_pos) == 3:
                        cz = camera_pos[2]
                    else:
                        cz = 0.0
                    rel_z = ez - cz
                    perspective_scale = 1.0 / (1.0 + abs(rel_z) * 0.001)
                    radius = int(base_radius * perspective_scale)
                    
                    pygame.draw.circle(self.screen, color, screen_pos, radius)
                    pygame.draw.circle(self.screen, self.WHITE, screen_pos, radius, 2)
        
        # Draw UI overlay
        self._draw_ui(camera_pos, frame_count, fps, len(entities), camera_yaw, camera_pitch, fpv_mode)
        
        # Update display
        pygame.display.flip()
        # Higher target FPS (120 FPS) - tick() without argument allows unlimited FPS
        # Use tick_busy_loop for more accurate timing on high refresh rate displays
        self.clock.tick_busy_loop(120)
    
    def _draw_ui(self, camera_pos, frame_count: int, fps: float, entity_count: int, 
                 camera_yaw: float = 0.0, camera_pitch: float = 0.0, camera_roll: float = 0.0, fpv_mode: bool = False):
        """Draw UI overlay with game information (3D, FPV)"""
        # Background for UI
        ui_surface = pygame.Surface((200, 140))
        ui_surface.set_alpha(180)
        ui_surface.fill((0, 0, 0))
        self.screen.blit(ui_surface, (10, 10))
        
        # Extract camera coordinates
        if len(camera_pos) == 3:
            cx, cy, cz = camera_pos
        else:
            cx, cy = camera_pos
            cz = 0.0
        
        # Draw text
        view_mode = "FPV" if fpv_mode else "3rd Person"
        texts = [
            f"Frame: {frame_count}",
            f"FPS: {fps:.1f}",
            f"Entities: {entity_count}",
            f"Camera: ({cx:.0f}, {cy:.0f}, {cz:.0f})",
            f"Yaw: {camera_yaw:.1f}° Pitch: {camera_pitch:.1f}° Roll: {camera_roll:.1f}°",
            f"6DOF: Q/E for roll",
            f"View: {view_mode}",
            f"Controls: WASD + SPACE"
        ]
        
        y_offset = 15
        for text in texts:
            text_surface = self.small_font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (15, y_offset))
            y_offset += 20
    
    def get_input(self, fpv_mode: bool = False) -> Dict[str, Any]:
        """Get input from pygame events (3D with jumping and mouse look)"""
        keys_pressed = {
            'w': False, 'a': False, 's': False, 'd': False, 'space': False, 'esc': False
        }
        
        # Get current mouse position (always available for direction-based movement)
        mouse_pos = pygame.mouse.get_pos()
        mouse_delta = (0, 0)  # Mouse movement delta for camera rotation
        mouse_clicked = False
        quit_requested = False
        
        # Capture mouse in FPV mode for 6DOF camera control
        if fpv_mode:
            if not self.mouse_captured:
                # Hide mouse cursor and capture it for 6DOF camera control
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)  # Lock mouse to window
                self.mouse_captured = True
                # Center mouse cursor
                pygame.mouse.set_pos(self.width // 2, self.height // 2)
            else:
                # Recenter mouse each frame to get continuous relative movement
                center_x, center_y = self.width // 2, self.height // 2
                current_pos = pygame.mouse.get_pos()
                if abs(current_pos[0] - center_x) > 10 or abs(current_pos[1] - center_y) > 10:
                    pygame.mouse.set_pos(center_x, center_y)
        else:
            if self.mouse_captured:
                # Release mouse capture when not in FPV mode
                pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)
                self.mouse_captured = False
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            
            if event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
                # Calculate mouse delta for camera rotation (6DOF)
                if hasattr(event, 'rel') and self.mouse_captured:
                    # Use relative motion when mouse is captured (best for 6DOF)
                    mouse_delta = event.rel
                elif self.mouse_captured:
                    # If relative motion not available, calculate from center
                    center_x, center_y = self.width // 2, self.height // 2
                    mouse_delta = (mouse_pos[0] - center_x, mouse_pos[1] - center_y)
                else:
                    # Mouse not captured, calculate delta from previous position
                    prev_mouse_pos = getattr(self, '_prev_mouse_pos', mouse_pos)
                    mouse_delta = (mouse_pos[0] - prev_mouse_pos[0], mouse_pos[1] - prev_mouse_pos[1])
                    self._prev_mouse_pos = mouse_pos
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_clicked = True
                # Clicking in FPV mode can also capture mouse if not already captured
                if fpv_mode and not self.mouse_captured:
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
                    self.mouse_captured = True
                    pygame.mouse.set_pos(self.width // 2, self.height // 2)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Toggle mouse capture with ESC
                    if self.mouse_captured:
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False)
                        self.mouse_captured = False
                    keys_pressed['esc'] = True
                elif event.key == pygame.K_F11:
                    # Toggle fullscreen with F11
                    self.toggle_fullscreen()
        
        # Check for held keys using pygame.key.get_pressed() - this is the reliable way
        keys = pygame.key.get_pressed()
        keys_pressed['w'] = keys[pygame.K_w]
        keys_pressed['a'] = keys[pygame.K_a]
        keys_pressed['s'] = keys[pygame.K_s]
        keys_pressed['d'] = keys[pygame.K_d]
        keys_pressed['space'] = keys[pygame.K_SPACE]
        keys_pressed['q'] = keys[pygame.K_q]  # Roll left (6DOF)
        keys_pressed['e'] = keys[pygame.K_e]  # Roll right (6DOF)
        
        return {
            'quit': quit_requested,
            'keys': keys_pressed,
            'mouse': {
                'x': mouse_pos[0], 
                'y': mouse_pos[1], 
                'dx': mouse_delta[0],  # Mouse delta X for camera yaw
                'dy': mouse_delta[1],  # Mouse delta Y for camera pitch
                'clicked': mouse_clicked
            }
        }
    
    def cleanup(self):
        """Clean up pygame resources"""
        if PYGAME_AVAILABLE:
            # Release mouse capture before quitting
            if self.mouse_captured:
                pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)
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
    
    # Show epilepsy warning before starting game
    def show_epilepsy_warning():
        """Display epilepsy warning screen and wait for user acknowledgment"""
        warning_screen = renderer.screen
        clock = pygame.time.Clock()
        
        # Warning colors
        RED = (255, 68, 68)
        DARK_RED = (204, 0, 0)
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        DARK_BG = (26, 26, 46)
        
        # Create warning font
        try:
            title_font = pygame.font.Font(None, 72)
            text_font = pygame.font.Font(None, 36)
            button_font = pygame.font.Font(None, 48)
        except:
            title_font = pygame.font.SysFont('arial', 72, bold=True)
            text_font = pygame.font.SysFont('arial', 36)
            button_font = pygame.font.SysFont('arial', 48, bold=True)
        
        acknowledged = False
        
        while not acknowledged:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE or event.key == pygame.K_ESCAPE:
                        acknowledged = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    acknowledged = True
            
            # Clear screen with dark background
            warning_screen.fill(DARK_BG)
            
            # Draw warning border
            pygame.draw.rect(warning_screen, RED, (50, 50, renderer.width - 100, renderer.height - 100), 5)
            
            # Warning icon (⚠)
            icon_text = text_font.render("⚠", True, RED)
            icon_rect = icon_text.get_rect(center=(renderer.width // 2, 150))
            warning_screen.blit(icon_text, icon_rect)
            
            # Title
            title_text = title_font.render("EPILEPSY WARNING", True, RED)
            title_rect = title_text.get_rect(center=(renderer.width // 2, 250))
            warning_screen.blit(title_text, title_rect)
            
            # Warning message
            message_lines = [
                "This application contains flashing lights, rapid visual changes,",
                "and intense visual effects that may trigger seizures in people",
                "with photosensitive epilepsy or other photosensitive conditions.",
                "",
                "If you have a history of epilepsy or are sensitive to flashing",
                "lights, please use caution or avoid using this application."
            ]
            
            y_offset = 350
            for line in message_lines:
                if line:  # Skip empty lines
                    text = text_font.render(line, True, WHITE)
                    text_rect = text.get_rect(center=(renderer.width // 2, y_offset))
                    warning_screen.blit(text, text_rect)
                y_offset += 45
            
            # Button
            button_text = button_font.render("Press ENTER, SPACE, or CLICK to Continue", True, WHITE)
            button_rect = button_text.get_rect(center=(renderer.width // 2, renderer.height - 150))
            
            # Draw button background
            button_bg_rect = pygame.Rect(button_rect.x - 20, button_rect.y - 10, 
                                        button_rect.width + 40, button_rect.height + 20)
            pygame.draw.rect(warning_screen, DARK_RED, button_bg_rect)
            pygame.draw.rect(warning_screen, RED, button_bg_rect, 3)
            warning_screen.blit(button_text, button_rect)
            
            pygame.display.flip()
            clock.tick(60)
        
        return True
    
    # Show warning and check if user wants to continue
    if not show_epilepsy_warning():
        print("Game cancelled by user")
        return
    
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
    
    # Final verification - ensure player is definitely not in a wall
    player_radius = 12.0
    safety_margin = 5.0
    if maze.check_collision(start_world, player_radius):
        print("⚠ Warning: Starting position may be in wall, attempting to find safe position...")
        # Try to find any safe position with extra margin
        for attempts in range(200):
            test_x = (random.random() - 0.5) * 2000
            test_y = (random.random() - 0.5) * 2000
            # Use larger margin to ensure player can move
            if not maze.check_collision((test_x, test_y), player_radius + safety_margin):
                # Double-check it's actually a path cell
                test_cell = maze.world_to_cell((test_x, test_y))
                if maze.is_path_cell(test_cell):
                    start_world = (test_x, test_y)
                    start_cell = maze.world_to_cell(start_world)
                    print(f"✓ Found safe position at ({test_x:.1f}, {test_y:.1f})")
                    break
    
    # Create player entity at valid maze position
    player = GameEntity(
        id='player',
        position=(start_world[0], start_world[1], 0.0),  # 3D position (Z starts at ground)
        velocity=(0.0, 0.0, 0.0),  # 3D velocity
        health=100.0,
        description="Player character",
        properties={
            'affected_by_gravity': True,  # Player affected by gravity for jumping
            'radius': 12.0,  # Player collision radius
            'on_ground': True  # Start on ground
        }
    )
    engine.add_entity(player)
    
    # Find a nearby wall and set camera to look at it
    import math
    player_x, player_y = start_world[0], start_world[1]
    start_cell = maze.world_to_cell(start_world)
    cell_size = maze.cell_size
    
    # Search for walls in 4 cardinal directions
    search_radius = 5  # Search up to 5 cells away
    nearest_wall = None
    nearest_distance = float('inf')
    
    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            if dx == 0 and dy == 0:
                continue  # Skip the player's own cell
            
            test_cell = (start_cell[0] + dx, start_cell[1] + dy)
            test_world = maze.cell_to_world(test_cell)
            
            # Check if this cell is a wall
            chunk_x = test_cell[0] // maze.chunk_size
            chunk_y = test_cell[1] // maze.chunk_size
            if test_cell[0] < 0:
                chunk_x = (test_cell[0] - maze.chunk_size + 1) // maze.chunk_size
            if test_cell[1] < 0:
                chunk_y = (test_cell[1] - maze.chunk_size + 1) // maze.chunk_size
            
            chunk = maze._get_or_create_chunk(chunk_x, chunk_y)
            local_x = test_cell[0] % maze.chunk_size
            local_y = test_cell[1] % maze.chunk_size
            if test_cell[0] < 0:
                local_x = (test_cell[0] % maze.chunk_size + maze.chunk_size) % maze.chunk_size
            if test_cell[1] < 0:
                local_y = (test_cell[1] % maze.chunk_size + maze.chunk_size) % maze.chunk_size
            
            # Check if it's a wall
            if (local_x, local_y) in chunk.walls:
                # Calculate distance to wall center
                wall_world_x, wall_world_y = test_world
                wall_center_x = wall_world_x + cell_size / 2
                wall_center_y = wall_world_y + cell_size / 2
                distance = math.sqrt((wall_center_x - player_x)**2 + (wall_center_y - player_y)**2)
                
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_wall = (wall_center_x, wall_center_y)
    
    # Set camera yaw to look at the nearest wall
    if nearest_wall:
        wall_x, wall_y = nearest_wall
        # Calculate angle from player to wall (in degrees)
        dx = wall_x - player_x
        dy = wall_y - player_y
        angle_rad = math.atan2(dy, dx)  # atan2 gives angle in radians (0 = +X, 90 = +Y)
        angle_deg = math.degrees(angle_rad)
        # Convert to camera yaw system:
        # Camera forward: forward_x = sin(yaw), forward_y = -cos(yaw)
        # So yaw = 0 means looking along -Y (up), yaw = 90 means looking along +X (right)
        # To convert from atan2 angle (0 = +X, 90 = +Y) to camera yaw:
        # We need: atan2 angle + 90 = camera yaw (because camera yaw 0 = -Y, which is atan2 -90)
        camera_yaw = angle_deg + 90.0
        # Rotate by 180 degrees
        camera_yaw += 180.0
        # Normalize to 0-360 range
        camera_yaw = camera_yaw % 360.0
        engine.camera_yaw = camera_yaw
        print(f"✓ Set initial camera to look at wall at ({wall_x:.1f}, {wall_y:.1f}), yaw: {camera_yaw:.1f}°")
    else:
        # Fallback: look in a random direction, rotated by 180 degrees
        engine.camera_yaw = (random.uniform(0, 360) + 180.0) % 360.0
        print(f"⚠ No nearby wall found, set random camera yaw: {engine.camera_yaw:.1f}°")
    
    # Create many NPCs at different valid positions near the start
    # Performance optimized: can handle many NPCs with staggered pathfinding
    MAX_NPCS = 25  # Reduced from 50 to 25 for better FPS with Game of Life rendering
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
            position=(npc_world[0], npc_world[1], 0.0),  # 3D position (Z starts at ground)
            velocity=(0.0, 0.0, 0.0),  # 3D velocity
            health=50.0,
            description=f"NPC {i+1}",
            properties={
                'affected_by_gravity': True,  # NPCs affected by gravity
                'radius': 8.0,  # NPC collision radius
                'path': None,  # Pathfinding path
                'path_index': 0,  # Current waypoint index
                'last_path_update': 0.0,  # Last time path was calculated
                'on_ground': True  # Start on ground
            }
        )
        engine.add_entity(npc)
    
    print("\n" + "="*60)
    print("KOPIS ENGINE - Pygame Visualization with Maze (3D FPV)")
    print("="*60)
    print("Controls: W/A/S/D to move, SPACE to jump, Mouse to look around (6DOF)")
    print("Mouse is captured in FPV mode - ESC releases capture, click to recapture")
    print("6DOF Camera: Mouse for yaw/pitch (unlimited 360°), Q/E for roll")
    print("F11: Toggle fullscreen mode")
    print("NPCs will automatically move towards the player")
    print("Navigate through the maze - avoid walls!")
    print("Sound effects: footsteps and collisions")
    print("3D rendering with first-person view (FPV)")
    print("="*60)
    
    running = True
    frame = 0
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 60.0
    
    try:
        while running:
            # Get input from pygame (pass FPV mode for mouse capture)
            input_data = renderer.get_input(fpv_mode=engine.fpv_mode)
            
            if input_data.get('quit', False):
                running = False
                break
            
            # Handle ESC key to toggle mouse capture or quit
            if input_data.get('keys', {}).get('esc', False):
                # ESC can be used to release mouse if needed
                # (Mouse capture is automatically managed based on FPV mode)
                pass
            
            # Store screen dimensions in engine for movement calculation
            engine.screen_width = renderer.width
            engine.screen_height = renderer.height
            
            # Process frame
            result = engine.process_frame(input_data)
            frame += 1
            fps_counter += 1
            
            # Calculate FPS more accurately
            now = time.time()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                current_fps = round((fps_counter * 1.0) / elapsed)
                fps_counter = 0
                fps_timer = now
            
            # Get camera position and rotation from engine (smoothly follows player) - 6DOF
            camera_pos = engine.camera_pos
            camera_yaw = engine.camera_yaw
            camera_pitch = engine.camera_pitch
            camera_roll = engine.camera_roll
            fpv_mode = engine.fpv_mode
            
            # Render with pygame (FPV mode with 6DOF)
            renderer.render(engine.entities, camera_pos, frame, current_fps, camera_yaw, camera_pitch, camera_roll, fpv_mode)
            
            # No sleep needed - clock.tick_busy_loop handles timing
            # This allows higher FPS while maintaining smooth frame pacing
    
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
