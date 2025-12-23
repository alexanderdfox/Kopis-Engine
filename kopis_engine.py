"""
Kopis Engine - Transformer-Based Game Engine
A game engine architecture based on transformer circuits with:
- Stacked transformers for state processing
- Parallel branches for multi-system processing
- NAND gate for logical operations
- Feedback loop for state persistence

USAGE:
    python kopis_engine.py              # Run the game with default settings
    python kopis_engine.py --help       # Show help and options
    
CONTROLS:
    W/A/S/D     - Move forward/left/backward/right (relative to camera direction)
    SPACE       - Jump
    Mouse       - Look around (camera rotation)
    Left Click  - Swing sword
    Q/E         - Roll camera (tilt left/right)
    ESC         - Release mouse cursor
    F11         - Toggle fullscreen
    
GAMEPLAY:
    - Navigate through an infinite procedurally generated 3D maze
    - Fight NPCs with your sword (left click or tap)
    - Avoid NPC contact or you'll take damage (5 HP per touch)
    - Sword deals 20 damage per hit to NPCs
    - Each story is 10 feet tall, player is 6 feet tall
    - Find stairs and holes to explore multiple levels
    
REQUIREMENTS:
    - Python 3.7+
    - pygame (install with: pip install pygame)
    - Optional: torch and transformers (for advanced AI features)
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
import argparse

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("=" * 60)
    print("ERROR: pygame is not installed")
    print("=" * 60)
    print("Kopis Engine requires pygame to run.")
    print("Please install it with:")
    print("  pip install pygame")
    print("\nOr using pip3:")
    print("  pip3 install pygame")
    print("=" * 60)


class GameState(Enum):
    """Game state enumeration"""
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    VICTORY = "victory"


# Constants for height measurements
FEET_TO_UNITS = 30.0  # 1 foot = 30 units
STORY_HEIGHT_UNITS = 10.0 * FEET_TO_UNITS  # 10 feet = 300 units
PLAYER_HEIGHT_UNITS = 6.0 * FEET_TO_UNITS  # 6 feet = 180 units

@dataclass
class GameEntity:
    """Represents a game entity (player, NPC, object) - 3D coordinates"""
    id: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, z) - 3D position
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (vx, vy, vz) - 3D velocity
    health: float = 100.0
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    height: float = PLAYER_HEIGHT_UNITS  # Entity height in units (default 6 feet = 180 units)


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
                print("    Loading transformer model (this may take 10-30 seconds)...")
                print("    Note: First-time download may take longer (model size ~250MB)")
                # Use CPU by default to avoid bus errors and resource conflicts
                model_id = 'distilbert-base-uncased-finetuned-sst-2-english'
                # Force CPU to prevent multiprocessing issues that cause bus errors
                StackedTransformers._shared_pipeline = pipeline(
                    'text-classification',
                    model=model_id,
                    device=-1,  # Always use CPU to avoid bus errors
                    torch_dtype=torch.float32
                )
                print("    ✓ Transformer model loaded successfully (CPU mode for stability)")
            except Exception as e:
                print(f"    ⚠ Warning: Could not load transformer model: {e}")
                print("    ✓ Falling back to simple processing mode (game will still work)")
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
        # No hard ground/ceiling limits - infinite levels supported
        # Ground and ceiling heights are determined per-cell by maze geometry
        
        for entity in entities:
            # Get current state (3D)
            x, y, z = entity.position
            vx, vy, vz = entity.velocity
            
            # Apply gravity (if entity is affected by gravity) - affects Z axis
            # Positive Z is up, so gravity pulls down (decreases upward velocity)
            # When vz is positive (moving up), gravity reduces it
            # When vz is negative (falling), gravity makes it more negative (faster fall)
            if entity.properties.get('affected_by_gravity', True):
                # Always apply gravity - this ensures entities fall when in the air
                # Only stop falling when hitting the ground (checked in collision below)
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
            
            # Ground and ceiling collision (Z axis) - check per-cell heights from maze
            if maze:
                # Get the cell we're in
                cell_x = int(new_x / maze.cell_size)
                cell_y = int(new_y / maze.cell_size)
                chunk_x, chunk_y = maze._get_chunk_coords((new_x, new_y))
                chunk = maze._get_or_create_chunk(chunk_x, chunk_y)
                local_x = ((cell_x % maze.chunk_size) + maze.chunk_size) % maze.chunk_size
                local_y = ((cell_y % maze.chunk_size) + maze.chunk_size) % maze.chunk_size
                
                # Calculate which level we're on (each level is 10 feet = 300 units high)
                level = int(new_z / STORY_HEIGHT_UNITS)
                base_floor_z = level * STORY_HEIGHT_UNITS
                base_ceiling_z = (level + 1) * STORY_HEIGHT_UNITS
                
                # Check for stairs - calculate floor height based on stair slope
                stair_info = chunk.stairs.get((local_x, local_y))
                floor_z = base_floor_z
                if stair_info:
                    # Calculate position within the cell (0.0 to 1.0)
                    # For simplicity, use a simple linear interpolation across the cell
                    # In a more advanced implementation, we'd track which direction the stairs face
                    cell_world_x = cell_x * maze.cell_size
                    cell_world_y = cell_y * maze.cell_size
                    cell_center_x = cell_world_x + maze.cell_size / 2.0
                    cell_center_y = cell_world_y + maze.cell_size / 2.0
                    
                    # Determine stair height based on direction and position
                    target_level_offset = stair_info['target_level']
                    stair_height_range = STORY_HEIGHT_UNITS * target_level_offset  # Total height change
                    
                    # For up stairs, floor slopes from current level to next level
                    # For down stairs, floor slopes from current level to previous level
                    # Calculate how far along the stair we are (using X position within cell)
                    progress = (new_x - cell_world_x) / maze.cell_size
                    # Clamp progress to [0, 1]
                    if progress < 0.0:
                        progress = 0.0
                    elif progress > 1.0:
                        progress = 1.0
                    
                    if stair_info['direction'] == 'up':
                        # Stairs go from floor_z to floor_z + STORY_HEIGHT_UNITS
                        floor_z = base_floor_z + progress * STORY_HEIGHT_UNITS
                    else:  # down
                        # Stairs go from floor_z to floor_z - STORY_HEIGHT_UNITS
                        floor_z = base_floor_z - progress * STORY_HEIGHT_UNITS
                    
                    ceiling_z = floor_z + STORY_HEIGHT_UNITS  # Ceiling is always 10 feet above floor
                else:
                    ceiling_z = base_ceiling_z
                
                # Solid floor collision (no holes allow falling through)
                if new_z < floor_z:
                    # Hit or below floor - snap to floor and stop falling
                    new_z = floor_z
                    if vz < 0:  # Only stop downward velocity if hitting ground
                        vz = 0
                    entity.properties['on_ground'] = True
                elif abs(new_z - floor_z) < 2.0 and abs(vz) < 10.0:  # Very close to floor and slow velocity
                    # Snapping to floor when very close and slow (prevents jitter)
                    new_z = floor_z
                    if vz < 0:
                        vz = 0
                    entity.properties['on_ground'] = True
                else:
                    # In the air - allow falling
                    entity.properties['on_ground'] = False
                
                # Solid ceiling collision (no holes allow passing through)
                if new_z > ceiling_z:
                    new_z = ceiling_z
                    if vz > 0:  # Only stop upward velocity if hitting ceiling
                        vz = 0  # Stop upward velocity, gravity will pull down
                    entity.properties['on_ground'] = False
            else:
                # No maze - use simple ground level check
                if new_z < 0.0:
                    new_z = 0.0
                    if vz < 0:
                        vz = 0
                    entity.properties['on_ground'] = True
                else:
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
        self.npc_damage_cooldowns = {}  # Track damage cooldowns per NPC
        
        # Sword swing state
        self.sword_swing_state = 0.0  # 0.0 = idle, 0.0-1.0 = swing animation progress
        self.sword_swing_speed = 8.0  # Animation speed (swings per second when active)
        self.sword_damage_cooldowns = {}  # Track damage cooldowns per entity (prevent multiple hits per swing)
        
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
    
    def _check_sword_damage(self, delta_time: float):
        """Check if sword is hitting entities during swing and apply damage"""
        if not self.player:
            return
        
        import math
        import time
        
        # Only apply damage during active swing (middle portion of animation for best feel)
        swing_progress = self.sword_swing_state
        if swing_progress < 0.2 or swing_progress > 0.8:
            # Too early or too late in swing - no damage
            return
        
        current_time = time.time()
        player_pos = self.player.position
        player_radius = self.player.properties.get('radius', 12.0)
        
        # Sword reach: extends forward from player position
        # Calculate forward direction based on camera yaw
        yaw_rad = math.radians(self.camera_yaw)
        forward_x = math.sin(yaw_rad)
        forward_y = -math.cos(yaw_rad)
        
        # Sword range: about 100-150 units in front of player
        sword_range = 120.0  # units
        sword_hit_radius = 40.0  # Wider hit radius for sword swing
        
        # Calculate sword tip position (in front of player)
        sword_tip_x = player_pos[0] + forward_x * sword_range
        sword_tip_y = player_pos[1] + forward_y * sword_range
        sword_tip_z = player_pos[2] if len(player_pos) == 3 else 0.0
        
        # Check all entities for damage
        for entity in self.entities:
            # Skip player and already damaged entities this swing
            if entity.id == 'player':
                continue
            
            # Check cooldown (each entity can only be hit once per swing)
            if entity.id in self.sword_damage_cooldowns:
                continue
            
            # Get entity position
            entity_pos = entity.position
            entity_radius = entity.properties.get('radius', 8.0) if hasattr(entity, 'properties') else 8.0
            
            # Calculate distance from sword tip to entity center
            dx = entity_pos[0] - sword_tip_x
            dy = entity_pos[1] - sword_tip_y
            dz = (entity_pos[2] if len(entity_pos) == 3 else 0.0) - sword_tip_z
            
            # Check if entity is in sword hit radius
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            combined_radius = sword_hit_radius + entity_radius
            
            if distance < combined_radius:
                # Hit! Apply damage
                damage = 20.0  # Sword does 20 damage per hit
                entity.health = max(0.0, entity.health - damage)
                
                # Mark as damaged this swing (prevent multiple hits)
                self.sword_damage_cooldowns[entity.id] = current_time
                
                # Play sound effect if available
                if self.sound_manager:
                    self.sound_manager.play('collision', volume=0.6)
                
                # Update entity in entities list
                for i, e in enumerate(self.entities):
                    if e.id == entity.id:
                        self.entities[i] = entity
                        break
                
                print(f"⚔ Sword hit {entity.id} for {damage} damage! ({entity.health:.1f} HP remaining)")
    
    def _check_npc_collisions(self, delta_time: float):
        """
        Check if player is touching any NPCs and apply damage with cooldown
        Player loses 5 health when touching an NPC, with 0.5 second cooldown per NPC
        """
        if not self.player:
            return
        
        import math
        import time
        
        current_time = time.time()
        player_pos = self.player.position
        player_radius = self.player.properties.get('radius', 12.0)
        
        for entity in self.entities:
            # Only check NPCs
            if entity.id == 'player' or 'npc' not in entity.id:
                continue
            
            npc_pos = entity.position
            npc_radius = entity.properties.get('radius', 8.0)
            
            # Calculate 3D distance
            dx = player_pos[0] - npc_pos[0]
            dy = player_pos[1] - npc_pos[1]
            dz = player_pos[2] - npc_pos[2] if len(player_pos) == 3 and len(npc_pos) == 3 else 0.0
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            
            # Check if touching (combined radii)
            combined_radius = player_radius + npc_radius
            if distance < combined_radius:
                # Check cooldown (0.5 seconds per NPC)
                last_damage_time = self.npc_damage_cooldowns.get(entity.id, 0)
                cooldown_duration = 0.5  # 0.5 seconds
                
                if current_time - last_damage_time >= cooldown_duration:
                    # Apply damage
                    self.player.health = max(0.0, self.player.health - 5.0)
                    self.npc_damage_cooldowns[entity.id] = current_time
                    
                    # Check if player is dead
                    if self.player.health <= 0.0:
                        self.game_state = GameState.GAME_OVER
                    
                    # Update player entity in the entities list
                    for i, e in enumerate(self.entities):
                        if e.id == 'player':
                            updated_player = GameEntity(
                                id=self.player.id,
                                position=self.player.position,
                                velocity=self.player.velocity,
                                health=self.player.health,
                                description=self.player.description,
                                properties=self.player.properties,
                                height=getattr(self.player, 'height', PLAYER_HEIGHT_UNITS)
                            )
                            self.entities[i] = updated_player
                            self.player = updated_player
                            break
    
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
        
        # Update sword swing animation
        if self.sword_swing_state > 0.0:
            self.sword_swing_state += self.sword_swing_speed * delta_time
            if self.sword_swing_state >= 1.0:
                self.sword_swing_state = 0.0  # Animation complete, return to idle
                # Reset damage cooldowns when swing completes (allow new swing to damage)
                self.sword_damage_cooldowns = {}
        
        # Check for sword swing input (mouse left button or touch)
        mouse = input_data.get('mouse', {})
        if isinstance(mouse, dict):
            mouse_clicked = mouse.get('clicked', False) or mouse.get('left_button', False)
            # Also check for touch on iOS/mobile
            touch_active = mouse.get('touch', False) or mouse.get('tap', False)
            if (mouse_clicked or touch_active) and self.sword_swing_state == 0.0:
                # Start sword swing animation
                self.sword_swing_state = 0.01  # Small value to start animation
                # Reset damage cooldowns for new swing
                self.sword_damage_cooldowns = {}
        
        # Check for sword damage during swing
        if self.sword_swing_state > 0.0 and self.sword_swing_state < 1.0:
            self._check_sword_damage(delta_time)
        
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
        
        # Check for player-NPC collisions and apply damage
        if self.player:
            self._check_npc_collisions(delta_time)
        
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
        # Floor and ceiling heights per cell (z values in world space)
        # Default: floor_z = 0, ceiling_z = STORY_HEIGHT_UNITS for each level (10 feet = 300 units)
        self.floor_heights = {}  # (local_x, local_y) -> floor_z (default: level * STORY_HEIGHT_UNITS)
        self.ceiling_heights = {}  # (local_x, local_y) -> ceiling_z (default: (level + 1) * STORY_HEIGHT_UNITS)
        # Stairs: (local_x, local_y) -> {'direction': 'up' or 'down', 'target_level': int}
        self.stairs = {}  # (local_x, local_y) -> {'direction': 'up'|'down', 'target_level': int}
        # Holes: (local_x, local_y) -> True (floor hole) or 'ceiling' (ceiling hole)
        self.holes = {}  # (local_x, local_y) -> 'floor' or 'ceiling' or None
        # Windows: set of (local_x, local_y, 'north'|'south'|'east'|'west') tuples
        self.windows = set()  # Set of (local_x, local_y, wall_direction) tuples
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
        
        # Generate stairs, holes, and windows after maze is created
        self._generate_stairs_holes_windows(rng)
    
    def _generate_stairs_holes_windows(self, rng):
        """Generate stairs, holes in floors/ceilings, and windows in walls"""
        width = self.chunk_size
        height = self.chunk_size
        
        # Generate stairs (10-15% of path cells) - increased frequency
        path_list = list(self.paths)
        num_stairs = max(1, int(len(path_list) * rng.uniform(0.10, 0.15)))
        stairs_cells = rng.sample(path_list, min(num_stairs, len(path_list)))
        
        for x, y in stairs_cells:
            # Randomly choose up or down stairs (each direction has equal probability)
            direction = rng.choice(['up', 'down'])
            # Target level: up goes to +1 level, down goes to -1 level
            # Stairs indicate direction, actual level change happens during physics/rendering
            self.stairs[(x, y)] = {
                'direction': direction,
                'target_level': 1 if direction == 'up' else -1
            }
        
        # Generate holes in floors (5-10% of path cells, but not in stairs) - increased frequency
        path_list_no_stairs = [p for p in path_list if p not in stairs_cells]
        num_floor_holes = max(0, int(len(path_list_no_stairs) * rng.uniform(0.05, 0.10)))
        floor_hole_cells = rng.sample(path_list_no_stairs, min(num_floor_holes, len(path_list_no_stairs)))
        for x, y in floor_hole_cells:
            self.holes[(x, y)] = 'floor'
        
        # Generate holes in ceilings (4-8% of path cells, but not in stairs or floor holes) - increased frequency
        path_list_no_holes = [p for p in path_list_no_stairs if p not in floor_hole_cells]
        num_ceiling_holes = max(0, int(len(path_list_no_holes) * rng.uniform(0.04, 0.08)))
        ceiling_hole_cells = rng.sample(path_list_no_holes, min(num_ceiling_holes, len(path_list_no_holes)))
        for x, y in ceiling_hole_cells:
            self.holes[(x, y)] = 'ceiling'
        
        # Generate windows in walls (10-20% of wall cells that border paths)
        wall_list = list(self.walls)
        windows_candidates = []
        for x, y in wall_list:
            # Check if this wall is adjacent to a path (so window makes sense)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.paths:
                    # Determine which wall face should have the window
                    if dx == 1:  # Path to the east, window on east wall
                        windows_candidates.append((x, y, 'east'))
                    elif dx == -1:  # Path to the west, window on west wall
                        windows_candidates.append((x, y, 'west'))
                    elif dy == 1:  # Path to the south, window on south wall
                        windows_candidates.append((x, y, 'south'))
                    elif dy == -1:  # Path to the north, window on north wall
                        windows_candidates.append((x, y, 'north'))
                    break  # Only one window per wall cell
        
        # Randomly select windows from candidates
        num_windows = max(1, int(len(windows_candidates) * rng.uniform(0.10, 0.20)))
        selected_windows = rng.sample(windows_candidates, min(num_windows, len(windows_candidates)))
        for window_info in selected_windows:
            self.windows.add(window_info)
    
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
        
        # Initialize joysticks
        pygame.joystick.init()
        self.joysticks = []
        self.joystick_deadzone = 0.15  # Dead zone to prevent drift
        self.joystick_axis_max = 1.0
        
        # Detect and initialize joysticks
        joystick_count = pygame.joystick.get_count()
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            self.joysticks.append(joystick)
            print(f"    ✓ Joystick {i} initialized: {joystick.get_name()}")
        
        if joystick_count == 0:
            print("    ⚠ No joysticks/gamepads detected")
        
        # Single shared Game of Life instance for all walls
        self.shared_game_of_life = GameOfLife(seed=42)  # Fixed seed so all walls use same pattern
        self.game_of_life_update_counter = 0  # Update Game of Life every N frames
        self.game_of_life_reset_counter = 0  # Counter for resetting Game of Life every 30 frames
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
        
        # Load sword image
        self.sword_image = None
        try:
            import os
            sword_path = os.path.join(os.path.dirname(__file__), 'kopis.png')
            if os.path.exists(sword_path):
                self.sword_image = pygame.image.load(sword_path).convert_alpha()
                # Scale sword to reasonable size for weapon overlay
                # Target height: about 1/4 to 1/3 of screen height
                original_size = self.sword_image.get_size()
                target_height = min(200, self.height // 3)  # Max 200px or 1/3 screen height
                scale_factor = target_height / original_size[1]
                new_width = int(original_size[0] * scale_factor)
                new_height = int(original_size[1] * scale_factor)
                self.sword_image = pygame.transform.scale(self.sword_image, (new_width, new_height))
                # Rotate 180 degrees so blade is at top (handle at bottom)
                self.sword_image = pygame.transform.rotate(self.sword_image, 180.0)
                print(f"✓ Loaded sword image: {sword_path} ({new_width}x{new_height}), rotated 180°")
            else:
                print(f"⚠ Sword image not found at {sword_path}")
        except Exception as e:
            print(f"⚠ Could not load sword image: {e}")
            self.sword_image = None
    
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
                        
                        # Generate random color for each pixel based on position
                        pixel_seed = hash((x, y, 'face')) % 1000000
                        rng_r = (pixel_seed * 9301 + 49297) % 233280
                        rng_g = (pixel_seed * 7919 + 49307) % 233280
                        rng_b = (pixel_seed * 6997 + 49317) % 233280
                        blood_red = 50 + (rng_r % 206)  # 50-255
                        blood_green = 50 + (rng_g % 206)  # 50-255
                        blood_blue = 50 + (rng_b % 206)  # 50-255
                        
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
               camera_yaw: float = 0.0, camera_pitch: float = 0.0, camera_roll: float = 0.0, fpv_mode: bool = False, 
               sword_swing_state: float = 0.0, settings: Optional[Dict] = None):
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
                
                # Load settings or use defaults
                if settings:
                    FOV = float(settings.get('fov', 60))
                    MAX_DEPTH = float(settings.get('max_depth', 1000))
                    RAYCAST_SKIP = int(settings.get('raycast_skip', 2))
                else:
                    FOV = 60.0  # Field of view in degrees
                    MAX_DEPTH = 1000.0  # Maximum ray distance
                    RAYCAST_SKIP = 2
                CEILING_LEVEL = STORY_HEIGHT_UNITS  # Ceiling height (10 feet = 300 units)
                CAMERA_HEIGHT_OFFSET = PLAYER_HEIGHT_UNITS * 0.833  # Eye level (5 feet = 150 units, ~83% of 6 feet)
                
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
                
                # Draw floor and ceiling per column (to support stairs and variable heights)
                # We'll draw them per-column during raycasting to handle stairs properly
                half_height = self.height / 2
                horizon_y = half_height - math.tan(pitch_rad) * (self.height / 2)
                
                # Calculate current level based on camera Z (each level is 10 feet = 300 units)
                current_level = int(cam_z / STORY_HEIGHT_UNITS)
                level_floor_z = current_level * STORY_HEIGHT_UNITS
                level_ceiling_z = (current_level + 1) * STORY_HEIGHT_UNITS
                
                # Basic ceiling (default, will be modified per-column for stairs)
                ceiling_top = 0
                ceiling_bottom = max(0, int(horizon_y))
                pygame.draw.rect(self.screen, (26, 26, 26), (0, ceiling_top, self.width, ceiling_bottom - ceiling_top))
                
                # Basic floor (default, will be modified per-column for stairs)
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
                                            # Generate random color for each pixel based on position
                                            pixel_seed = hash((x, y, 'ceiling')) % 1000000
                                            rng_r = (pixel_seed * 9301 + 49297) % 233280
                                            rng_g = (pixel_seed * 7919 + 49307) % 233280
                                            rng_b = (pixel_seed * 6997 + 49317) % 233280
                                            blood_red = 50 + (rng_r % 206)  # 50-255
                                            blood_green = 50 + (rng_g % 206)  # 50-255
                                            blood_blue = 50 + (rng_b % 206)  # 50-255
                                            
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
                                            # Generate random color for each pixel based on position
                                            pixel_seed = hash((x, y, 'floor')) % 1000000
                                            rng_r = (pixel_seed * 9301 + 49297) % 233280
                                            rng_g = (pixel_seed * 7919 + 49307) % 233280
                                            rng_b = (pixel_seed * 6997 + 49317) % 233280
                                            blood_red = 50 + (rng_r % 206)  # 50-255
                                            blood_green = 50 + (rng_g % 206)  # 50-255
                                            blood_blue = 50 + (rng_b % 206)  # 50-255
                                            
                                            # Draw with sample rate width/height to fill gaps
                                            pygame.draw.rect(self.screen, (blood_red, blood_green, blood_blue),
                                                           (screen_x, screen_y, max(1, int(cell_w * floor_sample_rate)), 
                                                            max(1, int(cell_h * floor_sample_rate))))
                
                # Raycast for each column
                # Use settings if available, otherwise adaptive quality
                if settings:
                    raycast_skip = int(settings.get('raycast_skip', 2))
                else:
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
                        
                        # Check for window in this wall
                        chunk_x, chunk_y = self.maze._get_chunk_coords((map_x * cell_size, map_y * cell_size))
                        chunk = self.maze._get_or_create_chunk(chunk_x, chunk_y)
                        local_x = ((map_x % self.maze.chunk_size) + self.maze.chunk_size) % self.maze.chunk_size
                        local_y = ((map_y % self.maze.chunk_size) + self.maze.chunk_size) % self.maze.chunk_size
                        
                        # Determine wall direction for window check
                        # side == 0 means X-side wall, side == 1 means Y-side wall
                        # Use ray direction to determine which face of the wall we hit
                        wall_dir = None
                        if side == 0:  # X-side wall
                            wall_dir = 'west' if ray_dir_x < 0 else 'east'
                        else:  # Y-side wall
                            wall_dir = 'north' if ray_dir_y < 0 else 'south'
                        
                        has_window = (local_x, local_y, wall_dir) in chunk.windows
                        
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
                        wall_height_pixels = min(self.height, int(draw_end)) - max(0, int(draw_start))
                        
                        if has_window:
                            # Draw window: lighter section in middle of wall
                            window_start_ratio = 0.35  # Window starts at 35% up the wall
                            window_end_ratio = 0.65    # Window ends at 65% up the wall
                            window_start_screen = draw_start + (draw_end - draw_start) * window_start_ratio
                            window_end_screen = draw_start + (draw_end - draw_start) * window_end_ratio
                            
                            # Draw wall top
                            if int(draw_start) < int(window_start_screen):
                                pygame.draw.rect(self.screen, (wall_r, wall_g, wall_b),
                                               (x, max(0, int(draw_start)), wall_width, 
                                                int(window_start_screen) - max(0, int(draw_start))))
                            
                            # Draw window (lighter, bluish color to simulate sky/outside)
                            window_r = min(255, int(wall_r * 1.5))
                            window_g = min(255, int(wall_g * 1.5))
                            window_b = min(255, int(wall_b * 1.3 + 40))
                            pygame.draw.rect(self.screen, (window_r, window_g, window_b),
                                           (x, max(0, int(window_start_screen)), wall_width, 
                                            int(window_end_screen) - max(0, int(window_start_screen))))
                            
                            # Draw wall bottom
                            if int(window_end_screen) < int(draw_end):
                                pygame.draw.rect(self.screen, (wall_r, wall_g, wall_b),
                                               (x, max(0, int(window_end_screen)), wall_width, 
                                                min(self.height, int(draw_end)) - max(0, int(window_end_screen))))
                        else:
                            # Draw solid wall
                            pygame.draw.rect(self.screen, (wall_r, wall_g, wall_b),
                                           (x, max(0, int(draw_start)), wall_width, wall_height_pixels))
                        
                        # Check cell before the wall for stairs and render them
                        # Go back one step to get the cell before the wall based on ray direction
                        if side == 0:  # X-side wall
                            pre_cell_x = map_x - (1 if ray_dir_x > 0 else -1)
                            pre_cell_y = map_y
                        else:  # Y-side wall
                            pre_cell_x = map_x
                            pre_cell_y = map_y - (1 if ray_dir_y > 0 else -1)
                        
                        # Check for stairs in the cell before the wall
                        chunk_x_pre, chunk_y_pre = self.maze._get_chunk_coords((pre_cell_x * cell_size, pre_cell_y * cell_size))
                        chunk_pre = self.maze._get_or_create_chunk(chunk_x_pre, chunk_y_pre)
                        local_x_pre = ((pre_cell_x % self.maze.chunk_size) + self.maze.chunk_size) % self.maze.chunk_size
                        local_y_pre = ((pre_cell_y % self.maze.chunk_size) + self.maze.chunk_size) % self.maze.chunk_size
                        
                        stair_info_pre = chunk_pre.stairs.get((local_x_pre, local_y_pre))
                        
                        # If there are stairs, render them as visible steps
                        if stair_info_pre:
                            # Calculate stair rendering - draw visible step lines
                            # Draw darker lines to indicate stair steps
                            num_steps = 4  # Number of visible steps
                            step_height = STORY_HEIGHT_UNITS / num_steps  # Height per step
                            
                            for step in range(num_steps + 1):
                                step_z = level_floor_z + (step * step_height) if stair_info_pre['direction'] == 'up' else level_floor_z - (step * step_height)
                                # Calculate screen y for this step height
                                step_p = (cam_z - step_z) / (cam_z - level_floor_z) if (cam_z - level_floor_z) != 0 else 1.0
                                step_screen_y = horizon_y + (self.height - horizon_y) * step_p
                                
                                if int(draw_end) <= step_screen_y < self.height:
                                    # Draw step line (darker color to show depth)
                                    pygame.draw.line(self.screen, (35, 35, 35),
                                                   (x, int(step_screen_y)),
                                                   (x + raycast_skip, int(step_screen_y)), 1)
                        
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
                                                # Generate random color for each pixel based on position
                                                pixel_seed = hash((pattern_x, y, 'wall')) % 1000000
                                                rng_r = (pixel_seed * 9301 + 49297) % 233280
                                                rng_g = (pixel_seed * 7919 + 49307) % 233280
                                                rng_b = (pixel_seed * 6997 + 49317) % 233280
                                                blood_red = 50 + (rng_r % 206)  # 50-255
                                                blood_green = 50 + (rng_g % 206)  # 50-255
                                                blood_blue = 50 + (rng_b % 206)  # 50-255
                                                
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
                
                # Reset Game of Life with new random pattern every 30 frames
                self.game_of_life_reset_counter += 1
                if self.game_of_life_reset_counter >= 30:
                    self.game_of_life_reset_counter = 0
                    # Generate new random seed for fresh pattern
                    import random
                    new_seed = random.randint(0, 1000000)
                    self.shared_game_of_life = GameOfLife(seed=new_seed)
                    self.game_of_life_surface_cache = None  # Clear cache on reset
                
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
                # Use settings for FOV if available
                if settings:
                    FOV = float(settings.get('fov', 60))
                else:
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
                
                # All entities are green squares
                color = self.GREEN  # Green color for all entities
                entity_height = getattr(entity, 'height', PLAYER_HEIGHT_UNITS)  # Use entity height (6 feet = 180 units)
                
                # Calculate sprite size based on distance and entity height
                # Size should scale with distance, using entity height as base
                sprite_size = max(5, min(100, entity_height * (self.height / sprite['depth']) / 2.0))
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
                
                # Draw sprite as green cube (3D box)
                # Calculate cube dimensions (width = depth = height for a cube)
                cube_width = entity_height / 2.0  # Cube is half the entity height
                cube_size_screen = max(5, min(100, cube_width * (self.height / sprite['depth']) / 2.0))
                
                # Draw cube as isometric-style box (visible faces)
                # Calculate cube vertices and draw faces
                center_x = sprite['screen_x']
                center_y = sprite['screen_y']
                
                # Draw top face (darker)
                top_brightness = brightness * 0.7
                top_color = tuple(int(c * top_brightness) for c in color[:3])
                # Top face vertices (shifted up and back)
                top_offset = cube_size_screen * 0.3
                top_points = [
                    (center_x - cube_size_screen/2 - top_offset, center_y - cube_size_screen/2 - top_offset),
                    (center_x + cube_size_screen/2 - top_offset, center_y - cube_size_screen/2 - top_offset),
                    (center_x + cube_size_screen/2 + top_offset, center_y - cube_size_screen/2 + top_offset),
                    (center_x - cube_size_screen/2 + top_offset, center_y - cube_size_screen/2 + top_offset)
                ]
                pygame.draw.polygon(self.screen, top_color, top_points)
                
                # Draw front face (brighter)
                front_points = [
                    (center_x - cube_size_screen/2, center_y - cube_size_screen/2),
                    (center_x + cube_size_screen/2, center_y - cube_size_screen/2),
                    (center_x + cube_size_screen/2, center_y + cube_size_screen/2),
                    (center_x - cube_size_screen/2, center_y + cube_size_screen/2)
                ]
                pygame.draw.polygon(self.screen, color, front_points)
                pygame.draw.polygon(self.screen, (255, 255, 255), front_points, 2)
                
                # Draw right face (medium brightness)
                right_brightness = brightness * 0.85
                right_color = tuple(int(c * right_brightness) for c in color[:3])
                right_points = [
                    (center_x + cube_size_screen/2, center_y - cube_size_screen/2),
                    (center_x + cube_size_screen/2 + top_offset*2, center_y - cube_size_screen/2 + top_offset*2),
                    (center_x + cube_size_screen/2 + top_offset*2, center_y + cube_size_screen/2 + top_offset*2),
                    (center_x + cube_size_screen/2, center_y + cube_size_screen/2)
                ]
                pygame.draw.polygon(self.screen, right_color, right_points)
                
                # Draw edges
                pygame.draw.lines(self.screen, (255, 255, 255), True, top_points, 1)
                pygame.draw.lines(self.screen, (255, 255, 255), True, right_points, 1)
        else:
            # 2D fallback rendering
            sorted_entities = sorted(entities, key=lambda e: e.position[2] if len(e.position) == 3 else 0.0)
            
            for entity in sorted_entities:
                screen_pos = self.world_to_screen(entity.position, camera_pos, camera_yaw, camera_pitch, camera_roll, fpv_mode)
                
                if 0 <= screen_pos[0] <= self.width and 0 <= screen_pos[1] <= self.height:
                    # All entities are green squares
                    color = self.GREEN
                    entity_height = getattr(entity, 'height', PLAYER_HEIGHT_UNITS)  # Use entity height (6 feet = 180 units)
                    base_size = entity_height / 6.0  # Scale based on height
                    
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
                    size = int(base_size * perspective_scale)
                    half_size = size // 2
                    
                    # Draw green cube (3D box)
                    center_x = screen_pos[0]
                    center_y = screen_pos[1]
                    cube_width = entity_height / 6.0  # Cube size
                    cube_size = int(cube_width * perspective_scale)
                    cube_offset = int(cube_size * 0.3)  # Offset for isometric effect
                    
                    # Draw top face (darker)
                    top_brightness = 0.7
                    top_color = tuple(int(c * top_brightness) for c in color[:3])
                    top_points = [
                        (center_x - cube_size//2 - cube_offset, center_y - cube_size//2 - cube_offset),
                        (center_x + cube_size//2 - cube_offset, center_y - cube_size//2 - cube_offset),
                        (center_x + cube_size//2 + cube_offset, center_y - cube_size//2 + cube_offset),
                        (center_x - cube_size//2 + cube_offset, center_y - cube_size//2 + cube_offset)
                    ]
                    pygame.draw.polygon(self.screen, top_color, top_points)
                    
                    # Draw front face (brighter)
                    front_points = [
                        (center_x - cube_size//2, center_y - cube_size//2),
                        (center_x + cube_size//2, center_y - cube_size//2),
                        (center_x + cube_size//2, center_y + cube_size//2),
                        (center_x - cube_size//2, center_y + cube_size//2)
                    ]
                    pygame.draw.polygon(self.screen, color, front_points)
                    pygame.draw.polygon(self.screen, self.WHITE, front_points, 2)
                    
                    # Draw right face (medium brightness)
                    right_brightness = 0.85
                    right_color = tuple(int(c * right_brightness) for c in color[:3])
                    right_points = [
                        (center_x + cube_size//2, center_y - cube_size//2),
                        (center_x + cube_size//2 + cube_offset*2, center_y - cube_size//2 + cube_offset*2),
                        (center_x + cube_size//2 + cube_offset*2, center_y + cube_size//2 + cube_offset*2),
                        (center_x + cube_size//2, center_y + cube_size//2)
                    ]
                    pygame.draw.polygon(self.screen, right_color, right_points)
                    
                    # Draw edges
                    pygame.draw.lines(self.screen, self.WHITE, True, top_points, 1)
                    pygame.draw.lines(self.screen, self.WHITE, True, right_points, 1)
        
        # Draw UI overlay
        self._draw_ui(camera_pos, frame_count, fps, len(entities), camera_yaw, camera_pitch, fpv_mode)
        
        # Draw sword overlay in FPV mode
        if fpv_mode and self.sword_image:
            self._render_sword_overlay(sword_swing_state)
        
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
    
    def _render_sword_overlay(self, swing_state: float = 0.0):
        """Render sword as weapon overlay in first-person view (like Doom weapon)"""
        if not self.sword_image:
            return
        
        import math
        
        # Calculate swing animation (0.0 = idle/ready, 1.0 = swing complete)
        # Swing from right to left with rotation
        if swing_state <= 0.0:
            # Idle position: slightly to the right, angled slightly
            x_offset = self.width * 0.65  # Right side of screen
            y_offset = self.height * 0.7  # Lower part of screen
            rotation = -15.0  # Slight rotation to the left (counter-clockwise)
            scale = 1.0
        else:
            # Animated swing: arc from right to left
            # Swing progress: 0.0 to 1.0
            progress = min(1.0, swing_state)
            
            # Arc motion: horizontal sweep
            # X moves from right (0.65) to center (0.5) and back
            if progress < 0.5:
                # First half: move left
                arc_progress = progress * 2.0  # 0.0 to 1.0
                x_offset = self.width * (0.65 - arc_progress * 0.2)  # Move from 0.65 to 0.45
            else:
                # Second half: return right
                arc_progress = (progress - 0.5) * 2.0  # 0.0 to 1.0
                x_offset = self.width * (0.45 + arc_progress * 0.2)  # Return from 0.45 to 0.65
            
            # Vertical bounce: slight upward motion during swing
            bounce = abs(math.sin(progress * math.pi)) * 30.0  # Up to 30 pixels bounce
            y_offset = self.height * 0.7 - bounce
            
            # Rotation: sword rotates during swing (chopping motion)
            rotation = -15.0 + (progress * 90.0) - (abs(progress - 0.5) * 60.0)  # Rotates through swing
            
            # Scale: slight scale change for impact effect
            scale = 1.0 + abs(math.sin(progress * math.pi * 2)) * 0.1  # 10% scale change
        
        # Rotate and scale the sword image
        if rotation != 0.0 or scale != 1.0:
            # Scale first
            if scale != 1.0:
                original_size = self.sword_image.get_size()
                new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                scaled_image = pygame.transform.scale(self.sword_image, new_size)
            else:
                scaled_image = self.sword_image
            
            # Rotate (blade at top after 180° rotation, so rotation center should be near top)
            if rotation != 0.0:
                rotated_image = pygame.transform.rotate(scaled_image, rotation)
            else:
                rotated_image = scaled_image
        else:
            rotated_image = self.sword_image
        
        # Get image rect and position (blade at top after 180° rotation)
        img_rect = rotated_image.get_rect()
        # Position so blade is at the specified offset (top of image is blade after rotation)
        screen_x = int(x_offset - img_rect.width / 2)
        screen_y = int(y_offset)  # Top of image (blade) at this y position
        
        # Draw sword with slight transparency during swing for effect
        if swing_state > 0.0 and swing_state < 1.0:
            # Slight fade during swing
            alpha = int(255 * (1.0 - abs(swing_state - 0.5) * 0.3))  # Up to 15% transparency at midpoint
            rotated_image.set_alpha(alpha)
        
        # Blit the sword
        self.screen.blit(rotated_image, (screen_x, screen_y))
        
        # Reset alpha if it was changed
        if swing_state > 0.0:
            rotated_image.set_alpha(255)
    
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
                if event.button == 1:  # Left mouse button
                    mouse_clicked = True
                # Clicking in FPV mode can also capture mouse if not already captured
                if fpv_mode and not self.mouse_captured:
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
                    self.mouse_captured = True
                    pygame.mouse.set_pos(self.width // 2, self.height // 2)
            elif event.type == pygame.FINGERDOWN:  # Touch event for iOS/mobile
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
        
        # Joystick input (gamepad support)
        joystick_move_forward = 0.0
        joystick_move_right = 0.0
        joystick_camera_dx = 0.0
        joystick_camera_dy = 0.0
        joystick_jump = False
        joystick_sword = False
        joystick_roll_left = False
        joystick_roll_right = False
        
        # Use first joystick if available
        if len(self.joysticks) > 0:
            joy = self.joysticks[0]
            num_axes = joy.get_numaxes()
            num_buttons = joy.get_numbuttons()
            
            # Left stick: movement (axis 0 = horizontal, axis 1 = vertical)
            # Right stick: camera (axis 2/3 or axis 4/5 depending on controller)
            if num_axes >= 2:
                # Left stick horizontal (strafe left/right)
                axis_0 = joy.get_axis(0)
                if abs(axis_0) > self.joystick_deadzone:
                    joystick_move_right = axis_0
                
                # Left stick vertical (forward/backward) - inverted
                axis_1 = joy.get_axis(1)
                if abs(axis_1) > self.joystick_deadzone:
                    joystick_move_forward = -axis_1  # Invert for forward when pushed up
            
            # Right stick: camera rotation
            # Xbox controller: axes 3 (horizontal) and 4 (vertical)
            # PS4 controller: axes 2 (horizontal) and 3 (vertical)
            if num_axes >= 4:
                # Try Xbox layout first (axes 3 and 4)
                axis_3 = joy.get_axis(3) if num_axes > 3 else 0.0
                axis_4 = joy.get_axis(4) if num_axes > 4 else 0.0
                
                # Also try PS4 layout (axes 2 and 3)
                axis_2 = joy.get_axis(2) if num_axes > 2 else 0.0
                
                # Use the axis with more movement (detect which layout)
                if abs(axis_3) > abs(axis_2):
                    # Xbox layout
                    if abs(axis_3) > self.joystick_deadzone:
                        joystick_camera_dx = axis_3 * 2.0  # Scale for sensitivity
                    if abs(axis_4) > self.joystick_deadzone:
                        joystick_camera_dy = -axis_4 * 2.0  # Invert and scale
                else:
                    # PS4 layout
                    if abs(axis_2) > self.joystick_deadzone:
                        joystick_camera_dx = axis_2 * 2.0
                    if abs(axis_3) > self.joystick_deadzone:
                        joystick_camera_dy = -axis_3 * 2.0
            
            # Buttons: Xbox mapping
            # Button 0 = A (jump)
            # Button 1 = B
            # Button 2 = X
            # Button 3 = Y
            # Button 4 = Left Bumper (L1)
            # Button 5 = Right Bumper (R1)
            # Button 6 = Back/Select
            # Button 7 = Start
            # Button 8 = Left Stick Press
            # Button 9 = Right Stick Press
            # Triggers: axes 4 and 5 (or 2 and 3 on some controllers)
            
            if num_buttons >= 1:
                # Jump (A button, typically button 0)
                joystick_jump = joy.get_button(0)
            
            # Sword swing: Right trigger (axis 5) or X button (button 2)
            if num_axes >= 6:
                trigger = joy.get_axis(5)
                if trigger > 0.5:  # Trigger pressed more than halfway
                    joystick_sword = True
            if num_buttons >= 3:
                if joy.get_button(2):  # X button
                    joystick_sword = True
            
            # Roll: Shoulder buttons (L1 = button 4, R1 = button 5)
            if num_buttons >= 5:
                joystick_roll_left = joy.get_button(4)  # Left bumper
                joystick_roll_right = joy.get_button(5)  # Right bumper
        
        # Combine joystick and keyboard input (joystick takes precedence for movement)
        if abs(joystick_move_forward) > 0.01 or abs(joystick_move_right) > 0.01:
            # Joystick movement overrides keyboard
            keys_pressed['w'] = joystick_move_forward > 0.1
            keys_pressed['s'] = joystick_move_forward < -0.1
            keys_pressed['a'] = joystick_move_right < -0.1
            keys_pressed['d'] = joystick_move_right > 0.1
        
        if joystick_jump:
            keys_pressed['space'] = True
        
        if joystick_sword:
            mouse_clicked = True
        
        if joystick_roll_left:
            keys_pressed['q'] = True
        if joystick_roll_right:
            keys_pressed['e'] = True
        
        # Combine mouse and joystick camera input
        final_mouse_dx = mouse_delta[0] + joystick_camera_dx * 0.15  # Scale joystick sensitivity
        final_mouse_dy = mouse_delta[1] + joystick_camera_dy * 0.15
        
        return {
            'quit': quit_requested,
            'keys': keys_pressed,
            'mouse': {
                'x': mouse_pos[0], 
                'y': mouse_pos[1], 
                'dx': final_mouse_dx,  # Combined mouse + joystick delta X for camera yaw
                'dy': final_mouse_dy,  # Combined mouse + joystick delta Y for camera pitch
                'clicked': mouse_clicked,
                'left_button': mouse_clicked,  # Alias for compatibility
                'touch': mouse_clicked,  # For iOS/mobile touch events
                'tap': mouse_clicked  # Alternative name for tap
            },
            'joystick': {
                'move_forward': joystick_move_forward,
                'move_right': joystick_move_right,
                'camera_dx': joystick_camera_dx,
                'camera_dy': joystick_camera_dy,
                'jump': joystick_jump,
                'sword': joystick_sword
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
    """
    Interactive game loop with pygame visualization, maze, and sound effects
    
    This is the main entry point for the Kopis Engine game.
    It initializes all systems and runs the game loop.
    """
    # Check dependencies
    if not PYGAME_AVAILABLE:
        print("\n" + "=" * 60)
        print("ERROR: pygame is required but not installed")
        print("=" * 60)
        print("Kopis Engine needs pygame to run.")
        print("Please install it with one of these commands:")
        print("  pip install pygame")
        print("  pip3 install pygame")
        print("  python -m pip install pygame")
        print("\nAfter installing, run the game again.")
        print("=" * 60 + "\n")
        sys.exit(1)
    
    # Initialize systems with progress feedback
    print("\n" + "=" * 60)
    print("KOPIS ENGINE - Initializing...")
    print("=" * 60)
    
    print("\n[1/4] Creating infinite maze system...")
    try:
        maze = Maze(chunk_size=20, cell_size=50.0, load_radius=3)
        print("    ✓ Infinite maze system initialized")
        print("    ✓ Chunks will be generated on-demand as you explore")
    except Exception as e:
        print(f"    ✗ Error initializing maze: {e}")
        print("    The game may not work correctly.")
        sys.exit(1)
    
    print("\n[2/4] Initializing sound system...")
    sound_manager = SoundManager()
    if sound_manager.enabled:
        print("    ✓ Sound system initialized")
        print("    ✓ Sound effects will play during gameplay")
    else:
        print("    ⚠ Sound system unavailable (game will run without sound)")
    
    print("\n[3/4] Initializing game engine...")
    try:
        engine = KopisEngine(maze=maze, sound_manager=sound_manager)
        engine.visualize_circuit()
        print("    ✓ Game engine initialized")
        print("    ✓ All systems ready")
    except Exception as e:
        print(f"    ✗ Error initializing engine: {e}")
        sys.exit(1)
    
    print("\n[4/4] Setting up graphics renderer...")
    try:
        renderer = PygameRenderer(width=800, height=600, maze=maze)
        print("    ✓ Graphics renderer initialized")
        print("    ✓ Window created (800x600)")
    except Exception as e:
        print(f"    ✗ Error initializing renderer: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("INITIALIZATION COMPLETE!")
    print("=" * 60)
    
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
    
    def show_new_game_screen():
        """Display new game screen after epilepsy warning"""
        new_game_screen = renderer.screen
        clock = pygame.time.Clock()
        
        # Colors
        GREEN = (68, 255, 68)
        DARK_GREEN = (0, 204, 0)
        WHITE = (255, 255, 255)
        DARK_BG = (26, 46, 26)
        
        # Create fonts
        try:
            title_font = pygame.font.Font(None, 72)
            text_font = pygame.font.Font(None, 36)
            button_font = pygame.font.Font(None, 48)
        except:
            title_font = pygame.font.SysFont('arial', 72, bold=True)
            text_font = pygame.font.SysFont('arial', 36)
            button_font = pygame.font.SysFont('arial', 48, bold=True)
        
        acknowledged = False
        settings = None
        
        while not acknowledged:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False, None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        acknowledged = True
                    elif event.key == pygame.K_o or event.key == pygame.K_O:
                        # Show options screen
                        options_result = show_options_screen()
                        if options_result[0]:  # User didn't quit
                            if options_result[1]:  # Settings were saved
                                settings = options_result[1]
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    # Check if clicking on options button
                    options_button_rect = pygame.Rect(renderer.width // 2 - 100, renderer.height - 120, 200, 40)
                    if options_button_rect.collidepoint(mouse_pos):
                        options_result = show_options_screen()
                        if options_result[0]:  # User didn't quit
                            if options_result[1]:  # Settings were saved
                                settings = options_result[1]
                    else:
                        acknowledged = True
            
            # Clear screen with dark background
            new_game_screen.fill(DARK_BG)
            
            # Draw border
            pygame.draw.rect(new_game_screen, GREEN, (50, 50, renderer.width - 100, renderer.height - 100), 5)
            
            # Title
            title_text = title_font.render("NEW GAME", True, GREEN)
            title_rect = title_text.get_rect(center=(renderer.width // 2, 250))
            new_game_screen.blit(title_text, title_rect)
            
            # Message
            message_lines = [
                "Welcome to Kopis Engine!",
                "",
                "Navigate the infinite maze.",
                "Avoid NPCs or you'll take damage!",
                "",
                "Use WASD to move, Space to jump,",
                "and Mouse to look around."
            ]
            
            y_offset = 350
            for line in message_lines:
                if line:  # Skip empty lines
                    text = text_font.render(line, True, WHITE)
                    text_rect = text.get_rect(center=(renderer.width // 2, y_offset))
                    new_game_screen.blit(text, text_rect)
                y_offset += 45
            
            # Buttons
            start_button_text = button_font.render("Press ENTER, SPACE, or CLICK to Start", True, WHITE)
            start_button_rect = start_button_text.get_rect(center=(renderer.width // 2, renderer.height - 200))
            start_button_bg_rect = pygame.Rect(start_button_rect.x - 20, start_button_rect.y - 10, 
                                              start_button_rect.width + 40, start_button_rect.height + 20)
            pygame.draw.rect(new_game_screen, DARK_GREEN, start_button_bg_rect)
            pygame.draw.rect(new_game_screen, GREEN, start_button_bg_rect, 3)
            new_game_screen.blit(start_button_text, start_button_rect)
            
            # Options button
            options_button_text = pygame.font.Font(None, 36).render("OPTIONS (O)", True, WHITE)
            options_button_rect = options_button_text.get_rect(center=(renderer.width // 2, renderer.height - 120))
            options_button_bg_rect = pygame.Rect(options_button_rect.x - 15, options_button_rect.y - 8, 
                                                options_button_rect.width + 30, options_button_rect.height + 16)
            pygame.draw.rect(new_game_screen, (80, 80, 120), options_button_bg_rect)
            pygame.draw.rect(new_game_screen, (120, 120, 180), options_button_bg_rect, 3)
            new_game_screen.blit(options_button_text, options_button_rect)
            
            pygame.display.flip()
            clock.tick(60)
        
        return True, settings
    
    def show_options_screen():
        """Display options screen with settings"""
        options_screen = renderer.screen
        clock = pygame.time.Clock()
        
        # Colors
        BLUE = (91, 155, 213)
        DARK_BLUE = (15, 52, 96)
        CYAN = (0, 212, 255)
        WHITE = (255, 255, 255)
        DARK_BG = (26, 26, 46)
        GREEN = (68, 255, 68)
        DARK_GREEN = (0, 204, 0)
        
        # Create fonts
        try:
            title_font = pygame.font.Font(None, 72)
            text_font = pygame.font.Font(None, 36)
            small_font = pygame.font.Font(None, 24)
            button_font = pygame.font.Font(None, 48)
        except:
            title_font = pygame.font.SysFont('arial', 72, bold=True)
            text_font = pygame.font.SysFont('arial', 36)
            small_font = pygame.font.SysFont('arial', 24)
            button_font = pygame.font.SysFont('arial', 48, bold=True)
        
        # Default settings
        settings = {
            'npc_count': 25,
            'fov': 60,
            'max_depth': 1000,
            'raycast_skip': 2
        }
        
        # Try to load from file
        settings_file = 'kopis_settings.json'
        try:
            import json
            import os
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    settings.update(loaded_settings)
        except Exception as e:
            print(f"Could not load settings: {e}")
        
        # Slider positions and ranges
        slider_y_start = 200
        slider_spacing = 80
        slider_width = 400
        slider_height = 30
        slider_x = renderer.width // 2 - slider_width // 2
        
        selected_setting = None
        settings_keys = ['npc_count', 'fov', 'max_depth', 'raycast_skip']
        setting_labels = {
            'npc_count': ('NPC Count', 5, 50, 5),
            'fov': ('Field of View (FOV)', 45, 110, 5),
            'max_depth': ('Max Render Distance', 500, 2000, 100),
            'raycast_skip': ('Raycast Skip (Performance)', 1, 4, 1)
        }
        
        def get_slider_value(key):
            """Get slider value (0-1) for a setting"""
            min_val, max_val, step = setting_labels[key][1], setting_labels[key][2], setting_labels[key][3]
            current = settings[key]
            return (current - min_val) / (max_val - min_val)
        
        def set_slider_value(key, value_01):
            """Set setting from slider value (0-1)"""
            min_val, max_val, step = setting_labels[key][1], setting_labels[key][2], setting_labels[key][3]
            raw_val = min_val + value_01 * (max_val - min_val)
            # Round to nearest step
            settings[key] = int(round(raw_val / step) * step)
        
        running = True
        saved = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False, None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        # Save settings
                        try:
                            import json
                            with open(settings_file, 'w') as f:
                                json.dump(settings, f, indent=2)
                            saved = True
                        except Exception as e:
                            print(f"Could not save settings: {e}")
                        running = False
                        break
                    elif event.key == pygame.K_ESCAPE:
                        # Cancel without saving
                        running = False
                        break
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    mx, my = mouse_pos
                    
                    # Check if clicking on a slider
                    for i, key in enumerate(settings_keys):
                        slider_y = slider_y_start + i * slider_spacing
                        if (slider_x <= mx <= slider_x + slider_width and
                            slider_y - 10 <= my <= slider_y + slider_height + 10):
                            selected_setting = key
                            # Update value based on mouse X
                            rel_x = (mx - slider_x) / slider_width
                            rel_x = max(0.0, min(1.0, rel_x))
                            set_slider_value(key, rel_x)
                            break
                    
                    # Check buttons
                    save_button_rect = pygame.Rect(renderer.width // 2 - 200, renderer.height - 200, 180, 50)
                    cancel_button_rect = pygame.Rect(renderer.width // 2 + 20, renderer.height - 200, 180, 50)
                    reset_button_rect = pygame.Rect(renderer.width // 2 - 90, renderer.height - 130, 180, 50)
                    
                    if save_button_rect.collidepoint(mouse_pos):
                        try:
                            import json
                            with open(settings_file, 'w') as f:
                                json.dump(settings, f, indent=2)
                            saved = True
                        except Exception as e:
                            print(f"Could not save settings: {e}")
                        running = False
                        break
                    elif cancel_button_rect.collidepoint(mouse_pos):
                        running = False
                        break
                    elif reset_button_rect.collidepoint(mouse_pos):
                        settings = {
                            'npc_count': 25,
                            'fov': 60,
                            'max_depth': 1000,
                            'raycast_skip': 2
                        }
                
                if event.type == pygame.MOUSEMOTION and selected_setting:
                    # Update slider while dragging
                    mx, my = event.pos
                    rel_x = (mx - slider_x) / slider_width
                    rel_x = max(0.0, min(1.0, rel_x))
                    set_slider_value(selected_setting, rel_x)
                
                if event.type == pygame.MOUSEBUTTONUP:
                    selected_setting = None
            
            # Clear screen
            options_screen.fill(DARK_BG)
            
            # Draw border
            pygame.draw.rect(options_screen, BLUE, (50, 50, renderer.width - 100, renderer.height - 100), 5)
            
            # Title
            title_text = title_font.render("OPTIONS", True, BLUE)
            title_rect = title_text.get_rect(center=(renderer.width // 2, 120))
            options_screen.blit(title_text, title_rect)
            
            # Draw sliders and labels
            for i, key in enumerate(settings_keys):
                slider_y = slider_y_start + i * slider_spacing
                label, min_val, max_val, step = setting_labels[key]
                
                # Label
                label_text = text_font.render(f"{label}: {settings[key]}", True, WHITE)
                label_rect = label_text.get_rect(left=slider_x, bottom=slider_y - 15)
                options_screen.blit(label_text, label_rect)
                
                # Slider track
                pygame.draw.rect(options_screen, DARK_BLUE, (slider_x, slider_y, slider_width, slider_height))
                pygame.draw.rect(options_screen, BLUE, (slider_x, slider_y, slider_width, slider_height), 2)
                
                # Slider thumb
                slider_value = get_slider_value(key)
                thumb_x = slider_x + int(slider_value * slider_width)
                thumb_rect = pygame.Rect(thumb_x - 10, slider_y - 5, 20, slider_height + 10)
                pygame.draw.rect(options_screen, CYAN, thumb_rect)
                pygame.draw.rect(options_screen, BLUE, thumb_rect, 2)
            
            # Buttons
            save_button_rect = pygame.Rect(renderer.width // 2 - 200, renderer.height - 200, 180, 50)
            cancel_button_rect = pygame.Rect(renderer.width // 2 + 20, renderer.height - 200, 180, 50)
            reset_button_rect = pygame.Rect(renderer.width // 2 - 90, renderer.height - 130, 180, 50)
            
            pygame.draw.rect(options_screen, DARK_GREEN, save_button_rect)
            pygame.draw.rect(options_screen, GREEN, save_button_rect, 3)
            save_text = button_font.render("SAVE", True, WHITE)
            save_text_rect = save_text.get_rect(center=save_button_rect.center)
            options_screen.blit(save_text, save_text_rect)
            
            pygame.draw.rect(options_screen, (100, 100, 100), cancel_button_rect)
            pygame.draw.rect(options_screen, (150, 150, 150), cancel_button_rect, 3)
            cancel_text = button_font.render("CANCEL", True, WHITE)
            cancel_text_rect = cancel_text.get_rect(center=cancel_button_rect.center)
            options_screen.blit(cancel_text, cancel_text_rect)
            
            pygame.draw.rect(options_screen, (80, 80, 80), reset_button_rect)
            pygame.draw.rect(options_screen, (120, 120, 120), reset_button_rect, 3)
            reset_text = small_font.render("RESET DEFAULTS", True, WHITE)
            reset_text_rect = reset_text.get_rect(center=reset_button_rect.center)
            options_screen.blit(reset_text, reset_text_rect)
            
            # Instructions
            instr_text = small_font.render("Click and drag sliders to adjust. Press ENTER/SPACE to save, ESC to cancel", True, (150, 150, 150))
            instr_rect = instr_text.get_rect(center=(renderer.width // 2, renderer.height - 50))
            options_screen.blit(instr_text, instr_rect)
            
            pygame.display.flip()
            clock.tick(60)
        
        if saved:
            return True, settings
        else:
            return True, None  # Return None if cancelled
    
    def show_game_over_screen():
        """Display game over screen with new game option"""
        game_over_screen = renderer.screen
        clock = pygame.time.Clock()
        
        # Colors
        RED = (255, 68, 68)
        DARK_RED = (204, 0, 0)
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        DARK_BG = (46, 26, 26)
        GREEN = (68, 255, 68)
        DARK_GREEN = (0, 204, 0)
        
        # Create fonts
        try:
            title_font = pygame.font.Font(None, 72)
            text_font = pygame.font.Font(None, 36)
            button_font = pygame.font.Font(None, 48)
        except:
            title_font = pygame.font.SysFont('arial', 72, bold=True)
            text_font = pygame.font.SysFont('arial', 36)
            button_font = pygame.font.SysFont('arial', 48, bold=True)
        
        choice = None  # None = waiting, True = new game, False = quit
        
        while choice is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        choice = True  # New game
                    if event.key == pygame.K_ESCAPE:
                        choice = False  # Quit
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Check which button was clicked
                    mouse_pos = pygame.mouse.get_pos()
                    new_game_rect = pygame.Rect(renderer.width // 2 - 200, renderer.height - 200, 400, 60)
                    quit_rect = pygame.Rect(renderer.width // 2 - 200, renderer.height - 120, 400, 60)
                    if new_game_rect.collidepoint(mouse_pos):
                        choice = True
                    elif quit_rect.collidepoint(mouse_pos):
                        choice = False
            
            # Clear screen with dark background
            game_over_screen.fill(DARK_BG)
            
            # Draw border
            pygame.draw.rect(game_over_screen, RED, (50, 50, renderer.width - 100, renderer.height - 100), 5)
            
            # Title
            title_text = title_font.render("GAME OVER", True, RED)
            title_rect = title_text.get_rect(center=(renderer.width // 2, 200))
            game_over_screen.blit(title_text, title_rect)
            
            # Message
            message_lines = [
                "Your health has reached zero!",
                "",
                "The maze has claimed another victim."
            ]
            
            y_offset = 300
            for line in message_lines:
                if line:  # Skip empty lines
                    text = text_font.render(line, True, WHITE)
                    text_rect = text.get_rect(center=(renderer.width // 2, y_offset))
                    game_over_screen.blit(text, text_rect)
                y_offset += 45
            
            # New Game button
            new_game_text = button_font.render("NEW GAME (ENTER/SPACE)", True, WHITE)
            new_game_rect = pygame.Rect(renderer.width // 2 - 200, renderer.height - 200, 400, 60)
            pygame.draw.rect(game_over_screen, DARK_GREEN, new_game_rect)
            pygame.draw.rect(game_over_screen, GREEN, new_game_rect, 3)
            new_game_text_rect = new_game_text.get_rect(center=new_game_rect.center)
            game_over_screen.blit(new_game_text, new_game_text_rect)
            
            # Quit button
            quit_text = button_font.render("QUIT (ESC)", True, WHITE)
            quit_rect = pygame.Rect(renderer.width // 2 - 200, renderer.height - 120, 400, 60)
            pygame.draw.rect(game_over_screen, DARK_RED, quit_rect)
            pygame.draw.rect(game_over_screen, RED, quit_rect, 3)
            quit_text_rect = quit_text.get_rect(center=quit_rect.center)
            game_over_screen.blit(quit_text, quit_text_rect)
            
            pygame.display.flip()
            clock.tick(60)
        
        return choice
    
    # Show warning and check if user wants to continue
    print("\nShowing epilepsy warning screen...")
    if not show_epilepsy_warning():
        print("\nGame cancelled by user during epilepsy warning.")
        return
    
    # Show new game screen and get settings
    print("Showing new game screen...")
    new_game_result = show_new_game_screen()
    if not new_game_result[0]:
        print("\nGame cancelled by user during new game screen.")
        return
    
    # Load settings (from options screen or defaults)
    settings = new_game_result[1]
    if settings is None:
        # Load from file or use defaults
        settings_file = 'kopis_settings.json'
        try:
            import json
            import os
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
            else:
                settings = {
                    'npc_count': 25,
                    'fov': 60,
                    'max_depth': 1000,
                    'raycast_skip': 2
                }
        except Exception as e:
            print(f"Could not load settings, using defaults: {e}")
            settings = {
                'npc_count': 25,
                'fov': 60,
                'max_depth': 1000,
                'raycast_skip': 2
            }
    
    print(f"    ✓ Loaded settings: NPCs={settings.get('npc_count', 25)}, FOV={settings.get('fov', 60)}, MaxDepth={settings.get('max_depth', 1000)}")
    
    # Find a valid starting position in the maze (on a path) - RANDOM
    import random
    import math
    
    print("\nFinding safe starting position for player...")
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
        print(f"    ✓ Found {len(valid_positions)} valid starting positions, selected random position")
    else:
        # Fallback: try more aggressive search
        print("    ⚠ No valid positions found in initial search, trying fallback...")
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
            print(f"    ✓ Found {len(fallback_positions)} fallback positions, selected random position")
        else:
            # Final safety: try random positions
            print("    ⚠ Using random position search as last resort")
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
        print("    ⚠ Warning: Starting position may be in wall, attempting to find safe position...")
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
                    print(f"    ✓ Found safe starting position at ({test_x:.1f}, {test_y:.1f})")
                    break
    
    # Create player entity at valid maze position
    print("\nCreating player entity...")
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
        },
        height=PLAYER_HEIGHT_UNITS  # 6 feet = 180 units
    )
    engine.add_entity(player)
    print(f"    ✓ Player created at position ({start_world[0]:.1f}, {start_world[1]:.1f}, 0.0)")
    print(f"    ✓ Player health: {player.health} HP")
    
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
        print(f"    ✓ Set initial camera to look at wall (yaw: {camera_yaw:.1f}°)")
    else:
        # Fallback: look in a random direction, rotated by 180 degrees
        engine.camera_yaw = (random.uniform(0, 360) + 180.0) % 360.0
        print(f"    ⚠ No nearby wall found, using random camera angle")
    
    # Create many NPCs at different valid positions within 10 seconds distance at 60fps
    # Player speed = 200.0 units/second, so 10 seconds = 2000 units max distance
    # Spawn NPCs at various distances up to 2000 units
    print("\nSpawning NPCs...")
    MAX_NPCS = settings.get('npc_count', 25)  # Load from settings
    MAX_SPAWN_DISTANCE = 2000.0  # 10 seconds * 200 units/second = 2000 units
    MIN_SPAWN_DISTANCE = 200.0  # Minimum distance to avoid crowding
    npc_count = 0
    
    for i in range(MAX_NPCS):
        # Use random angle and distance for varied spawn positions
        angle = (i / MAX_NPCS) * 2 * math.pi + random.uniform(0, 0.5)  # Add some randomness
        # Distance ranges from MIN_SPAWN_DISTANCE to MAX_SPAWN_DISTANCE (within 10 second travel time)
        dist = MIN_SPAWN_DISTANCE + random.uniform(0, MAX_SPAWN_DISTANCE - MIN_SPAWN_DISTANCE)
        
        # Calculate spawn position in world coordinates
        npc_world_target = (start_world[0] + math.cos(angle) * dist, 
                           start_world[1] + math.sin(angle) * dist)
        
        # Find nearest path cell for NPC at this position
        npc_cell = maze.find_nearest_path_cell((npc_world_target[0], npc_world_target[1]), search_radius=10)
        if npc_cell is None:
            continue  # Skip this NPC if no valid path cell found
        
        # Ensure chunk is loaded for the NPC position
        npc_world_temp = maze.cell_to_world(npc_cell)
        maze._ensure_chunks_loaded(npc_world_temp)
        
        # Get final world position for NPC
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
            },
            height=PLAYER_HEIGHT_UNITS  # 6 feet = 180 units
        )
        engine.add_entity(npc)
        npc_count += 1
    
    print(f"    ✓ Spawned {npc_count} NPCs within 10 second travel distance")
    print("\n" + "=" * 60)
    print("GAME READY - Starting...")
    print("=" * 60)
    print("\nCONTROLS:")
    print("  Movement:")
    print("    W/A/S/D        - Move forward/left/backward/right (strafe)")
    print("    SPACE          - Jump")
    print("    Mouse          - Look around (rotate camera)")
    print("  Combat:")
    print("    Left Click     - Swing sword (deal 20 damage)")
    print("    Touch (Mobile) - Swing sword")
    print("  Camera:")
    print("    Q/E            - Roll camera (tilt left/right)")
    print("    ESC            - Release mouse cursor")
    print("  Display:")
    print("    F11            - Toggle fullscreen mode")
    print("\nGAMEPLAY TIPS:")
    print("  • NPCs move toward you - use your sword to fight them")
    print("  • Avoid NPC contact - each touch deals 5 damage to you")
    print("  • Sword hits deal 20 damage to NPCs")
    print("  • Explore the infinite maze - find stairs to change levels")
    print("  • Each story is 10 feet tall, you're 6 feet tall")
    print("  • Watch your health in the top-left corner")
    print("\n" + "=" * 60)
    print("Press any key after acknowledging the epilepsy warning to start...")
    print("=" * 60 + "\n")
    
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
            
            # Check for game over state after processing frame
            if engine.game_state == GameState.GAME_OVER:
                # Show game over screen
                new_game_choice = show_game_over_screen()
                if new_game_choice:
                    # Restart game - reset player health and state
                    if engine.player:
                        # Update player entity with full health
                        for i, e in enumerate(engine.entities):
                            if e.id == 'player':
                                updated_player = GameEntity(
                                    id=engine.player.id,
                                    position=engine.player.position,
                                    velocity=engine.player.velocity,
                                    health=100.0,  # Reset health
                                    description=engine.player.description,
                                    properties=engine.player.properties,
                                    height=getattr(engine.player, 'height', PLAYER_HEIGHT_UNITS)
                                )
                                engine.entities[i] = updated_player
                                engine.player = updated_player
                                break
                        engine.game_state = GameState.PLAYING
                        engine.npc_damage_cooldowns = {}  # Reset damage cooldowns
                    continue
                else:
                    # Quit game
                    running = False
                    break
            
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
            
            # Render with pygame (FPV mode with 6DOF), passing settings
            renderer.render(engine.entities, camera_pos, frame, current_fps, camera_yaw, camera_pitch, camera_roll, fpv_mode, engine.sword_swing_state, settings)
            
            # No sleep needed - clock.tick_busy_loop handles timing
            # This allows higher FPS while maintaining smooth frame pacing
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("GAME INTERRUPTED")
        print("=" * 60)
        print("Game was stopped by user (Ctrl+C)")
        print("Thanks for playing Kopis Engine!")
        print("=" * 60)
    except Exception as e:
        print("\n\n" + "=" * 60)
        print("ERROR: Game crashed")
        print("=" * 60)
        print(f"An unexpected error occurred: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nIf this problem persists, please report it with the error details above.")
        print("=" * 60)
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        renderer.cleanup()
        if sound_manager:
            sound_manager.cleanup()
        engine.cleanup()
        
        # Print final stats
        print("\n" + "=" * 60)
        print("GAME SESSION STATISTICS")
        print("=" * 60)
        try:
            stats = engine.get_stats()
            print(json.dumps(stats, indent=2))
        except Exception as e:
            print(f"Could not retrieve statistics: {e}")
        print("=" * 60)
        print("\nThanks for playing Kopis Engine!")
        print("Run the game again anytime with: python kopis_engine.py\n")


if __name__ == "__main__":
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Kopis Engine - A transformer-based 3D game engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kopis_engine.py              # Run with default settings
  python kopis_engine.py --width 1920  # Run with custom window width
  python kopis_engine.py --fullscreen  # Run in fullscreen mode
  
For more information, see the docstring at the top of this file.
        """
    )
    parser.add_argument('--width', type=int, default=800,
                       help='Window width in pixels (default: 800)')
    parser.add_argument('--height', type=int, default=600,
                       help='Window height in pixels (default: 600)')
    parser.add_argument('--fullscreen', action='store_true',
                       help='Start in fullscreen mode')
    parser.add_argument('--no-sound', action='store_true',
                       help='Disable sound effects')
    
    args = parser.parse_args()
    
    # Override renderer size if specified (this would need to be passed to renderer)
    # For now, just show a message if custom size is requested
    if args.width != 800 or args.height != 600:
        print(f"Note: Custom window size requested: {args.width}x{args.height}")
        print("      (Currently using default 800x600 - custom sizes coming soon)")
    
    if args.fullscreen:
        print("Note: Fullscreen mode requested")
        print("      (Press F11 in-game to toggle fullscreen)")
    
    if args.no_sound:
        print("Note: Sound disabled via command line")
        # This would need to be passed to SoundManager
    
    # Run the game
    try:
        main()
    except SystemExit:
        # Allow sys.exit() calls to work normally
        raise
    except Exception as e:
        print("\n" + "=" * 60)
        print("FATAL ERROR: Game could not start")
        print("=" * 60)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Make sure pygame is installed: pip install pygame")
        print("  2. Check that you have Python 3.7 or higher")
        print("  3. Try updating pygame: pip install --upgrade pygame")
        print("  4. Report this error if it persists")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
