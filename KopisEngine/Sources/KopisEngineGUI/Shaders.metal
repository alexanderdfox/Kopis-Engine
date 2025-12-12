#include <metal_stdlib>
using namespace metal;

// Metal 4 (MSL 2.4+) compatible shaders with modern features
// Requires macOS 13.0+ and Metal 4 capable GPU

// Uniform structure matching Swift
struct RaycastUniforms {
    uint width;
    uint height;
    float cameraX;
    float cameraY;
    float cameraZ;
    float cameraYaw;
    float cameraPitch;
    float fov;
    float maxDepth;
    float cellSize;
};

// Entity data for billboard rendering
struct EntityData {
    float3 position;
    float3 velocity;
    float health;
    uint idHash;
    float radius;
};

// Entity rendering uniforms
struct EntityUniforms {
    uint entityCount;
    float3 cameraPos;
    float cameraYaw;
    float cameraPitch;
    float fov;
    float2 screenSize;
};

// Simple DDA raycast function (optimized for Metal 4)
float2 raycastWall(
    float camX, float camY,
    float rayDirX, float rayDirY,
    float maxDepth, float cellSize
) {
    // Avoid division by zero
    float rayDirX_safe = abs(rayDirX) < 0.0001 ? (rayDirX < 0 ? -0.0001 : 0.0001) : rayDirX;
    float rayDirY_safe = abs(rayDirY) < 0.0001 ? (rayDirY < 0 ? -0.0001 : 0.0001) : rayDirY;
    
    int mapX = int(floor(camX / cellSize));
    int mapY = int(floor(camY / cellSize));
    
    float deltaDistX = abs(1.0 / rayDirX_safe);
    float deltaDistY = abs(1.0 / rayDirY_safe);
    
    int stepX, stepY;
    float sideDistX, sideDistY;
    
    if (rayDirX_safe < 0.0) {
        stepX = -1;
        sideDistX = (camX / cellSize - float(mapX)) * deltaDistX;
    } else {
        stepX = 1;
        sideDistX = (float(mapX) + 1.0 - camX / cellSize) * deltaDistX;
    }
    
    if (rayDirY_safe < 0.0) {
        stepY = -1;
        sideDistY = (camY / cellSize - float(mapY)) * deltaDistY;
    } else {
        stepY = 1;
        sideDistY = (float(mapY) + 1.0 - camY / cellSize) * deltaDistY;
    }
    
    // Perform DDA
    bool hit = false;
    int side = 0;
    float distance = 0.0;
    
    // Simplified collision check (assume walls at integer grid positions)
    // In a full implementation, this would query the maze structure
    for (int i = 0; i < 1000; i++) {
        if (sideDistX < sideDistY) {
            sideDistX += deltaDistX;
            mapX += stepX;
            side = 0;
        } else {
            sideDistY += deltaDistY;
            mapY += stepY;
            side = 1;
        }
        
        // Check collision (simplified - check if mapX/mapY is odd for wall)
        // This is a placeholder - real implementation needs maze data
        bool isWall = ((mapX % 2 == 1) || (mapY % 2 == 1));
        
        if (isWall) {
            if (side == 0) {
                distance = (float(mapX) - camX / cellSize + (1.0 - float(stepX)) / 2.0) / rayDirX_safe;
            } else {
                distance = (float(mapY) - camY / cellSize + (1.0 - float(stepY)) / 2.0) / rayDirY_safe;
            }
            distance *= cellSize;
            hit = true;
            break;
        }
        
        if (distance > maxDepth) break;
    }
    
    return float2(distance, float(side));
}

// Metal 4 compute shader for raycasting (optimized)
kernel void raycast_compute(
    texture2d<float, access::write> output [[texture(0)]],
    constant RaycastUniforms& uniforms [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }
    
    float width = float(uniforms.width);
    float height = float(uniforms.height);
    
    // Calculate ray direction
    float cameraX = 2.0 * float(gid.x) / width - 1.0;
    
    float forwardX = sin(uniforms.cameraYaw);
    float forwardY = -cos(uniforms.cameraYaw);
    float rightX = cos(uniforms.cameraYaw);
    float rightY = sin(uniforms.cameraYaw);
    
    float rayDirX = forwardX + rightX * cameraX * tan(uniforms.fov / 2.0);
    float rayDirY = forwardY + rightY * cameraX * tan(uniforms.fov / 2.0);
    
    // Raycast
    float2 result = raycastWall(
        uniforms.cameraX,
        uniforms.cameraY,
        rayDirX,
        rayDirY,
        uniforms.maxDepth,
        uniforms.cellSize
    );
    
    float distance = result.x;
    float side = result.y;
    
    // Calculate wall height
    float lineHeight = abs(height / (distance / uniforms.cellSize));
    float horizonY = height / 2.0 + tan(uniforms.cameraPitch) * height / 2.0;
    float drawStart = -lineHeight / 2.0 + height / 2.0 + tan(uniforms.cameraPitch) * height / 2.0;
    float drawEnd = lineHeight / 2.0 + height / 2.0 + tan(uniforms.cameraPitch) * height / 2.0;
    
    float4 color;
    
    // Check if this pixel is part of the wall
    if (float(gid.y) >= drawStart && float(gid.y) < drawEnd && distance < uniforms.maxDepth) {
        // Wall color with distance-based shading
        float brightness = max(0.3, min(1.0, 1.0 - distance / 500.0));
        float baseR = 30.0 + float((int(uniforms.cameraX / uniforms.cellSize) * 7) % 50);
        float baseG = 25.0 + float((int(uniforms.cameraX / uniforms.cellSize) * 5) % 35);
        float baseB = 20.0 + float((int(uniforms.cameraX / uniforms.cellSize) * 3) % 30);
        
        float wallR = baseR * brightness * (side == 1.0 ? 0.8 : 1.0);
        float wallG = baseG * brightness * (side == 1.0 ? 0.8 : 1.0);
        float wallB = baseB * brightness * (side == 1.0 ? 0.8 : 1.0);
        
        color = float4(wallR / 255.0, wallG / 255.0, wallB / 255.0, 1.0);
    } else if (float(gid.y) < horizonY) {
        // Ceiling
        color = float4(0.2, 0.2, 0.2, 1.0);
    } else {
        // Floor
        color = float4(0.16, 0.16, 0.16, 1.0);
    }
    
    output.write(color, gid);
}

// Vertex shader for fullscreen quad
struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut vertex_main(uint vertexID [[vertex_id]]) {
    VertexOut out;
    out.position = float4(
        (vertexID == 2) ? 3.0 : -1.0,
        (vertexID == 1) ? -3.0 : 1.0,
        0.0,
        1.0
    );
    out.texCoord = float2(
        (vertexID == 2) ? 2.0 : 0.0,
        (vertexID == 1) ? 2.0 : 0.0
    );
    return out;
}

// Fragment shader for fullscreen quad
fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    texture2d<float> texture [[texture(0)]]
) {
    constexpr sampler s(min_filter::linear, mag_filter::linear);
    return texture.sample(s, in.texCoord);
}

// Metal 4: Entity billboard rendering compute shader
kernel void render_entities_compute(
    texture2d<float, access::read_write> output [[texture(0)]],
    constant EntityUniforms& uniforms [[buffer(0)]],
    const device EntityData* entities [[buffer(1)]],
    constant RaycastUniforms& raycastUniforms [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uniforms.screenSize.x || gid.y >= uniforms.screenSize.y) {
        return;
    }
    
    // Sample existing raycast output
    float4 baseColor = output.read(gid);
    
    // Render entities as billboards (simplified - full implementation would use proper depth sorting)
    float2 screenPos = float2(gid);
    
    for (uint i = 0; i < uniforms.entityCount; i++) {
        EntityData entity = entities[i];
        
        // Transform to camera space
        float3 dx = entity.position - uniforms.cameraPos;
        
        float yawRad = uniforms.cameraYaw;
        float pitchRad = uniforms.cameraPitch;
        
        // Rotate by yaw
        float tempX = dx.x * cos(yawRad) - dx.y * sin(yawRad);
        float tempY = dx.x * sin(yawRad) + dx.y * cos(yawRad);
        float tempZ = dx.z;
        
        // Rotate by pitch (3D projection)
        float finalY = tempY * cos(pitchRad) - tempZ * sin(pitchRad);
        float finalZ = tempY * sin(pitchRad) + tempZ * cos(pitchRad);
        
        // Project to screen
        float fovScale = 1.0 / tan(uniforms.fov / 2.0);
        float depth = max(0.1, -tempX);
        
        if (depth <= 0 || tempX < 0) continue; // Behind camera
        
        float2 entityScreenPos = float2(
            uniforms.screenSize.x / 2.0 + finalY * fovScale * (uniforms.screenSize.y / depth),
            uniforms.screenSize.y / 2.0 - finalZ * fovScale * (uniforms.screenSize.y / depth)
        );
        
        float entitySize = entity.radius * 2.0 * (uniforms.screenSize.y / depth);
        float dist = length(screenPos - entityScreenPos);
        
        if (dist < entitySize) {
            // Draw entity as circle
            float alpha = 1.0 - (dist / entitySize);
            float4 entityColor = float4(1.0, 0.0, 0.0, alpha * 0.8); // Red for NPCs
            baseColor = mix(baseColor, entityColor, entityColor.a);
        }
    }
    
    output.write(baseColor, gid);
}
