#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in float fragViewDepth;
layout(location = 4) in vec4 fragTangent;
layout(location = 5) flat in uint fragMaterialIndex;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform DebugPush {
    uint mode;
} debug;

vec3 HeatmapColor(float t) {
    t = clamp(t, 0.0, 1.0);
    return vec3(
        clamp(1.5 - abs(t - 0.75) * 4.0, 0.0, 1.0),
        clamp(1.5 - abs(t - 0.5)  * 4.0, 0.0, 1.0),
        clamp(1.5 - abs(t - 0.25) * 4.0, 0.0, 1.0)
    );
}

void main() {
    switch (debug.mode) {
        case 1: // Wireframe - simple white
            outColor = vec4(1.0, 1.0, 1.0, 1.0);
            break;
        case 2: // World normals
            outColor = vec4(normalize(fragNormal) * 0.5 + 0.5, 1.0);
            break;
        case 3: { // Tangent normals
            vec3 N = normalize(fragNormal);
            vec3 T = normalize(fragTangent.xyz);
            vec3 B = cross(N, T) * fragTangent.w;
            outColor = vec4(T * 0.5 + 0.5, 1.0);
            break;
        }
        case 4: // UVs
            outColor = vec4(fract(fragTexCoord), 0.0, 1.0);
            break;
        case 5: { // Mip level (false color) - approximate from screen derivatives
            vec2 dx = dFdx(fragTexCoord);
            vec2 dy = dFdy(fragTexCoord);
            float maxDeriv = max(length(dx), length(dy));
            float mip = log2(max(maxDeriv * 1024.0, 1.0));
            outColor = vec4(HeatmapColor(mip / 10.0), 1.0);
            break;
        }
        case 6: // Overdraw (simple - visualize constant color for additive blending)
            outColor = vec4(0.1, 0.02, 0.0, 1.0);
            break;
        case 7: { // Cascade coloring - use view depth to estimate cascade
            float d = fragViewDepth;
            vec3 col;
            if      (d < 5.0)  col = vec3(1.0, 0.2, 0.2);
            else if (d < 15.0) col = vec3(0.2, 1.0, 0.2);
            else if (d < 40.0) col = vec3(0.2, 0.2, 1.0);
            else               col = vec3(1.0, 1.0, 0.2);
            outColor = vec4(col, 1.0);
            break;
        }
        case 8: { // Depth buffer (linearized)
            float linearZ = fragViewDepth / 100.0;
            outColor = vec4(vec3(1.0 - linearZ), 1.0);
            break;
        }
        default:
            outColor = vec4(1.0, 0.0, 1.0, 1.0);
            break;
    }
}
