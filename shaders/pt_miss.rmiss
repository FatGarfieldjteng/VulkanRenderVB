#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "pt_common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(set = 0, binding = 10) uniform samplerCube envMap;

void main() {
    vec3 dir = gl_WorldRayDirectionEXT;
    vec3 envColor = texture(envMap, dir).rgb;

    payload.hitT          = -1.0;
    payload.albedo        = envColor;
    payload.normal        = vec3(0.0);
    payload.emissive      = vec3(0.0);
    payload.materialFlags = 0u;
}
