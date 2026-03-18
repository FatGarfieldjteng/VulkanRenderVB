#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 1) rayPayloadInEXT float shadowPayload;

void main() {
    shadowPayload = 1.0; // Not in shadow
}
