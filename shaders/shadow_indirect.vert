#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent;

struct ObjectData {
    mat4  model;
    vec4  aabbMin;
    vec4  aabbMax;
    uint  materialIndex;
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
};

layout(std430, set = 0, binding = 0) readonly buffer ObjectSSBO {
    ObjectData objects[];
};

layout(push_constant) uniform PushConstants {
    mat4 cascadeViewProj;
} pc;

void main() {
    ObjectData obj = objects[gl_BaseInstance];
    gl_Position = pc.cascadeViewProj * obj.model * vec4(inPosition, 1.0);
}
