#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent;

layout(std140, set = 1, binding = 0) uniform FrameUBO {
    mat4  view;
    mat4  projection;
    mat4  viewProjection;
    vec4  cameraPos;
    vec4  sunDirection;
    vec4  sunColor;
    mat4  cascadeViewProj[4];
    vec4  cascadeSplits;
} frame;

layout(push_constant) uniform PushConstants {
    mat4 model;
    uint materialIndex;
} pc;

layout(location = 0) out vec3  fragWorldPos;
layout(location = 1) out vec3  fragNormal;
layout(location = 2) out vec2  fragTexCoord;
layout(location = 3) out float fragViewDepth;
layout(location = 4) out vec4  fragTangent;

void main() {
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    fragWorldPos  = worldPos.xyz;

    mat3 normalMat = mat3(pc.model);
    fragNormal  = normalMat * inNormal;
    fragTangent = vec4(normalMat * inTangent.xyz, inTangent.w);

    fragTexCoord = inTexCoord;

    vec4 viewPos  = frame.view * worldPos;
    fragViewDepth = -viewPos.z;

    gl_Position = frame.viewProjection * worldPos;
}
