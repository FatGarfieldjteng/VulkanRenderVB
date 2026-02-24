#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec3 fragNormal;

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    uint textureIndex;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 texColor = texture(textures[nonuniformEXT(pc.textureIndex)], fragTexCoord);

    vec3 lightDir = normalize(vec3(1.0, 1.0, 0.5));
    vec3 normal   = normalize(fragNormal);
    float diffuse = max(dot(normal, lightDir), 0.0);
    float ambient = 0.15;

    outColor = vec4(texColor.rgb * (ambient + diffuse * 0.85), texColor.a);
}
