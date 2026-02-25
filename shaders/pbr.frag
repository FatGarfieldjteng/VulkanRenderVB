#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in float fragViewDepth;

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(set = 1, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 projection;
    mat4 viewProjection;
    vec4 cameraPos;
    vec4 sunDirection;
    vec4 sunColor;       // w = intensity
    mat4 cascadeViewProj[4];
    vec4 cascadeSplits;
} frame;

struct MaterialParams {
    vec4  baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    uint  baseColorTexIdx;
    uint  normalTexIdx;
    uint  metallicRoughnessTexIdx;
    uint  aoTexIdx;
    uint  emissiveTexIdx;
    float _pad;
};

layout(std430, set = 1, binding = 1) readonly buffer MaterialSSBO {
    MaterialParams materials[];
};

layout(set = 1, binding = 2) uniform sampler2DArrayShadow shadowMap;

layout(push_constant) uniform PushConstants {
    mat4 model;
    uint materialIndex;
} pc;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom  = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float SampleShadowPCF(vec3 worldPos, uint cascade) {
    vec4 lightSpace = frame.cascadeViewProj[cascade] * vec4(worldPos, 1.0);
    vec3 projCoords = lightSpace.xyz / lightSpace.w;

    vec2 shadowUV = projCoords.xy * 0.5 + 0.5;
    float currentDepth = projCoords.z;

    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 ||
        shadowUV.y < 0.0 || shadowUV.y > 1.0)
        return 1.0;

    float shadow = 0.0;
    vec2 texelSize = vec2(1.0 / 2048.0);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            shadow += texture(shadowMap, vec4(shadowUV + offset, float(cascade), currentDepth));
        }
    }
    return shadow / 9.0;
}

float ComputeShadow(vec3 worldPos) {
    uint cascade = 3;
    if      (fragViewDepth < frame.cascadeSplits.x) cascade = 0;
    else if (fragViewDepth < frame.cascadeSplits.y) cascade = 1;
    else if (fragViewDepth < frame.cascadeSplits.z) cascade = 2;
    return SampleShadowPCF(worldPos, cascade);
}

void main() {
    MaterialParams mat = materials[pc.materialIndex];

    vec4 baseColor = mat.baseColorFactor *
        texture(textures[nonuniformEXT(mat.baseColorTexIdx)], fragTexCoord);

    vec4 mrSample = texture(textures[nonuniformEXT(mat.metallicRoughnessTexIdx)], fragTexCoord);
    float roughness = clamp(mat.roughnessFactor * mrSample.g, 0.04, 1.0);
    float metallic  = clamp(mat.metallicFactor  * mrSample.b, 0.0, 1.0);

    float ao = texture(textures[nonuniformEXT(mat.aoTexIdx)], fragTexCoord).r;

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(frame.cameraPos.xyz - fragWorldPos);

    vec3 albedo = baseColor.rgb;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 L = normalize(-frame.sunDirection.xyz);
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);

    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3  F = FresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 specular = (D * G * F) / (4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    float shadow = ComputeShadow(fragWorldPos);
    vec3 radiance = frame.sunColor.rgb * frame.sunColor.w;
    vec3 Lo = (diffuse + specular) * radiance * NdotL * shadow;

    float hemiFactor = dot(N, vec3(0, 1, 0)) * 0.5 + 0.5;
    vec3 skyColor = mix(vec3(0.05, 0.05, 0.08), vec3(0.15, 0.2, 0.35), hemiFactor);
    vec3 ambient = skyColor * albedo * ao;

    vec3 emissive = texture(textures[nonuniformEXT(mat.emissiveTexIdx)], fragTexCoord).rgb;

    vec3 color = ambient + Lo + emissive;

    // Reinhard tonemapping (no manual gamma -- sRGB framebuffer handles it)
    color = color / (color + vec3(1.0));

    outColor = vec4(color, baseColor.a);
}
