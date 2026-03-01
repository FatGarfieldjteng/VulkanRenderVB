#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3  fragWorldPos;
layout(location = 1) in vec3  fragNormal;
layout(location = 2) in vec2  fragTexCoord;
layout(location = 3) in float fragViewDepth;
layout(location = 4) in vec4  fragTangent;

layout(set = 0, binding = 0) uniform sampler2D textures[];

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
layout(set = 1, binding = 3) uniform samplerCube irradianceMap;
layout(set = 1, binding = 4) uniform samplerCube prefilterMap;
layout(set = 1, binding = 5) uniform sampler2D   brdfLUT;

layout(push_constant) uniform PushConstants {
    mat4 model;
    uint materialIndex;
} pc;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;
const float MAX_PREFILTER_LOD = 4.0;

// ---- PBR functions ----

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

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ---- Shadow functions ----

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

// ---- Normal mapping ----

vec3 ComputeNormal(MaterialParams mat) {
    vec3 Ng = normalize(fragNormal);
    vec3 T  = fragTangent.xyz;

    if (length(T) < 0.01)
        return Ng;

    T = normalize(T - dot(T, Ng) * Ng);
    vec3 B = cross(Ng, T) * fragTangent.w;
    mat3 TBN = mat3(T, B, Ng);

    vec3 normalSample = texture(textures[nonuniformEXT(mat.normalTexIdx)], fragTexCoord).rgb;
    normalSample = normalSample * 2.0 - 1.0;
    return normalize(TBN * normalSample);
}

// ---- Main ----

void main() {
    MaterialParams mat = materials[pc.materialIndex];

    vec4 baseColor = mat.baseColorFactor *
        texture(textures[nonuniformEXT(mat.baseColorTexIdx)], fragTexCoord);

    vec4 mrSample = texture(textures[nonuniformEXT(mat.metallicRoughnessTexIdx)], fragTexCoord);
    float roughness = clamp(mat.roughnessFactor * mrSample.g, 0.04, 1.0);
    float metallic  = clamp(mat.metallicFactor  * mrSample.b, 0.0, 1.0);

    float ao = texture(textures[nonuniformEXT(mat.aoTexIdx)], fragTexCoord).r;

    vec3 N = ComputeNormal(mat);
    vec3 V = normalize(frame.cameraPos.xyz - fragWorldPos);

    vec3 albedo = baseColor.rgb;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // ---- Direct sunlight (Cook-Torrance) ----
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

    // ---- IBL ambient ----
    float NdotV = max(dot(N, V), 0.0);
    vec3 F_ibl = FresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kD_ibl = (1.0 - F_ibl) * (1.0 - metallic);

    vec3 irradiance  = texture(irradianceMap, N).rgb;
    vec3 diffuseIBL  = kD_ibl * irradiance * albedo;

    vec3 R = reflect(-V, N);
    vec3 prefilteredColor = textureLod(prefilterMap, R, roughness * MAX_PREFILTER_LOD).rgb;
    vec2 brdf = texture(brdfLUT, vec2(NdotV, roughness)).rg;
    vec3 specularIBL = prefilteredColor * (F_ibl * brdf.x + brdf.y);

    vec3 ambient = (diffuseIBL + specularIBL) * ao;

    // ---- Emissive ----
    vec3 emissive = texture(textures[nonuniformEXT(mat.emissiveTexIdx)], fragTexCoord).rgb;

    // ---- Final ----
    vec3 color = ambient + Lo + emissive;

    // Reinhard tonemapping (sRGB swapchain handles gamma)
    color = color / (color + vec3(1.0));

    outColor = vec4(color, baseColor.a);
}
