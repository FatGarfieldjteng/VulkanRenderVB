#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : require

#include "pt_common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(scalar, set = 0, binding = 6) readonly buffer VertexBuffer  { Vertex vertices[]; };
layout(scalar, set = 0, binding = 7) readonly buffer IndexBuffer   { uint indices[]; };
layout(std430, set = 0, binding = 8) readonly buffer MaterialBuffer { MaterialParams materials[]; };
layout(std430, set = 0, binding = 9) readonly buffer InstanceBuffer { InstanceInfo instanceInfos[]; };

layout(set = 1, binding = 0) uniform sampler2D textures[];

hitAttributeEXT vec2 attribs;

void main() {
    uint instIdx = gl_InstanceCustomIndexEXT;
    InstanceInfo info = instanceInfos[instIdx];

    uint i0 = indices[info.firstIndex + gl_PrimitiveID * 3 + 0];
    uint i1 = indices[info.firstIndex + gl_PrimitiveID * 3 + 1];
    uint i2 = indices[info.firstIndex + gl_PrimitiveID * 3 + 2];

    Vertex v0 = vertices[uint(int(i0) + info.vertexOffset)];
    Vertex v1 = vertices[uint(int(i1) + info.vertexOffset)];
    Vertex v2 = vertices[uint(int(i2) + info.vertexOffset)];

    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    vec3 localPos    = v0.position * bary.x + v1.position * bary.y + v2.position * bary.z;
    vec3 localNormal = v0.normal   * bary.x + v1.normal   * bary.y + v2.normal   * bary.z;
    vec2 texCoord    = v0.texCoord * bary.x + v1.texCoord * bary.y + v2.texCoord * bary.z;

    mat3 normalMat = mat3(gl_ObjectToWorldEXT);
    vec3 N = normalize(normalMat * localNormal);

    MaterialParams mat = materials[info.materialIndex];

    vec4 baseColor = mat.baseColorFactor *
        texture(textures[nonuniformEXT(mat.baseColorTexIdx)], texCoord);

    vec4 mrSample = texture(textures[nonuniformEXT(mat.metallicRoughnessTexIdx)], texCoord);
    float roughness = clamp(mat.roughnessFactor * mrSample.g, 0.04, 1.0);
    float metallic  = clamp(mat.metallicFactor  * mrSample.b, 0.0, 1.0);

    // Normal mapping
    vec3 tangentLocal = v0.tangent.xyz * bary.x + v1.tangent.xyz * bary.y + v2.tangent.xyz * bary.z;
    float tangentW    = v0.tangent.w;
    vec3 T = normalize(normalMat * tangentLocal);

    if (length(T) > 0.001) {
        T = normalize(T - dot(T, N) * N);
        vec3 B = cross(N, T) * tangentW;
        mat3 TBN = mat3(T, B, N);
        vec3 normalSample = texture(textures[nonuniformEXT(mat.normalTexIdx)], texCoord).rgb;
        normalSample = normalSample * 2.0 - 1.0;
        N = normalize(TBN * normalSample);
    }

    // Emissive
    vec3 emissive = vec3(0.0);
    uint flags = 0u;
    vec3 emissiveTex = texture(textures[nonuniformEXT(mat.emissiveTexIdx)], texCoord).rgb;
    emissive = emissiveTex * mat.baseColorFactor.rgb;
    if (dot(emissive, emissive) > 0.0)
        flags |= MAT_FLAG_EMISSIVE;

    // World-space hit position
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // Pack all data into payload
    payload.hitPos        = worldPos;
    payload.hitT          = gl_HitTEXT;
    payload.normal        = N;
    payload.metallic      = metallic;
    payload.albedo        = baseColor.rgb;
    payload.roughness     = roughness;
    payload.emissive      = emissive;
    payload.materialFlags = flags;
}
