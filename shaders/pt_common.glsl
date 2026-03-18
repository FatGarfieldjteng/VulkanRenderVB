
#ifndef PT_COMMON_GLSL
#define PT_COMMON_GLSL

struct HitPayload {
    vec3  hitPos;
    float hitT;          // -1 = miss
    vec3  normal;
    float metallic;
    vec3  albedo;
    float roughness;
    vec3  emissive;
    uint  materialFlags;
};

const uint MAT_FLAG_EMISSIVE     = 1u;
const uint MAT_FLAG_TRANSMISSIVE = 2u;

struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 texCoord;
    vec4 tangent;
};

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

struct InstanceInfo {
    int  vertexOffset;
    uint firstIndex;
    uint indexCount;
    uint materialIndex;
};

const float PI     = 3.14159265359;
const float INV_PI = 0.31830988618;
const float EPSILON = 1e-4;

// ---------- RNG (PCG) ----------

uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// ---------- Sampling ----------

vec3 sampleCosineHemisphere(vec3 N, vec2 u) {
    float phi = 2.0 * PI * u.x;
    float cosTheta = sqrt(u.y);
    float sinTheta = sqrt(1.0 - u.y);
    vec3 T = normalize(cross(N, abs(N.y) < 0.99 ? vec3(0,1,0) : vec3(1,0,0)));
    vec3 B = cross(N, T);
    return normalize(T * cos(phi) * sinTheta + B * sin(phi) * sinTheta + N * cosTheta);
}

vec3 sampleGGXVNDF(vec3 Ve, float roughness, vec2 u) {
    float a = roughness * roughness;
    vec3 Vh = normalize(vec3(a * Ve.x, a * Ve.y, Ve.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1,0,0);
    vec3 T2 = cross(Vh, T1);
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1*t1) + s * t2;
    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1*t1 - t2*t2)) * Vh;
    return normalize(vec3(a * Nh.x, a * Nh.y, max(0.0, Nh.z)));
}

// ---------- BRDF ----------

float DistributionGGX(float NdotH, float a2) {
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float GeometrySmithG1(float NdotV, float a2) {
    return 2.0 * NdotV / (NdotV + sqrt(a2 + (1.0 - a2) * NdotV * NdotV));
}

float GeometrySmith(float NdotV, float NdotL, float a2) {
    return GeometrySmithG1(NdotV, a2) * GeometrySmithG1(NdotL, a2);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

#endif
