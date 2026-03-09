#version 460

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrColor;
layout(set = 0, binding = 1) uniform sampler2D bloomTex;
layout(set = 0, binding = 2) uniform sampler2D aoTex;

layout(std430, set = 0, binding = 3) readonly buffer ExposureSSBO {
    float autoExposure;
};

layout(push_constant) uniform Params {
    uint  curveType;        // 0 = ACES, 1 = AgX
    float exposureBias;
    float whitePoint;       // ACES
    float shoulderStrength; // ACES
    float linearStrength;   // ACES
    float linearAngle;      // ACES
    float toeStrength;      // ACES
    float saturation;       // AgX
    float agxPunch;         // AgX
    float bloomStrength;
    uint  useAutoExposure;
};

// ---- ACES Filmic (Uncharted 2 variant with configurable params) ----
vec3 ACESFilmic(vec3 x) {
    float A = shoulderStrength;
    float B = linearStrength;
    float C = linearAngle;
    float D = toeStrength;
    float W = whitePoint;

    vec3 num = (x * (A * x + C * B) + D * 0.02);
    vec3 den = (x * (A * x + B) + D * 0.3);
    vec3 mapped = num / den - vec3(0.02 / 0.3);

    float wNum = (W * (A * W + C * B) + D * 0.02);
    float wDen = (W * (A * W + B) + D * 0.3);
    float wMapped = wNum / wDen - 0.02 / 0.3;

    return mapped / wMapped;
}

// ---- AgX ----
vec3 AgXDefaultContrastApprox(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return + 15.5     * x4 * x2
           - 40.14    * x4 * x
           + 31.96    * x4
           - 6.868    * x2 * x
           + 0.4298   * x2
           + 0.1191   * x
           - 0.00232;
}

vec3 AgXToneMap(vec3 color) {
    const mat3 agxTransform = mat3(
        0.842479062253094,  0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772,  0.0784336,
        0.0792237451477643, 0.0791661274605434,  0.879142973793104
    );

    const mat3 agxTransformInv = mat3(
        1.19687900512017,   -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417,   -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433,  1.15107367264116
    );

    const float minEV = -12.47393;
    const float maxEV = 4.026069;

    color = agxTransform * color;
    color = clamp(log2(color), minEV, maxEV);
    color = (color - minEV) / (maxEV - minEV);
    color = AgXDefaultContrastApprox(color);

    color = agxTransformInv * color;

    // AgX punch (increased saturation/contrast)
    vec3 lum = vec3(dot(color, vec3(0.2126, 0.7152, 0.0722)));
    color = lum + (color - lum) * saturation;
    color = mix(color, color * color * (3.0 - 2.0 * color), agxPunch);

    return color;
}

void main() {
    vec3 hdr = texture(hdrColor, fragUV).rgb;
    float ao = texture(aoTex, fragUV).r;
    vec3 bloom = texture(bloomTex, fragUV).rgb;

    hdr *= ao;
    hdr += bloom * bloomStrength;

    float ev = useAutoExposure != 0 ? autoExposure : 1.0;
    hdr *= ev * exp2(exposureBias);

    vec3 mapped;
    if (curveType == 0)
        mapped = ACESFilmic(hdr);
    else
        mapped = AgXToneMap(hdr);

    mapped = clamp(mapped, 0.0, 1.0);
    outColor = vec4(mapped, 1.0);
}
