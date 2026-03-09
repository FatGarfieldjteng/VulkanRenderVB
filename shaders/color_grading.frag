#version 460

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D inputColor;
layout(set = 0, binding = 1) uniform sampler3D colorLUT;

layout(push_constant) uniform Params {
    float lutStrength;
    float vignetteIntensity;
    float vignetteRadius;
    float grainStrength;
    float grainTime;
    float chromaticAberration;
    uvec2 resolution;
};

float Random(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 uv = fragUV;
    vec3 color;

    // Chromatic aberration
    if (chromaticAberration > 0.0) {
        vec2 dir = uv - 0.5;
        float dist = length(dir);
        vec2 offset = dir * dist * chromaticAberration * 0.01;
        color.r = texture(inputColor, uv + offset).r;
        color.g = texture(inputColor, uv).g;
        color.b = texture(inputColor, uv - offset).b;
    } else {
        color = texture(inputColor, uv).rgb;
    }

    // 3D LUT color grading
    if (lutStrength > 0.0) {
        vec3 lutCoord = clamp(color, 0.0, 1.0);
        float lutSize = 32.0;
        lutCoord = lutCoord * ((lutSize - 1.0) / lutSize) + 0.5 / lutSize;
        vec3 graded = texture(colorLUT, lutCoord).rgb;
        color = mix(color, graded, lutStrength);
    }

    // Vignette
    if (vignetteIntensity > 0.0) {
        vec2 center = uv - 0.5;
        float vignette = 1.0 - smoothstep(vignetteRadius, vignetteRadius + 0.3,
                                           length(center) * 1.414);
        color *= mix(1.0, vignette, vignetteIntensity);
    }

    // Film grain
    if (grainStrength > 0.0) {
        float grain = Random(uv + vec2(grainTime)) * 2.0 - 1.0;
        color += color * grain * grainStrength;
    }

    outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
