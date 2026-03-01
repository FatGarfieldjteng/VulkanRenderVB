#pragma once

#include <glm/glm.hpp>
#include <cstdint>

/// GPU-side material parameters, must match GLSL layout (std430, 48 bytes).
struct GPUMaterialData {
    glm::vec4 baseColorFactor{1.0f};
    float     metallicFactor       = 0.0f;
    float     roughnessFactor      = 0.5f;
    uint32_t  baseColorTexIdx      = 0;
    uint32_t  normalTexIdx         = 0;
    uint32_t  metallicRoughnessTexIdx = 0;
    uint32_t  aoTexIdx             = 0;
    uint32_t  emissiveTexIdx       = 0;
    float     _pad                 = 0.0f;
};
static_assert(sizeof(GPUMaterialData) == 48, "GPUMaterialData must be 48 bytes for std430");

/// GPU-side per-frame uniform data, must match GLSL layout (std140, 512 bytes).
struct FrameData {
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 viewProjection;
    glm::vec4 cameraPos;
    glm::vec4 sunDirection;
    glm::vec4 sunColor;           // w = intensity
    glm::mat4 cascadeViewProj[4];
    glm::vec4 cascadeSplits;
};
static_assert(sizeof(FrameData) == 512, "FrameData must be 512 bytes for std140");
