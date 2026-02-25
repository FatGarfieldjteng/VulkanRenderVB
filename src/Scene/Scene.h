#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <cstdint>

struct SceneObject {
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f};
    int       meshIndex     = -1;
    int       materialIndex = -1;

    glm::mat4 GetModelMatrix() const {
        glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
        glm::mat4 R = glm::mat4_cast(rotation);
        glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
        return T * R * S;
    }
};

struct DirectionalLight {
    glm::vec3 direction{glm::normalize(glm::vec3(-0.4f, -0.8f, -0.3f))};
    glm::vec3 color{1.0f, 0.95f, 0.85f};
    float     intensity = 3.5f;
};

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
    glm::vec4 sunColor;      // w = intensity
    glm::mat4 cascadeViewProj[4];
    glm::vec4 cascadeSplits;
};
static_assert(sizeof(FrameData) == 512, "FrameData must be 512 bytes for std140");

class Scene {
public:
    SceneObject& AddObject() { mObjects.emplace_back(); return mObjects.back(); }

    std::vector<SceneObject>&       GetObjects()       { return mObjects; }
    const std::vector<SceneObject>& GetObjects() const { return mObjects; }

    DirectionalLight&       GetSun()       { return mSun; }
    const DirectionalLight& GetSun() const { return mSun; }

private:
    std::vector<SceneObject> mObjects;
    DirectionalLight         mSun;
};
