#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>
#include <cstdint>

struct MeshVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec4 tangent{0.0f}; // xyz = tangent, w = handedness (+1 or -1)
};

struct MeshData {
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t>   indices;
    int materialIndex = -1;
};

struct TextureData {
    std::vector<uint8_t> pixels;
    uint32_t width    = 0;
    uint32_t height   = 0;
    uint32_t channels = 4;
};

struct MaterialData {
    int       baseColorTextureIndex        = -1;
    int       normalTextureIndex           = -1;
    int       metallicRoughnessTextureIndex = -1;
    int       occlusionTextureIndex        = -1;
    int       emissiveTextureIndex         = -1;
    glm::vec4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
    float     metallicFactor  = 0.0f;
    float     roughnessFactor = 0.5f;
    glm::vec3 emissiveFactor{0.0f};
};

struct MeshInstance {
    int       meshIndex = -1;
    glm::vec3 translation{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f};
};

struct ModelData {
    std::vector<MeshData>     meshes;
    std::vector<TextureData>  textures;
    std::vector<MaterialData> materials;
    std::vector<MeshInstance> instances;
};

class ModelLoader {
public:
    static bool LoadGLTF(const std::string& path, ModelData& outModel);
    static void GenerateProceduralCube(ModelData& outModel);
    static void GenerateGroundPlane(MeshData& outMesh, float halfSize = 20.0f);
    static void GenerateUVSphere(MeshData& outMesh, float radius = 0.5f,
                                 uint32_t sectors = 64, uint32_t stacks = 32);
    static void ComputeTangents(MeshData& mesh);

    static void SortMeshesByVolume(std::vector<MeshData>& meshes);
};
