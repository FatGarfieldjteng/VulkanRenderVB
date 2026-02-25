#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <cstdint>

struct MeshVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
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

struct ModelData {
    std::vector<MeshData>     meshes;
    std::vector<TextureData>  textures;
    std::vector<MaterialData> materials;
};

class ModelLoader {
public:
    /// Load a glTF 2.0 file (.gltf or .glb).
    static bool LoadGLTF(const std::string& path, ModelData& outModel);

    /// Generate a textured cube with a checkerboard pattern.
    static void GenerateProceduralCube(ModelData& outModel);

    /// Generate a ground plane mesh.
    static void GenerateGroundPlane(MeshData& outMesh, float halfSize = 20.0f);
};
