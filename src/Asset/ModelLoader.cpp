#include "Asset/ModelLoader.h"
#include "Core/Logger.h"

#include <tiny_gltf.h>
#include <algorithm>
#include <cstring>

bool ModelLoader::LoadGLTF(const std::string& path, ModelData& outModel) {
    tinygltf::Model    gltfModel;
    tinygltf::TinyGLTF loader;
    std::string        err, warn;

    bool ok = false;
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".glb") {
        ok = loader.LoadBinaryFromFile(&gltfModel, &err, &warn, path);
    } else {
        ok = loader.LoadASCIIFromFile(&gltfModel, &err, &warn, path);
    }

    if (!warn.empty()) LOG_WARN("glTF warning: {}", warn);
    if (!err.empty())  LOG_ERROR("glTF error: {}", err);
    if (!ok) {
        LOG_ERROR("Failed to load glTF: {}", path);
        return false;
    }

    // --- textures ---
    for (const auto& image : gltfModel.images) {
        TextureData tex;
        tex.width    = static_cast<uint32_t>(image.width);
        tex.height   = static_cast<uint32_t>(image.height);
        tex.channels = 4;

        if (image.component == 4) {
            tex.pixels.assign(image.image.begin(), image.image.end());
        } else if (image.component == 3) {
            tex.pixels.resize(tex.width * tex.height * 4);
            for (uint32_t i = 0; i < tex.width * tex.height; i++) {
                tex.pixels[i * 4 + 0] = image.image[i * 3 + 0];
                tex.pixels[i * 4 + 1] = image.image[i * 3 + 1];
                tex.pixels[i * 4 + 2] = image.image[i * 3 + 2];
                tex.pixels[i * 4 + 3] = 255;
            }
        } else {
            tex.pixels.assign(image.image.begin(), image.image.end());
        }

        outModel.textures.push_back(std::move(tex));
    }

    // --- materials ---
    for (const auto& mat : gltfModel.materials) {
        MaterialData material;
        material.baseColorFactor = glm::vec4(
            static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[0]),
            static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[1]),
            static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[2]),
            static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[3]));

        int texIndex = mat.pbrMetallicRoughness.baseColorTexture.index;
        if (texIndex >= 0 && texIndex < static_cast<int>(gltfModel.textures.size())) {
            material.baseColorTextureIndex = gltfModel.textures[texIndex].source;
        }
        outModel.materials.push_back(material);
    }

    // --- meshes ---
    for (const auto& mesh : gltfModel.meshes) {
        for (const auto& primitive : mesh.primitives) {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES && primitive.mode != -1)
                continue;

            MeshData meshData;
            meshData.materialIndex = primitive.material;

            // Positions
            const float* posData    = nullptr;
            const float* normalData = nullptr;
            const float* uvData     = nullptr;
            size_t vertexCount      = 0;

            {
                auto it = primitive.attributes.find("POSITION");
                if (it != primitive.attributes.end()) {
                    const auto& accessor   = gltfModel.accessors[it->second];
                    const auto& bufferView = gltfModel.bufferViews[accessor.bufferView];
                    const auto& buffer     = gltfModel.buffers[bufferView.buffer];
                    posData     = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                    vertexCount = accessor.count;
                }
            }
            {
                auto it = primitive.attributes.find("NORMAL");
                if (it != primitive.attributes.end()) {
                    const auto& accessor   = gltfModel.accessors[it->second];
                    const auto& bufferView = gltfModel.bufferViews[accessor.bufferView];
                    const auto& buffer     = gltfModel.buffers[bufferView.buffer];
                    normalData = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                }
            }
            {
                auto it = primitive.attributes.find("TEXCOORD_0");
                if (it != primitive.attributes.end()) {
                    const auto& accessor   = gltfModel.accessors[it->second];
                    const auto& bufferView = gltfModel.bufferViews[accessor.bufferView];
                    const auto& buffer     = gltfModel.buffers[bufferView.buffer];
                    uvData = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                }
            }

            meshData.vertices.resize(vertexCount);
            for (size_t i = 0; i < vertexCount; i++) {
                auto& v = meshData.vertices[i];
                v.position = posData    ? glm::vec3(posData[i * 3], posData[i * 3 + 1], posData[i * 3 + 2])
                                        : glm::vec3(0.0f);
                v.normal   = normalData ? glm::vec3(normalData[i * 3], normalData[i * 3 + 1], normalData[i * 3 + 2])
                                        : glm::vec3(0.0f, 1.0f, 0.0f);
                v.texCoord = uvData     ? glm::vec2(uvData[i * 2], uvData[i * 2 + 1])
                                        : glm::vec2(0.0f);
            }

            // Indices
            if (primitive.indices >= 0) {
                const auto& accessor   = gltfModel.accessors[primitive.indices];
                const auto& bufferView = gltfModel.bufferViews[accessor.bufferView];
                const auto& buffer     = gltfModel.buffers[bufferView.buffer];
                const uint8_t* base    = &buffer.data[bufferView.byteOffset + accessor.byteOffset];

                meshData.indices.resize(accessor.count);
                switch (accessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                        for (size_t i = 0; i < accessor.count; i++)
                            meshData.indices[i] = reinterpret_cast<const uint16_t*>(base)[i];
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                        std::memcpy(meshData.indices.data(), base, accessor.count * sizeof(uint32_t));
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                        for (size_t i = 0; i < accessor.count; i++)
                            meshData.indices[i] = base[i];
                        break;
                    default:
                        break;
                }
            }

            outModel.meshes.push_back(std::move(meshData));
        }
    }

    LOG_INFO("Loaded glTF: {} meshes, {} textures, {} materials",
             outModel.meshes.size(), outModel.textures.size(), outModel.materials.size());
    return true;
}

void ModelLoader::GenerateProceduralCube(ModelData& outModel) {
    // 6 faces, 4 vertices each, CCW winding
    const glm::vec3 positions[8] = {
        {-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f},
        { 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
        {-0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f,  0.5f},
        { 0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f},
    };

    struct FaceDef { int v[4]; glm::vec3 n; };
    const FaceDef faces[6] = {
        {{4, 5, 6, 7}, { 0,  0,  1}}, // +Z
        {{1, 0, 3, 2}, { 0,  0, -1}}, // -Z
        {{5, 1, 2, 6}, { 1,  0,  0}}, // +X
        {{0, 4, 7, 3}, {-1,  0,  0}}, // -X
        {{7, 6, 2, 3}, { 0,  1,  0}}, // +Y
        {{0, 1, 5, 4}, { 0, -1,  0}}, // -Y
    };
    const glm::vec2 uvs[4] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};

    MeshData mesh;
    mesh.materialIndex = 0;
    for (int f = 0; f < 6; f++) {
        uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
        for (int c = 0; c < 4; c++) {
            mesh.vertices.push_back({positions[faces[f].v[c]], faces[f].n, uvs[c]});
        }
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 3);
    }

    // Checkerboard texture 256x256
    constexpr uint32_t texW = 256, texH = 256, tileSize = 32;
    TextureData tex;
    tex.width  = texW;
    tex.height = texH;
    tex.pixels.resize(texW * texH * 4);

    for (uint32_t y = 0; y < texH; y++) {
        for (uint32_t x = 0; x < texW; x++) {
            bool white = ((x / tileSize) + (y / tileSize)) % 2 == 0;
            uint8_t c  = white ? 230 : 50;
            uint32_t idx = (y * texW + x) * 4;
            tex.pixels[idx + 0] = c;
            tex.pixels[idx + 1] = c;
            tex.pixels[idx + 2] = c;
            tex.pixels[idx + 3] = 255;
        }
    }

    MaterialData mat;
    mat.baseColorTextureIndex = 0;

    outModel.meshes.push_back(std::move(mesh));
    outModel.textures.push_back(std::move(tex));
    outModel.materials.push_back(mat);
}
