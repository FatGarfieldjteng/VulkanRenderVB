#include "Asset/ModelLoader.h"
#include "Core/Logger.h"

#include <tiny_gltf.h>
#include <algorithm>
#include <cstring>
#include <cmath>

static int ResolveTextureSource(const tinygltf::Model& model, int texIndex) {
    if (texIndex >= 0 && texIndex < static_cast<int>(model.textures.size()))
        return model.textures[texIndex].source;
    return -1;
}

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

    for (const auto& mat : gltfModel.materials) {
        MaterialData material;
        const auto& pbr = mat.pbrMetallicRoughness;

        material.baseColorFactor = glm::vec4(
            static_cast<float>(pbr.baseColorFactor[0]),
            static_cast<float>(pbr.baseColorFactor[1]),
            static_cast<float>(pbr.baseColorFactor[2]),
            static_cast<float>(pbr.baseColorFactor[3]));
        material.metallicFactor  = static_cast<float>(pbr.metallicFactor);
        material.roughnessFactor = static_cast<float>(pbr.roughnessFactor);

        material.baseColorTextureIndex        = ResolveTextureSource(gltfModel, pbr.baseColorTexture.index);
        material.metallicRoughnessTextureIndex = ResolveTextureSource(gltfModel, pbr.metallicRoughnessTexture.index);
        material.normalTextureIndex           = ResolveTextureSource(gltfModel, mat.normalTexture.index);
        material.occlusionTextureIndex        = ResolveTextureSource(gltfModel, mat.occlusionTexture.index);
        material.emissiveTextureIndex         = ResolveTextureSource(gltfModel, mat.emissiveTexture.index);

        material.emissiveFactor = glm::vec3(
            static_cast<float>(mat.emissiveFactor[0]),
            static_cast<float>(mat.emissiveFactor[1]),
            static_cast<float>(mat.emissiveFactor[2]));

        outModel.materials.push_back(material);
    }

    for (const auto& mesh : gltfModel.meshes) {
        for (const auto& primitive : mesh.primitives) {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES && primitive.mode != -1)
                continue;

            MeshData meshData;
            meshData.materialIndex = primitive.material;

            const float* posData     = nullptr;
            const float* normalData  = nullptr;
            const float* uvData      = nullptr;
            const float* tangentData = nullptr;
            size_t vertexCount       = 0;

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
            {
                auto it = primitive.attributes.find("TANGENT");
                if (it != primitive.attributes.end()) {
                    const auto& accessor   = gltfModel.accessors[it->second];
                    const auto& bufferView = gltfModel.bufferViews[accessor.bufferView];
                    const auto& buffer     = gltfModel.buffers[bufferView.buffer];
                    tangentData = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
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
                v.tangent  = tangentData ? glm::vec4(tangentData[i * 4], tangentData[i * 4 + 1],
                                                     tangentData[i * 4 + 2], tangentData[i * 4 + 3])
                                         : glm::vec4(0.0f);
            }

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

            if (!tangentData)
                ComputeTangents(meshData);

            outModel.meshes.push_back(std::move(meshData));
        }
    }

    LOG_INFO("Loaded glTF: {} meshes, {} textures, {} materials",
             outModel.meshes.size(), outModel.textures.size(), outModel.materials.size());
    return true;
}

void ModelLoader::ComputeTangents(MeshData& mesh) {
    auto& verts   = mesh.vertices;
    auto& indices = mesh.indices;
    if (verts.empty() || indices.empty()) return;

    std::vector<glm::vec3> tan(verts.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> bitan(verts.size(), glm::vec3(0.0f));

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        uint32_t i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];
        const auto& v0 = verts[i0];
        const auto& v1 = verts[i1];
        const auto& v2 = verts[i2];

        glm::vec3 edge1  = v1.position - v0.position;
        glm::vec3 edge2  = v2.position - v0.position;
        glm::vec2 duv1   = v1.texCoord - v0.texCoord;
        glm::vec2 duv2   = v2.texCoord - v0.texCoord;

        float denom = duv1.x * duv2.y - duv2.x * duv1.y;
        float f = std::abs(denom) > 1e-8f ? 1.0f / denom : 0.0f;

        glm::vec3 t = f * (duv2.y * edge1 - duv1.y * edge2);
        glm::vec3 b = f * (-duv2.x * edge1 + duv1.x * edge2);

        tan[i0] += t;  tan[i1] += t;  tan[i2] += t;
        bitan[i0] += b; bitan[i1] += b; bitan[i2] += b;
    }

    for (size_t i = 0; i < verts.size(); i++) {
        glm::vec3 n = verts[i].normal;
        glm::vec3 t = tan[i];
        float tLen = glm::length(t);
        if (tLen < 1e-8f) {
            verts[i].tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            continue;
        }
        t = glm::normalize(t - n * glm::dot(n, t));
        float w = (glm::dot(glm::cross(n, t), bitan[i]) < 0.0f) ? -1.0f : 1.0f;
        verts[i].tangent = glm::vec4(t, w);
    }
}

void ModelLoader::GenerateProceduralCube(ModelData& outModel) {
    const glm::vec3 positions[8] = {
        {-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f},
        { 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
        {-0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f,  0.5f},
        { 0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f},
    };

    struct FaceDef { int v[4]; glm::vec3 n; };
    const FaceDef faces[6] = {
        {{4, 5, 6, 7}, { 0,  0,  1}},
        {{1, 0, 3, 2}, { 0,  0, -1}},
        {{5, 1, 2, 6}, { 1,  0,  0}},
        {{0, 4, 7, 3}, {-1,  0,  0}},
        {{7, 6, 2, 3}, { 0,  1,  0}},
        {{0, 1, 5, 4}, { 0, -1,  0}},
    };
    const glm::vec2 uvs[4] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};

    MeshData mesh;
    mesh.materialIndex = 0;
    for (int f = 0; f < 6; f++) {
        uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
        for (int c = 0; c < 4; c++) {
            MeshVertex v{};
            v.position = positions[faces[f].v[c]];
            v.normal   = faces[f].n;
            v.texCoord = uvs[c];
            mesh.vertices.push_back(v);
        }
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 3);
    }

    ComputeTangents(mesh);

    constexpr uint32_t texW = 256, texH = 256, tileSize = 32;
    TextureData tex;
    tex.width  = texW;
    tex.height = texH;
    tex.pixels.resize(texW * texH * 4);

    for (uint32_t y = 0; y < texH; y++) {
        for (uint32_t x = 0; x < texW; x++) {
            bool white = ((x / tileSize) + (y / tileSize)) % 2 == 0;
            uint8_t cv  = white ? 230 : 50;
            uint32_t idx = (y * texW + x) * 4;
            tex.pixels[idx + 0] = cv;
            tex.pixels[idx + 1] = cv;
            tex.pixels[idx + 2] = cv;
            tex.pixels[idx + 3] = 255;
        }
    }

    MaterialData mat;
    mat.baseColorTextureIndex = 0;
    mat.metallicFactor  = 0.0f;
    mat.roughnessFactor = 0.5f;

    outModel.meshes.push_back(std::move(mesh));
    outModel.textures.push_back(std::move(tex));
    outModel.materials.push_back(mat);
}

void ModelLoader::GenerateGroundPlane(MeshData& outMesh, float halfSize) {
    float uvScale = halfSize;
    outMesh.vertices = {
        {{-halfSize, 0, -halfSize}, {0, 1, 0}, {0,       0},       {1, 0, 0, 1}},
        {{ halfSize, 0, -halfSize}, {0, 1, 0}, {uvScale, 0},       {1, 0, 0, 1}},
        {{ halfSize, 0,  halfSize}, {0, 1, 0}, {uvScale, uvScale}, {1, 0, 0, 1}},
        {{-halfSize, 0,  halfSize}, {0, 1, 0}, {0,       uvScale}, {1, 0, 0, 1}},
    };
    outMesh.indices = { 0, 2, 1, 0, 3, 2 };
}
