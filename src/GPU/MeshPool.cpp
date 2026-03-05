#include "GPU/MeshPool.h"
#include "Resource/TransferManager.h"
#include "Core/Logger.h"

void MeshPool::Upload(VmaAllocator allocator, const TransferManager& transfer,
                      const std::vector<MeshData>& meshes)
{
    if (meshes.empty()) return;

    const uint32_t meshCount = static_cast<uint32_t>(meshes.size());

    VkDeviceSize totalVertexBytes = 0;
    VkDeviceSize totalIndexBytes  = 0;
    for (const auto& m : meshes) {
        totalVertexBytes += m.vertices.size() * sizeof(MeshVertex);
        totalIndexBytes  += m.indices.size()  * sizeof(uint32_t);
    }

    std::vector<MeshVertex> allVertices;
    std::vector<uint32_t>   allIndices;
    allVertices.reserve(totalVertexBytes / sizeof(MeshVertex));
    allIndices.reserve(totalIndexBytes / sizeof(uint32_t));

    uint32_t vertexOffset = 0;
    uint32_t firstIndex   = 0;

    for (uint32_t i = 0; i < meshCount; i++) {
        const auto& m = meshes[i];

        AABB bounds;
        for (const auto& v : m.vertices)
            bounds.Include(v.position);

        MeshDrawCommand cmd{};
        cmd.indexCount    = static_cast<uint32_t>(m.indices.size());
        cmd.instanceCount = 1;
        cmd.firstIndex    = firstIndex;
        cmd.vertexOffset  = static_cast<int32_t>(vertexOffset);
        cmd.firstInstance = 0;
        cmd.materialIndex = (m.materialIndex >= 0) ? static_cast<uint32_t>(m.materialIndex) : 0;
        cmd.bounds        = bounds;
        mDrawCommands.push_back(cmd);

        allVertices.insert(allVertices.end(), m.vertices.begin(), m.vertices.end());
        allIndices.insert(allIndices.end(), m.indices.begin(), m.indices.end());

        vertexOffset += static_cast<uint32_t>(m.vertices.size());
        firstIndex   += static_cast<uint32_t>(m.indices.size());
    }

    mVertexBuffer.CreateDeviceLocal(allocator, transfer,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        allVertices.data(), allVertices.size() * sizeof(MeshVertex));

    mIndexBuffer.CreateDeviceLocal(allocator, transfer,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        allIndices.data(), allIndices.size() * sizeof(uint32_t));

    LOG_INFO("MeshPool uploaded: {} meshes, {} vertices ({} KB), {} indices ({} KB)",
             meshes.size(), allVertices.size(),
             (allVertices.size() * sizeof(MeshVertex)) / 1024,
             allIndices.size(),
             (allIndices.size() * sizeof(uint32_t)) / 1024);
}

void MeshPool::Destroy(VmaAllocator allocator) {
    mVertexBuffer.Destroy(allocator);
    mIndexBuffer.Destroy(allocator);
    mDrawCommands.clear();
}
