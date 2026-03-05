#pragma once

#include "Resource/VulkanBuffer.h"
#include "Asset/ModelLoader.h"
#include "Math/AABB.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include <vector>
#include <cstdint>

struct MeshDrawCommand {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t  vertexOffset;
    uint32_t firstInstance;
    uint32_t materialIndex;
    AABB     bounds;
};

class TransferManager;

class MeshPool {
public:
    void Upload(VmaAllocator allocator, const TransferManager& transfer,
                const std::vector<MeshData>& meshes);
    void Destroy(VmaAllocator allocator);

    VkBuffer GetVertexBuffer() const { return mVertexBuffer.GetHandle(); }
    VkBuffer GetIndexBuffer()  const { return mIndexBuffer.GetHandle(); }

    const std::vector<MeshDrawCommand>& GetDrawCommands() const { return mDrawCommands; }
    uint32_t GetMeshCount() const { return static_cast<uint32_t>(mDrawCommands.size()); }

private:
    VulkanBuffer mVertexBuffer;
    VulkanBuffer mIndexBuffer;
    std::vector<MeshDrawCommand> mDrawCommands;
};
