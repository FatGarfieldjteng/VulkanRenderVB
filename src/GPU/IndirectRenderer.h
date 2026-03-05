#pragma once

#include "GPU/MeshPool.h"
#include "Resource/VulkanBuffer.h"
#include "Scene/ECS.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include <vector>
#include <cstdint>

struct GPUObjectData {
    glm::mat4 model;
    glm::vec4 aabbMin;       // xyz = local-space AABB min, w unused
    glm::vec4 aabbMax;       // xyz = local-space AABB max, w unused
    uint32_t  materialIndex;
    uint32_t  _pad[3];
};
static_assert(sizeof(GPUObjectData) == 112, "GPUObjectData must be 112 bytes for std430");

class IndirectRenderer {
public:
    void Initialize(VmaAllocator allocator, VkDevice device);
    void Shutdown(VmaAllocator allocator);

    void BuildCommands(VmaAllocator allocator, const TransferManager& transfer,
                       const MeshPool& meshPool, const Registry& registry,
                       float occluderRatio);

    VkBuffer GetIndirectBuffer() const { return mIndirectBuffer.GetHandle(); }
    VkBuffer GetObjectBuffer()   const { return mObjectSSBO.GetHandle(); }
    VkBuffer GetCountBuffer()    const { return mDrawCountBuffer.GetHandle(); }
    uint32_t GetDrawCount()      const { return mDrawCount; }
    uint32_t GetOccluderCount() const { return mOccluderCount; }

private:
    VulkanBuffer mIndirectBuffer;
    VulkanBuffer mObjectSSBO;
    VulkanBuffer mDrawCountBuffer;
    uint32_t     mDrawCount = 0;
    uint32_t     mOccluderCount = 0;
    VkDevice     mDevice    = VK_NULL_HANDLE;
};
