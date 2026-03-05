#include "GPU/IndirectRenderer.h"
#include "Resource/TransferManager.h"
#include "Core/Logger.h"

#include <cmath>

void IndirectRenderer::Initialize(VmaAllocator, VkDevice device) {
    mDevice = device;
}

void IndirectRenderer::Shutdown(VmaAllocator allocator) {
    mIndirectBuffer.Destroy(allocator);
    mObjectSSBO.Destroy(allocator);
    mDrawCountBuffer.Destroy(allocator);
    mDrawCount = 0;
}

void IndirectRenderer::BuildCommands(VmaAllocator allocator, const TransferManager& transfer,
                                     const MeshPool& meshPool, const Registry& registry,
                                     float occluderRatio)
{
    const auto& meshDrawCmds = meshPool.GetDrawCommands();

    std::vector<VkDrawIndexedIndirectCommand> indirectCmds;
    std::vector<GPUObjectData> objectData;

    registry.ForEachRenderable([&](Entity, const TransformComponent& tc,
                                   const MeshComponent& mc, const MaterialComponent& matc) {
        if (mc.meshIndex < 0 || mc.meshIndex >= static_cast<int>(meshDrawCmds.size())) return;

        const auto& poolCmd = meshDrawCmds[mc.meshIndex];

        VkDrawIndexedIndirectCommand cmd{};
        cmd.indexCount    = poolCmd.indexCount;
        cmd.instanceCount = 1;
        cmd.firstIndex    = poolCmd.firstIndex;
        cmd.vertexOffset  = poolCmd.vertexOffset;
        cmd.firstInstance = static_cast<uint32_t>(indirectCmds.size());
        indirectCmds.push_back(cmd);

        GPUObjectData obj{};
        obj.model         = tc.worldMatrix;
        obj.aabbMin       = glm::vec4(poolCmd.bounds.min, 0.0f);
        obj.aabbMax       = glm::vec4(poolCmd.bounds.max, 0.0f);
        obj.materialIndex = (matc.materialIndex >= 0) ? static_cast<uint32_t>(matc.materialIndex) : poolCmd.materialIndex;
        objectData.push_back(obj);
    });

    mDrawCount = static_cast<uint32_t>(indirectCmds.size());

    mOccluderCount = std::min(
        static_cast<uint32_t>(std::ceil(mDrawCount * occluderRatio)),
        mDrawCount);

    if (mDrawCount == 0) return;

    mIndirectBuffer.Destroy(allocator);
    mObjectSSBO.Destroy(allocator);
    mDrawCountBuffer.Destroy(allocator);

    mIndirectBuffer.CreateDeviceLocal(allocator, transfer,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        indirectCmds.data(), indirectCmds.size() * sizeof(VkDrawIndexedIndirectCommand));

    mObjectSSBO.CreateDeviceLocal(allocator, transfer,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        objectData.data(), objectData.size() * sizeof(GPUObjectData));

    uint32_t countData = mDrawCount;
    mDrawCountBuffer.CreateDeviceLocal(allocator, transfer,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        &countData, sizeof(uint32_t));
}
