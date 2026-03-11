#pragma once

#include "Resource/VulkanBuffer.h"
#include "GPU/MeshPool.h"
#include "Scene/ECS.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include <vector>
#include <cstdint>

class TransferManager;

struct BLASEntry {
    VkAccelerationStructureKHR handle    = VK_NULL_HANDLE;
    VulkanBuffer               buffer;
    VkDeviceSize               compactedSize = 0;
};

class AccelStructure {
public:
    void Initialize(VkDevice device, VmaAllocator allocator, const TransferManager& transfer);
    void Shutdown(VmaAllocator allocator);

    void BuildBLAS(const MeshPool& meshPool);
    void BuildTLAS(const Registry& registry, const MeshPool& meshPool);
    void UpdateTLAS(const Registry& registry, const MeshPool& meshPool);

    VkAccelerationStructureKHR GetTLAS() const { return mTLAS; }
    VkDeviceSize GetTotalBLASMemory() const { return mTotalBLASMemory; }
    VkDeviceSize GetTotalBLASMemoryPreCompaction() const { return mTotalBLASMemoryPreCompaction; }
    VkDeviceSize GetTLASMemory() const { return mTLASBuffer.GetSize(); }

private:
    void CompactBLAS();

    VkDevice     mDevice    = VK_NULL_HANDLE;
    VmaAllocator mAllocator = VK_NULL_HANDLE;
    const TransferManager* mTransfer = nullptr;

    std::vector<BLASEntry> mBLASEntries;
    VkDeviceSize mTotalBLASMemory = 0;
    VkDeviceSize mTotalBLASMemoryPreCompaction = 0;

    VkAccelerationStructureKHR mTLAS = VK_NULL_HANDLE;
    VulkanBuffer mTLASBuffer;
    VulkanBuffer mInstanceBuffer;
    bool         mTLASBuilt = false;
};
