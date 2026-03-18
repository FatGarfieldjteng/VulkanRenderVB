#pragma once

#include "Resource/VulkanBuffer.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <vector>
#include <cstdint>

class RTPipeline;

class ShaderBindingTable {
public:
    void Build(VkDevice device, VmaAllocator allocator,
               const RTPipeline& pipeline,
               const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& props,
               uint32_t numRayGenGroups,
               uint32_t numMissGroups,
               uint32_t numHitGroups,
               uint32_t numCallableGroups = 0,
               uint32_t hitRecordExtraBytes = 0);

    void WriteHitGroupData(uint32_t hitGroupIndex, const void* data, uint32_t dataSize);
    void UploadToGPU(VmaAllocator allocator, VkDevice device,
                     const class TransferManager& transfer);

    void Destroy(VmaAllocator allocator);

    VkStridedDeviceAddressRegionKHR GetRayGenRegion()  const;
    VkStridedDeviceAddressRegionKHR GetMissRegion()    const;
    VkStridedDeviceAddressRegionKHR GetHitRegion()     const;
    VkStridedDeviceAddressRegionKHR GetCallableRegion() const;

private:
    static VkDeviceSize AlignUp(VkDeviceSize val, VkDeviceSize align) {
        return (val + align - 1) & ~(align - 1);
    }

    VulkanBuffer    mBuffer;
    VkDeviceAddress mBaseAddress = 0;

    VkDeviceSize mHandleSize      = 0;
    VkDeviceSize mHandleAlignment = 0;
    VkDeviceSize mBaseAlignment   = 0;

    VkDeviceSize mRayGenOffset = 0, mRayGenStride = 0, mRayGenSize = 0;
    VkDeviceSize mMissOffset   = 0, mMissStride   = 0, mMissSize   = 0;
    VkDeviceSize mHitOffset    = 0, mHitStride    = 0, mHitSize    = 0;
    VkDeviceSize mCallOffset   = 0, mCallStride   = 0, mCallSize   = 0;

    std::vector<uint8_t> mCpuData;
    uint32_t mNumHitGroups = 0;
};
