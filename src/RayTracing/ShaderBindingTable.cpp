#include "RayTracing/ShaderBindingTable.h"
#include "RayTracing/RTPipeline.h"
#include "Resource/TransferManager.h"
#include "Core/Logger.h"

#include <cstring>
#include <algorithm>

void ShaderBindingTable::Build(VkDevice device, VmaAllocator allocator,
                                const RTPipeline& pipeline,
                                const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& props,
                                uint32_t numRayGenGroups,
                                uint32_t numMissGroups,
                                uint32_t numHitGroups,
                                uint32_t numCallableGroups,
                                uint32_t hitRecordExtraBytes) {
    mHandleSize      = props.shaderGroupHandleSize;
    mHandleAlignment = props.shaderGroupHandleAlignment;
    mBaseAlignment   = props.shaderGroupBaseAlignment;
    mNumHitGroups    = numHitGroups;

    VkDeviceSize rayGenRecordSize = AlignUp(mHandleSize, mHandleAlignment);
    VkDeviceSize missRecordSize   = AlignUp(mHandleSize, mHandleAlignment);
    VkDeviceSize hitRecordSize    = AlignUp(mHandleSize + hitRecordExtraBytes, mHandleAlignment);
    VkDeviceSize callRecordSize   = numCallableGroups > 0
                                    ? AlignUp(mHandleSize, mHandleAlignment)
                                    : 0;

    mRayGenStride = rayGenRecordSize;
    // Vulkan: raygen SBT size must equal stride (VUID-vkCmdTraceRaysKHR-size-04023)
    mRayGenSize   = (numRayGenGroups == 1) ? mRayGenStride : AlignUp(rayGenRecordSize * numRayGenGroups, mBaseAlignment);

    mMissStride = missRecordSize;
    mMissSize   = AlignUp(missRecordSize * numMissGroups, mBaseAlignment);

    mHitStride = hitRecordSize;
    mHitSize   = AlignUp(hitRecordSize * numHitGroups, mBaseAlignment);

    mCallStride = callRecordSize;
    mCallSize   = numCallableGroups > 0
                  ? AlignUp(callRecordSize * numCallableGroups, mBaseAlignment)
                  : 0;

    mRayGenOffset = 0;
    // Miss/hit regions must start at address aligned to shaderGroupBaseAlignment (64)
    mMissOffset   = AlignUp(mRayGenSize, mBaseAlignment);
    mHitOffset    = mMissOffset + mMissSize;
    mCallOffset   = mHitOffset + mHitSize;

    VkDeviceSize totalSize = mCallOffset + mCallSize;

    mCpuData.resize(totalSize, 0);

    auto handles = pipeline.GetShaderGroupHandles(device, static_cast<uint32_t>(mHandleSize));

    uint32_t groupIdx = 0;
    for (uint32_t i = 0; i < numRayGenGroups; i++) {
        std::memcpy(mCpuData.data() + mRayGenOffset + i * mRayGenStride,
                    handles.data() + groupIdx * mHandleSize, mHandleSize);
        groupIdx++;
    }
    for (uint32_t i = 0; i < numMissGroups; i++) {
        std::memcpy(mCpuData.data() + mMissOffset + i * mMissStride,
                    handles.data() + groupIdx * mHandleSize, mHandleSize);
        groupIdx++;
    }
    for (uint32_t i = 0; i < numHitGroups; i++) {
        std::memcpy(mCpuData.data() + mHitOffset + i * mHitStride,
                    handles.data() + groupIdx * mHandleSize, mHandleSize);
        groupIdx++;
    }
    for (uint32_t i = 0; i < numCallableGroups; i++) {
        std::memcpy(mCpuData.data() + mCallOffset + i * mCallStride,
                    handles.data() + groupIdx * mHandleSize, mHandleSize);
        groupIdx++;
    }

    LOG_INFO("SBT built: raygen={} miss={} hit={} callable={}, total {} bytes",
             numRayGenGroups, numMissGroups, numHitGroups, numCallableGroups, totalSize);
}

void ShaderBindingTable::WriteHitGroupData(uint32_t hitGroupIndex,
                                            const void* data, uint32_t dataSize) {
    if (hitGroupIndex >= mNumHitGroups) return;
    VkDeviceSize offset = mHitOffset + hitGroupIndex * mHitStride + mHandleSize;
    if (offset + dataSize > mCpuData.size()) return;
    std::memcpy(mCpuData.data() + offset, data, dataSize);
}

void ShaderBindingTable::UploadToGPU(VmaAllocator allocator, VkDevice device,
                                      const TransferManager& transfer) {
    mBuffer.Destroy(allocator);
    mBuffer.CreateDeviceLocal(allocator, transfer,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        mCpuData.data(), mCpuData.size());
    mBaseAddress = mBuffer.GetDeviceAddress(device);
}

void ShaderBindingTable::Destroy(VmaAllocator allocator) {
    mBuffer.Destroy(allocator);
    mCpuData.clear();
    mBaseAddress   = 0;
    mNumHitGroups  = 0;
}

VkStridedDeviceAddressRegionKHR ShaderBindingTable::GetRayGenRegion() const {
    return {mBaseAddress + mRayGenOffset, mRayGenStride, mRayGenSize};
}

VkStridedDeviceAddressRegionKHR ShaderBindingTable::GetMissRegion() const {
    return {mBaseAddress + mMissOffset, mMissStride, mMissSize};
}

VkStridedDeviceAddressRegionKHR ShaderBindingTable::GetHitRegion() const {
    return {mBaseAddress + mHitOffset, mHitStride, mHitSize};
}

VkStridedDeviceAddressRegionKHR ShaderBindingTable::GetCallableRegion() const {
    if (mCallSize == 0) return {0, 0, 0};
    return {mBaseAddress + mCallOffset, mCallStride, mCallSize};
}
