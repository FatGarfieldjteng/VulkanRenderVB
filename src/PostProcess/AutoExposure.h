#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

class ShaderManager;

class AutoExposure {
public:
    void Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders);
    void Shutdown(VkDevice device, VmaAllocator allocator);

    void Dispatch(VkCommandBuffer cmd, VkImageView hdrView, VkSampler hdrSampler,
                  uint32_t width, uint32_t height,
                  float minLogLum, float maxLogLum,
                  float deltaTime, float adaptSpeed);

    VkBuffer GetExposureBuffer() const { return mExposureBuffer; }

private:
    VkDevice mDevice = VK_NULL_HANDLE;

    VkDescriptorSetLayout mHistogramLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout mAverageLayout   = VK_NULL_HANDLE;
    VkPipelineLayout      mHistogramPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout      mAveragePipeLayout   = VK_NULL_HANDLE;
    VkPipeline            mHistogramPipeline = VK_NULL_HANDLE;
    VkPipeline            mAveragePipeline   = VK_NULL_HANDLE;
    VkDescriptorPool      mDescPool = VK_NULL_HANDLE;
    VkDescriptorSet       mHistogramSet = VK_NULL_HANDLE;
    VkDescriptorSet       mAverageSet   = VK_NULL_HANDLE;

    VkBuffer      mHistogramBuffer = VK_NULL_HANDLE;
    VmaAllocation mHistogramAlloc  = VK_NULL_HANDLE;
    VkBuffer      mExposureBuffer  = VK_NULL_HANDLE;
    VmaAllocation mExposureAlloc   = VK_NULL_HANDLE;

    VkSampler mLinearSampler = VK_NULL_HANDLE;
};
