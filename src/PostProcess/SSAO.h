#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

class ShaderManager;

class SSAO {
public:
    void Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders,
                    uint32_t width, uint32_t height);
    void Shutdown(VkDevice device, VmaAllocator allocator);
    void Resize(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height);

    void Dispatch(VkCommandBuffer cmd, VkImageView depthView,
                  const float* invProjection, const float* projInfo,
                  float radius, float bias, float intensity, float farPlane,
                  uint32_t width, uint32_t height);

    VkImageView GetAOView() const { return mAOView; }
    VkImage     GetAOImage() const { return mAOImage; }

private:
    void CreateImages(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h);
    void DestroyImages(VkDevice device, VmaAllocator allocator);
    void CreateDescriptors(VkDevice device);

    VkDevice mDevice = VK_NULL_HANDLE;

    VkDescriptorSetLayout mGTAOLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout mBlurLayout = VK_NULL_HANDLE;
    VkPipelineLayout      mGTAOPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout      mBlurPipeLayout = VK_NULL_HANDLE;
    VkPipeline            mGTAOPipeline = VK_NULL_HANDLE;
    VkPipeline            mBlurPipeline = VK_NULL_HANDLE;
    VkDescriptorPool      mDescPool = VK_NULL_HANDLE;
    VkDescriptorSet       mGTAOSet = VK_NULL_HANDLE;
    VkDescriptorSet       mBlurHSet = VK_NULL_HANDLE;
    VkDescriptorSet       mBlurVSet = VK_NULL_HANDLE;

    VkImage       mAOImage     = VK_NULL_HANDLE;
    VkImageView   mAOView      = VK_NULL_HANDLE;
    VmaAllocation mAOAlloc     = VK_NULL_HANDLE;
    VkImage       mAOTempImage = VK_NULL_HANDLE;
    VkImageView   mAOTempView  = VK_NULL_HANDLE;
    VmaAllocation mAOTempAlloc = VK_NULL_HANDLE;

    VkSampler mNearestSampler = VK_NULL_HANDLE;

    uint32_t mWidth = 0, mHeight = 0;
    VkImageLayout mAOLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout mAOTempLayout = VK_IMAGE_LAYOUT_UNDEFINED;
};
