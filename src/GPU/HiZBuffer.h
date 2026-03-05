#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <vector>
#include <cstdint>

class ShaderManager;

class HiZBuffer {
public:
    void Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders);
    void Shutdown(VkDevice device, VmaAllocator allocator);

    void Resize(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height);

    void BuildMipChain(VkCommandBuffer cmd) const;

    void SetSourceDepth(VkImageView depthView);

    VkImageView GetView()     const { return mHiZView; }
    VkImage     GetImage()    const { return mHiZImage; }
    VkSampler   GetSampler()  const { return mReduceSampler; }
    uint32_t    GetWidth()    const { return mWidth; }
    uint32_t    GetHeight()   const { return mHeight; }
    uint32_t    GetMipCount() const { return mMipCount; }

private:
    void CreateHiZImage(VkDevice device, VmaAllocator allocator);
    void CreateDescriptors(VkDevice device);
    void UpdateSourceDescriptor(VkDevice device);

    VkDevice      mDevice      = VK_NULL_HANDLE;
    VmaAllocator  mAllocator   = VK_NULL_HANDLE;

    VkImage       mHiZImage    = VK_NULL_HANDLE;
    VmaAllocation mHiZAlloc    = VK_NULL_HANDLE;
    VkImageView   mHiZView     = VK_NULL_HANDLE;
    std::vector<VkImageView> mMipViews;

    VkImageView   mSourceDepthView = VK_NULL_HANDLE;

    VkSampler     mReduceSampler = VK_NULL_HANDLE;
    VkPipeline    mPipeline      = VK_NULL_HANDLE;
    VkPipelineLayout mPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout mDescSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      mDescPool      = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> mDescSets;

    uint32_t mWidth    = 0;
    uint32_t mHeight   = 0;
    uint32_t mMipCount = 0;
};
