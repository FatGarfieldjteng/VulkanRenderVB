#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <vector>

class ShaderManager;

class Bloom {
public:
    static constexpr uint32_t MIP_COUNT = 6;

    void Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders,
                    uint32_t width, uint32_t height);
    void Shutdown(VkDevice device, VmaAllocator allocator);
    void Resize(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height);

    void Dispatch(VkCommandBuffer cmd, VkImageView hdrView, VkSampler hdrSampler,
                  uint32_t srcWidth, uint32_t srcHeight);

    VkImageView GetBloomView() const { return mMipViews.empty() ? VK_NULL_HANDLE : mMipViews[0]; }
    VkImage     GetBloomImage() const { return mBloomImage; }

private:
    void CreateImages(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h);
    void DestroyImages(VkDevice device, VmaAllocator allocator);
    void CreateDescriptors(VkDevice device);

    VkDevice mDevice = VK_NULL_HANDLE;

    VkDescriptorSetLayout mDownLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout mUpLayout   = VK_NULL_HANDLE;
    VkPipelineLayout      mDownPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout      mUpPipeLayout   = VK_NULL_HANDLE;
    VkPipeline            mDownPipeline = VK_NULL_HANDLE;
    VkPipeline            mUpPipeline   = VK_NULL_HANDLE;
    VkDescriptorPool      mDescPool = VK_NULL_HANDLE;

    std::vector<VkDescriptorSet> mDownSets;
    std::vector<VkDescriptorSet> mUpSets;

    VkImage       mBloomImage = VK_NULL_HANDLE;
    VmaAllocation mBloomAlloc = VK_NULL_HANDLE;
    VkImageView   mBloomFullView = VK_NULL_HANDLE;
    std::vector<VkImageView> mMipViews;

    VkSampler mLinearSampler = VK_NULL_HANDLE;

    uint32_t mWidth = 0, mHeight = 0;
};
