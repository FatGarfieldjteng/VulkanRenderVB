#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

class ShaderManager;

struct ColorGradingPushConstants {
    float    lutStrength;
    float    vignetteIntensity;
    float    vignetteRadius;
    float    grainStrength;
    float    grainTime;
    float    chromaticAberration;
    uint32_t resolutionX;
    uint32_t resolutionY;
};

class ColorGrading {
public:
    void Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders,
                    VkFormat outputFormat);
    void Shutdown(VkDevice device, VmaAllocator allocator);

    void Draw(VkCommandBuffer cmd, VkImageView outputView, VkExtent2D extent,
              VkImageView inputView, const ColorGradingPushConstants& params);

private:
    void CreateDefaultLUT(VkDevice device, VmaAllocator allocator);

    VkDevice mDevice = VK_NULL_HANDLE;
    bool mLUTLayoutReady = false;

    VkDescriptorSetLayout mDescLayout     = VK_NULL_HANDLE;
    VkPipelineLayout      mPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            mPipeline       = VK_NULL_HANDLE;
    VkDescriptorPool      mDescPool       = VK_NULL_HANDLE;
    VkDescriptorSet       mDescSet        = VK_NULL_HANDLE;

    VkImage       mLUTImage  = VK_NULL_HANDLE;
    VkImageView   mLUTView   = VK_NULL_HANDLE;
    VmaAllocation mLUTAlloc  = VK_NULL_HANDLE;
    VkSampler     mLinearSampler = VK_NULL_HANDLE;
};
