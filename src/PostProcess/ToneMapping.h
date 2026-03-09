#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

class ShaderManager;

struct ToneMappingPushConstants {
    uint32_t curveType;
    float    exposureBias;
    float    whitePoint;
    float    shoulderStrength;
    float    linearStrength;
    float    linearAngle;
    float    toeStrength;
    float    saturation;
    float    agxPunch;
    float    bloomStrength;
    uint32_t useAutoExposure;
};

class ToneMapping {
public:
    void Initialize(VkDevice device, ShaderManager& shaders, VkFormat outputFormat);
    void Shutdown(VkDevice device);

    void Draw(VkCommandBuffer cmd, VkImageView outputView, VkExtent2D extent,
              VkImageView hdrView, VkImageView bloomView, VkImageView aoView,
              VkBuffer exposureBuffer,
              const ToneMappingPushConstants& params);

    VkDescriptorSetLayout GetDescLayout() const { return mDescLayout; }

private:
    VkDevice mDevice = VK_NULL_HANDLE;

    VkDescriptorSetLayout mDescLayout     = VK_NULL_HANDLE;
    VkPipelineLayout      mPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            mPipeline       = VK_NULL_HANDLE;
    VkDescriptorPool      mDescPool       = VK_NULL_HANDLE;
    VkDescriptorSet       mDescSet        = VK_NULL_HANDLE;

    VkSampler mLinearSampler = VK_NULL_HANDLE;
};
