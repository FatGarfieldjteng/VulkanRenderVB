#pragma once

#include "Resource/VulkanBuffer.h"
#include "Resource/VulkanImage.h"
#include "Resource/ShaderManager.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

class RTShadows {
public:
    void Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders,
                    uint32_t width, uint32_t height);
    void Shutdown(VkDevice device, VmaAllocator allocator);
    void Resize(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height);

    void Dispatch(VkCommandBuffer cmd, VkAccelerationStructureKHR tlas,
                  VkImageView depthView, VkSampler depthSampler,
                  const glm::mat4& invViewProj,
                  const glm::vec3& lightDir, float lightRadius,
                  const glm::vec3& cameraPos);

    void Denoise(VkCommandBuffer cmd, VkImageView depthView, VkSampler depthSampler,
                 const glm::mat4& invViewProj);

    VkImageView GetOutputView() const { return mShadowImage[mOutputIdx].GetView(); }
    VkImage     GetOutputImage() const { return mShadowImage[mOutputIdx].GetImage(); }

    bool IsEnabled() const { return mEnabled; }
    void SetEnabled(bool e) { mEnabled = e; }

private:
    void CreateDescriptors();
    void UpdateDescriptors(VkAccelerationStructureKHR tlas,
                           VkImageView depthView, VkSampler depthSampler);
    void UpdateDenoiseDescriptors(VkImageView depthView, VkSampler depthSampler);

    VkDevice     mDevice    = VK_NULL_HANDLE;
    VmaAllocator mAllocator = VK_NULL_HANDLE;
    uint32_t     mWidth = 0, mHeight = 0;
    bool         mEnabled = true;
    bool         mDescriptorsDirty = true;

    VulkanImage  mShadowImage[2];   // ping-pong for denoise
    VkSampler    mSampler = VK_NULL_HANDLE;
    int          mOutputIdx = 1;  // denoise with 3 iters always ends at image[1]

    // Shadow trace pass
    VkDescriptorSetLayout mTraceDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool      mTraceDescPool   = VK_NULL_HANDLE;
    VkDescriptorSet       mTraceDescSet    = VK_NULL_HANDLE;
    VkPipelineLayout      mTracePipeLayout = VK_NULL_HANDLE;
    VkPipeline            mTracePipeline   = VK_NULL_HANDLE;

    // Denoise pass
    VkDescriptorSetLayout mDenoiseDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool      mDenoiseDescPool   = VK_NULL_HANDLE;
    VkDescriptorSet       mDenoiseDescSets[3] = {};  // 3 A-Trous iterations
    VkPipelineLayout      mDenoisePipeLayout = VK_NULL_HANDLE;
    VkPipeline            mDenoisePipeline   = VK_NULL_HANDLE;

    struct TracePushConstants {
        glm::mat4 invViewProj;
        glm::vec4 lightDir;
        glm::vec4 cameraPos;
        glm::uvec2 resolution;
    };

    struct DenoisePushConstants {
        glm::mat4  invViewProj;
        glm::uvec2 resolution;
        int32_t  stepSize;
        float    depthSigma;
        float    normalSigma;
    };
};
