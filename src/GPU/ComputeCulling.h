#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include "Resource/VulkanBuffer.h"
#include <cstdint>

class ShaderManager;

struct CullParams {
    glm::mat4 viewProjection;
    glm::vec4 frustumPlanes[6];
    glm::vec2 hiZSize;
    float     nearPlane;
    float     farPlane;
    uint32_t  drawCount;
    uint32_t  occluderCount;
    uint32_t  candidateCount;
    uint32_t  _pad;
};

class ComputeCulling {
public:
    void Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders);
    void Shutdown(VkDevice device, VmaAllocator allocator);

    void UpdateBuffers(VmaAllocator allocator,
                       VkBuffer srcIndirectBuffer, uint32_t drawCount,
                       VkBuffer objectBuffer,
                       VkImageView hiZView, VkSampler hiZSampler);

    void DispatchFrustum(VkCommandBuffer cmd, const CullParams& params) const;
    void DispatchOcclusion(VkCommandBuffer cmd, const CullParams& params) const;

    VkBuffer GetOccluderIndirectBuffer()  const { return mOccluderIndirectBuffer.GetHandle(); }
    VkBuffer GetOccluderCountBuffer()     const { return mOccluderCountBuffer.GetHandle(); }
    VkBuffer GetCandidateIndirectBuffer() const { return mCandidateIndirectBuffer.GetHandle(); }
    VkBuffer GetCandidateCountBuffer()    const { return mCandidateCountBuffer.GetHandle(); }
    VkBuffer GetVisibleIndirectBuffer()   const { return mVisibleIndirectBuffer.GetHandle(); }
    VkBuffer GetVisibleCountBuffer()      const { return mVisibleCountBuffer.GetHandle(); }

private:
    VkDevice     mDevice     = VK_NULL_HANDLE;
    VmaAllocator mAllocator  = VK_NULL_HANDLE;

    VkPipeline            mFrustumPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      mFrustumPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout mFrustumDescSetLayout  = VK_NULL_HANDLE;
    VkDescriptorSet       mFrustumDescSet        = VK_NULL_HANDLE;

    VkPipeline            mOcclusionPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      mOcclusionPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout mOcclusionDescSetLayout  = VK_NULL_HANDLE;
    VkDescriptorSet       mOcclusionDescSet        = VK_NULL_HANDLE;

    VkDescriptorPool mDescPool = VK_NULL_HANDLE;

    VulkanBuffer mOccluderIndirectBuffer;
    VulkanBuffer mOccluderCountBuffer;
    VulkanBuffer mCandidateIndirectBuffer;
    VulkanBuffer mCandidateCountBuffer;
    VulkanBuffer mVisibleIndirectBuffer;
    VulkanBuffer mVisibleCountBuffer;
    VulkanBuffer mParamsUBO;

    uint32_t mMaxDrawCount = 0;
};
