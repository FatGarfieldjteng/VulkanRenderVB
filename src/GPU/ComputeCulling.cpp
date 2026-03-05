#include "GPU/ComputeCulling.h"
#include "Resource/ShaderManager.h"
#include "Core/Logger.h"
#include "RHI/VulkanUtils.h"

#include <cstring>

static constexpr VkBufferUsageFlags kIndirectBufUsage =
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
static constexpr VkBufferUsageFlags kCountBufUsage =
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

void ComputeCulling::Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders) {
    mDevice    = device;
    mAllocator = allocator;

    // --- Frustum cull descriptor set layout (Set A) ---
    // 0: CullParams UBO, 1: srcIndirect, 2: objectSSBO,
    // 3: occluderIndirect, 4: occluderCount, 5: candidateIndirect, 6: candidateCount
    {
        VkDescriptorSetLayoutBinding bindings[7]{};
        for (uint32_t i = 0; i < 7; i++) {
            bindings[i].binding         = i;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 7;
        layoutInfo.pBindings    = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &mFrustumDescSetLayout));

        VkPipelineLayoutCreateInfo pipeLayoutInfo{};
        pipeLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeLayoutInfo.setLayoutCount = 1;
        pipeLayoutInfo.pSetLayouts    = &mFrustumDescSetLayout;
        VK_CHECK(vkCreatePipelineLayout(device, &pipeLayoutInfo, nullptr, &mFrustumPipelineLayout));

        VkShaderModule compModule = shaders.GetOrLoad("shaders/cull.comp.spv");
        VkComputePipelineCreateInfo compInfo{};
        compInfo.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        compInfo.stage.sType        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        compInfo.stage.stage        = VK_SHADER_STAGE_COMPUTE_BIT;
        compInfo.stage.module       = compModule;
        compInfo.stage.pName        = "main";
        compInfo.layout             = mFrustumPipelineLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compInfo, nullptr, &mFrustumPipeline));
    }

    // --- Occlusion test descriptor set layout (Set B) ---
    // 0: CullParams UBO, 1: candidateIndirect, 2: objectSSBO,
    // 3: visibleIndirect, 4: visibleCount, 5: Hi-Z sampler, 6: candidateCount
    {
        VkDescriptorSetLayoutBinding bindings[7]{};
        for (uint32_t i = 0; i < 7; i++) {
            bindings[i].binding         = i;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 7;
        layoutInfo.pBindings    = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &mOcclusionDescSetLayout));

        VkPipelineLayoutCreateInfo pipeLayoutInfo{};
        pipeLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeLayoutInfo.setLayoutCount = 1;
        pipeLayoutInfo.pSetLayouts    = &mOcclusionDescSetLayout;
        VK_CHECK(vkCreatePipelineLayout(device, &pipeLayoutInfo, nullptr, &mOcclusionPipelineLayout));

        VkShaderModule compModule = shaders.GetOrLoad("shaders/cull_occlusion.comp.spv");
        VkComputePipelineCreateInfo compInfo{};
        compInfo.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        compInfo.stage.sType        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        compInfo.stage.stage        = VK_SHADER_STAGE_COMPUTE_BIT;
        compInfo.stage.module       = compModule;
        compInfo.stage.pName        = "main";
        compInfo.layout             = mOcclusionPipelineLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compInfo, nullptr, &mOcclusionPipeline));
    }

    mParamsUBO.CreateHostVisible(allocator, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(CullParams));

    LOG_INFO("ComputeCulling initialized (two-pass)");
}

void ComputeCulling::Shutdown(VkDevice device, VmaAllocator allocator) {
    mOccluderIndirectBuffer.Destroy(allocator);
    mOccluderCountBuffer.Destroy(allocator);
    mCandidateIndirectBuffer.Destroy(allocator);
    mCandidateCountBuffer.Destroy(allocator);
    mVisibleIndirectBuffer.Destroy(allocator);
    mVisibleCountBuffer.Destroy(allocator);
    mParamsUBO.Destroy(allocator);

    if (mDescPool)                { vkDestroyDescriptorPool(device, mDescPool, nullptr);               mDescPool = VK_NULL_HANDLE; }
    if (mFrustumPipeline)         { vkDestroyPipeline(device, mFrustumPipeline, nullptr);              mFrustumPipeline = VK_NULL_HANDLE; }
    if (mFrustumPipelineLayout)   { vkDestroyPipelineLayout(device, mFrustumPipelineLayout, nullptr);  mFrustumPipelineLayout = VK_NULL_HANDLE; }
    if (mFrustumDescSetLayout)    { vkDestroyDescriptorSetLayout(device, mFrustumDescSetLayout, nullptr); mFrustumDescSetLayout = VK_NULL_HANDLE; }
    if (mOcclusionPipeline)       { vkDestroyPipeline(device, mOcclusionPipeline, nullptr);            mOcclusionPipeline = VK_NULL_HANDLE; }
    if (mOcclusionPipelineLayout) { vkDestroyPipelineLayout(device, mOcclusionPipelineLayout, nullptr); mOcclusionPipelineLayout = VK_NULL_HANDLE; }
    if (mOcclusionDescSetLayout)  { vkDestroyDescriptorSetLayout(device, mOcclusionDescSetLayout, nullptr); mOcclusionDescSetLayout = VK_NULL_HANDLE; }
}

void ComputeCulling::UpdateBuffers(VmaAllocator allocator,
                                   VkBuffer srcIndirectBuffer, uint32_t drawCount,
                                   VkBuffer objectBuffer,
                                   VkImageView hiZView, VkSampler hiZSampler)
{
    if (drawCount != mMaxDrawCount) {
        mOccluderIndirectBuffer.Destroy(allocator);
        mOccluderCountBuffer.Destroy(allocator);
        mCandidateIndirectBuffer.Destroy(allocator);
        mCandidateCountBuffer.Destroy(allocator);
        mVisibleIndirectBuffer.Destroy(allocator);
        mVisibleCountBuffer.Destroy(allocator);

        VkDeviceSize cmdSize = drawCount * sizeof(VkDrawIndexedIndirectCommand);
        mOccluderIndirectBuffer.CreateDeviceLocalEmpty(allocator, kIndirectBufUsage, cmdSize);
        mOccluderCountBuffer.CreateDeviceLocalEmpty(allocator, kCountBufUsage, sizeof(uint32_t));
        mCandidateIndirectBuffer.CreateDeviceLocalEmpty(allocator, kIndirectBufUsage, cmdSize);
        mCandidateCountBuffer.CreateDeviceLocalEmpty(allocator, kCountBufUsage, sizeof(uint32_t));
        mVisibleIndirectBuffer.CreateDeviceLocalEmpty(allocator, kIndirectBufUsage, cmdSize);
        mVisibleCountBuffer.CreateDeviceLocalEmpty(allocator, kCountBufUsage, sizeof(uint32_t));

        mMaxDrawCount = drawCount;
    }

    if (mDescPool) {
        vkDestroyDescriptorPool(mDevice, mDescPool, nullptr);
        mDescPool = VK_NULL_HANDLE;
    }

    VkDescriptorPoolSize poolSizes[3]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         poolSizes[0].descriptorCount = 2;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         poolSizes[1].descriptorCount = 12;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; poolSizes[2].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets       = 2;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes    = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &mDescPool));

    VkDescriptorSetLayout layouts[2] = { mFrustumDescSetLayout, mOcclusionDescSetLayout };
    VkDescriptorSet sets[2]{};
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = mDescPool;
    allocInfo.descriptorSetCount = 2;
    allocInfo.pSetLayouts        = layouts;
    VK_CHECK(vkAllocateDescriptorSets(mDevice, &allocInfo, sets));
    mFrustumDescSet  = sets[0];
    mOcclusionDescSet = sets[1];

    VkDescriptorBufferInfo paramsInfo   { mParamsUBO.GetHandle(), 0, sizeof(CullParams) };
    VkDescriptorBufferInfo srcIndInfo   { srcIndirectBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo objInfo      { objectBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo occIndInfo   { mOccluderIndirectBuffer.GetHandle(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo occCntInfo   { mOccluderCountBuffer.GetHandle(), 0, sizeof(uint32_t) };
    VkDescriptorBufferInfo candIndInfo  { mCandidateIndirectBuffer.GetHandle(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo candCntInfo  { mCandidateCountBuffer.GetHandle(), 0, sizeof(uint32_t) };
    VkDescriptorBufferInfo visIndInfo   { mVisibleIndirectBuffer.GetHandle(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo visCntInfo   { mVisibleCountBuffer.GetHandle(), 0, sizeof(uint32_t) };
    VkDescriptorImageInfo  hizInfo      { hiZSampler, hiZView, VK_IMAGE_LAYOUT_GENERAL };

    // --- Set A: frustum cull ---
    VkWriteDescriptorSet writesA[7]{};
    for (uint32_t i = 0; i < 7; i++) {
        writesA[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writesA[i].dstSet          = mFrustumDescSet;
        writesA[i].dstBinding      = i;
        writesA[i].descriptorCount = 1;
        writesA[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    writesA[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writesA[0].pBufferInfo = &paramsInfo;
    writesA[1].pBufferInfo = &srcIndInfo;
    writesA[2].pBufferInfo = &objInfo;
    writesA[3].pBufferInfo = &occIndInfo;
    writesA[4].pBufferInfo = &occCntInfo;
    writesA[5].pBufferInfo = &candIndInfo;
    writesA[6].pBufferInfo = &candCntInfo;
    vkUpdateDescriptorSets(mDevice, 7, writesA, 0, nullptr);

    // --- Set B: occlusion test ---
    VkWriteDescriptorSet writesB[7]{};
    for (uint32_t i = 0; i < 7; i++) {
        writesB[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writesB[i].dstSet          = mOcclusionDescSet;
        writesB[i].dstBinding      = i;
        writesB[i].descriptorCount = 1;
        writesB[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    writesB[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         writesB[0].pBufferInfo = &paramsInfo;
    writesB[1].pBufferInfo = &candIndInfo;
    writesB[2].pBufferInfo = &objInfo;
    writesB[3].pBufferInfo = &visIndInfo;
    writesB[4].pBufferInfo = &visCntInfo;
    writesB[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writesB[5].pImageInfo  = &hizInfo;
    writesB[6].pBufferInfo = &candCntInfo;
    vkUpdateDescriptorSets(mDevice, 7, writesB, 0, nullptr);
}

void ComputeCulling::DispatchFrustum(VkCommandBuffer cmd, const CullParams& params) const {
    if (mMaxDrawCount == 0) return;

    std::memcpy(mParamsUBO.GetMappedData(), &params, sizeof(CullParams));

    vkCmdFillBuffer(cmd, mOccluderCountBuffer.GetHandle(), 0, sizeof(uint32_t), 0);
    vkCmdFillBuffer(cmd, mCandidateCountBuffer.GetHandle(), 0, sizeof(uint32_t), 0);

    VkMemoryBarrier2 fillBarrier{};
    fillBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    fillBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    fillBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    fillBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    fillBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo dep{};
    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &fillBarrier;
    vkCmdPipelineBarrier2(cmd, &dep);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mFrustumPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            mFrustumPipelineLayout, 0, 1, &mFrustumDescSet, 0, nullptr);
    vkCmdDispatch(cmd, (params.drawCount + 63) / 64, 1, 1);

    VkMemoryBarrier2 computeBarrier{};
    computeBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    computeBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    computeBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

    VkDependencyInfo dep2{};
    dep2.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep2.memoryBarrierCount = 1;
    dep2.pMemoryBarriers    = &computeBarrier;
    vkCmdPipelineBarrier2(cmd, &dep2);
}

void ComputeCulling::DispatchOcclusion(VkCommandBuffer cmd, const CullParams& params) const {
    if (mMaxDrawCount == 0) return;

    std::memcpy(mParamsUBO.GetMappedData(), &params, sizeof(CullParams));

    vkCmdFillBuffer(cmd, mVisibleCountBuffer.GetHandle(), 0, sizeof(uint32_t), 0);

    VkMemoryBarrier2 fillBarrier{};
    fillBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    fillBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    fillBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    fillBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    fillBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo dep{};
    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &fillBarrier;
    vkCmdPipelineBarrier2(cmd, &dep);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mOcclusionPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            mOcclusionPipelineLayout, 0, 1, &mOcclusionDescSet, 0, nullptr);
    vkCmdDispatch(cmd, (mMaxDrawCount + 63) / 64, 1, 1);

    VkMemoryBarrier2 computeBarrier{};
    computeBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    computeBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    computeBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    computeBarrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo dep2{};
    dep2.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep2.memoryBarrierCount = 1;
    dep2.pMemoryBarriers    = &computeBarrier;
    vkCmdPipelineBarrier2(cmd, &dep2);
}
