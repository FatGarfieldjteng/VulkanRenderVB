#include "GPU/HiZBuffer.h"
#include "Resource/ShaderManager.h"
#include "Core/Logger.h"
#include "RHI/VulkanUtils.h"

#include <algorithm>
#include <cmath>

void HiZBuffer::Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders) {
    mDevice    = device;
    mAllocator = allocator;

    VkSamplerReductionModeCreateInfo reductionInfo{};
    reductionInfo.sType         = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO;
    reductionInfo.reductionMode = VK_SAMPLER_REDUCTION_MODE_MAX;

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.pNext     = &reductionInfo;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.minLod        = 0.0f;
    samplerInfo.maxLod        = VK_LOD_CLAMP_NONE;

    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &mReduceSampler));

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings    = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &mDescSetLayout));

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount = 1;
    pipeLayoutInfo.pSetLayouts    = &mDescSetLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &pipeLayoutInfo, nullptr, &mPipelineLayout));

    VkShaderModule compModule = shaders.GetOrLoad("shaders/hiz_reduce.comp.spv");
    VkComputePipelineCreateInfo compInfo{};
    compInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.module = compModule;
    compInfo.stage.pName  = "main";
    compInfo.layout       = mPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compInfo, nullptr, &mPipeline));

    LOG_INFO("HiZBuffer initialized");
}

void HiZBuffer::Shutdown(VkDevice device, VmaAllocator allocator) {
    for (auto view : mMipViews)
        if (view) vkDestroyImageView(device, view, nullptr);
    mMipViews.clear();

    if (mHiZView)  { vkDestroyImageView(device, mHiZView, nullptr); mHiZView = VK_NULL_HANDLE; }
    if (mHiZImage) { vmaDestroyImage(allocator, mHiZImage, mHiZAlloc); mHiZImage = VK_NULL_HANDLE; }

    if (mDescPool) { vkDestroyDescriptorPool(device, mDescPool, nullptr); mDescPool = VK_NULL_HANDLE; }
    if (mPipeline) { vkDestroyPipeline(device, mPipeline, nullptr); mPipeline = VK_NULL_HANDLE; }
    if (mPipelineLayout) { vkDestroyPipelineLayout(device, mPipelineLayout, nullptr); mPipelineLayout = VK_NULL_HANDLE; }
    if (mDescSetLayout)  { vkDestroyDescriptorSetLayout(device, mDescSetLayout, nullptr); mDescSetLayout = VK_NULL_HANDLE; }
    if (mReduceSampler)  { vkDestroySampler(device, mReduceSampler, nullptr); mReduceSampler = VK_NULL_HANDLE; }
}

void HiZBuffer::Resize(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height) {
    for (auto view : mMipViews)
        if (view) vkDestroyImageView(device, view, nullptr);
    mMipViews.clear();

    if (mHiZView) { vkDestroyImageView(device, mHiZView, nullptr); mHiZView = VK_NULL_HANDLE; }
    if (mHiZImage) { vmaDestroyImage(allocator, mHiZImage, mHiZAlloc); mHiZImage = VK_NULL_HANDLE; }
    if (mDescPool) { vkDestroyDescriptorPool(device, mDescPool, nullptr); mDescPool = VK_NULL_HANDLE; }
    mDescSets.clear();

    mWidth  = std::max(width  / 2, 1u);
    mHeight = std::max(height / 2, 1u);
    mMipCount = static_cast<uint32_t>(std::floor(std::log2(std::max(mWidth, mHeight)))) + 1;

    CreateHiZImage(device, allocator);
    CreateDescriptors(device);
}

void HiZBuffer::CreateHiZImage(VkDevice device, VmaAllocator allocator) {
    VkImageCreateInfo imgInfo{};
    imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgInfo.format        = VK_FORMAT_R32_SFLOAT;
    imgInfo.extent        = { mWidth, mHeight, 1 };
    imgInfo.mipLevels     = mMipCount;
    imgInfo.arrayLayers   = 1;
    imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VK_CHECK(vmaCreateImage(allocator, &imgInfo, &allocInfo, &mHiZImage, &mHiZAlloc, nullptr));

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = mHiZImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = VK_FORMAT_R32_SFLOAT;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = mMipCount;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &mHiZView));

    mMipViews.resize(mMipCount);
    for (uint32_t i = 0; i < mMipCount; i++) {
        VkImageViewCreateInfo mipView = viewInfo;
        mipView.subresourceRange.baseMipLevel = i;
        mipView.subresourceRange.levelCount   = 1;
        VK_CHECK(vkCreateImageView(device, &mipView, nullptr, &mMipViews[i]));
    }
}

void HiZBuffer::CreateDescriptors(VkDevice device) {
    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = mMipCount;
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = mMipCount;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets       = mMipCount;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes    = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &mDescPool));

    std::vector<VkDescriptorSetLayout> layouts(mMipCount, mDescSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = mDescPool;
    allocInfo.descriptorSetCount = mMipCount;
    allocInfo.pSetLayouts        = layouts.data();
    mDescSets.resize(mMipCount);
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, mDescSets.data()));
}

void HiZBuffer::SetSourceDepth(VkImageView depthView) {
    mSourceDepthView = depthView;
}

void HiZBuffer::UpdateSourceDescriptor(VkDevice device) {
    for (uint32_t i = 0; i < mMipCount; i++) {
        VkDescriptorImageInfo srcInfo{};
        srcInfo.sampler     = mReduceSampler;
        srcInfo.imageView   = (i == 0) ? mSourceDepthView : mMipViews[i - 1];
        srcInfo.imageLayout = (i == 0) ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                       : VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo dstInfo{};
        dstInfo.imageView   = mMipViews[i];
        dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = mDescSets[i];
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo      = &srcInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = mDescSets[i];
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo      = &dstInfo;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }
}

void HiZBuffer::BuildMipChain(VkCommandBuffer cmd) const {
    if (mMipCount == 0 || mSourceDepthView == VK_NULL_HANDLE) return;

    const_cast<HiZBuffer*>(this)->UpdateSourceDescriptor(mDevice);

    VkImageMemoryBarrier2 toGeneral{};
    toGeneral.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toGeneral.srcStageMask        = VK_PIPELINE_STAGE_2_NONE;
    toGeneral.srcAccessMask       = VK_ACCESS_2_NONE;
    toGeneral.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    toGeneral.dstAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    toGeneral.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    toGeneral.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.image               = mHiZImage;
    toGeneral.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mMipCount, 0, 1 };

    VkDependencyInfo dep{};
    dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount  = 1;
    dep.pImageMemoryBarriers     = &toGeneral;
    vkCmdPipelineBarrier2(cmd, &dep);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mPipeline);

    for (uint32_t mip = 0; mip < mMipCount; mip++) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                mPipelineLayout, 0, 1, &mDescSets[mip], 0, nullptr);

        uint32_t mipW = std::max(mWidth >> mip, 1u);
        uint32_t mipH = std::max(mHeight >> mip, 1u);
        vkCmdDispatch(cmd, (mipW + 7) / 8, (mipH + 7) / 8, 1);

        if (mip < mMipCount - 1) {
            VkImageMemoryBarrier2 mipBarrier{};
            mipBarrier.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            mipBarrier.srcStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mipBarrier.srcAccessMask    = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            mipBarrier.dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mipBarrier.dstAccessMask    = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            mipBarrier.oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
            mipBarrier.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
            mipBarrier.image            = mHiZImage;
            mipBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1 };

            VkDependencyInfo mipDep{};
            mipDep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            mipDep.imageMemoryBarrierCount = 1;
            mipDep.pImageMemoryBarriers    = &mipBarrier;
            vkCmdPipelineBarrier2(cmd, &mipDep);
        }
    }
}
