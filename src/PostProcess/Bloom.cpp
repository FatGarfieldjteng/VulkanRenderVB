#include "PostProcess/Bloom.h"
#include "Resource/ShaderManager.h"
#include "Core/Logger.h"
#include "RHI/VulkanUtils.h"

#include <algorithm>
#include <cmath>

void Bloom::Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders,
                       uint32_t width, uint32_t height) {
    mDevice = device;
    mWidth  = std::max(width >> 1, 1u);
    mHeight = std::max(height >> 1, 1u);

    // Linear sampler for bloom texture reading
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter     = VK_FILTER_LINEAR;
    samplerInfo.minFilter     = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.minLod        = 0.0f;
    samplerInfo.maxLod        = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &mLinearSampler));

    // Downsample layout: binding 0 = combined_image_sampler, binding 1 = storage_image
    VkDescriptorSetLayoutBinding downBindings[2]{};
    downBindings[0].binding         = 0;
    downBindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    downBindings[0].descriptorCount = 1;
    downBindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    downBindings[1].binding         = 1;
    downBindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    downBindings[1].descriptorCount = 1;
    downBindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorBindingFlags downBindFlags[2] = {
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo downFlagsInfo{};
    downFlagsInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    downFlagsInfo.bindingCount  = 2;
    downFlagsInfo.pBindingFlags = downBindFlags;

    VkDescriptorSetLayoutCreateInfo downLayoutInfo{};
    downLayoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    downLayoutInfo.pNext        = &downFlagsInfo;
    downLayoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    downLayoutInfo.bindingCount = 2;
    downLayoutInfo.pBindings    = downBindings;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &downLayoutInfo, nullptr, &mDownLayout));

    // Upsample layout: binding 0 = combined_image_sampler, binding 1 = storage_image (read-write)
    VkDescriptorSetLayoutBinding upBindings[2]{};
    upBindings[0].binding         = 0;
    upBindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    upBindings[0].descriptorCount = 1;
    upBindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    upBindings[1].binding         = 1;
    upBindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    upBindings[1].descriptorCount = 1;
    upBindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorBindingFlags upBindFlags[2] = {
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo upFlagsInfo{};
    upFlagsInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    upFlagsInfo.bindingCount  = 2;
    upFlagsInfo.pBindingFlags = upBindFlags;

    VkDescriptorSetLayoutCreateInfo upLayoutInfo{};
    upLayoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    upLayoutInfo.pNext        = &upFlagsInfo;
    upLayoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    upLayoutInfo.bindingCount = 2;
    upLayoutInfo.pBindings    = upBindings;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &upLayoutInfo, nullptr, &mUpLayout));

    // Push constants: vec2 srcTexelSize (8), uvec2 dstSize (8), uint/float (4) = 20 bytes
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset     = 0;
    pushRange.size       = 20;

    VkDescriptorSetLayout layouts[] = { mDownLayout, mUpLayout };
    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount = 1;
    pipeLayoutInfo.pSetLayouts    = layouts;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges    = &pushRange;

    VkPipelineLayoutCreateInfo downPipeLayoutInfo = pipeLayoutInfo;
    downPipeLayoutInfo.pSetLayouts = &mDownLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &downPipeLayoutInfo, nullptr, &mDownPipeLayout));

    VkPipelineLayoutCreateInfo upPipeLayoutInfo = pipeLayoutInfo;
    upPipeLayoutInfo.pSetLayouts = &mUpLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &upPipeLayoutInfo, nullptr, &mUpPipeLayout));

    // Create compute pipelines
    VkShaderModule downModule = shaders.GetOrLoad("shaders/bloom_downsample.comp.spv");
    VkComputePipelineCreateInfo downCompInfo{};
    downCompInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    downCompInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    downCompInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    downCompInfo.stage.module = downModule;
    downCompInfo.stage.pName  = "main";
    downCompInfo.layout       = mDownPipeLayout;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &downCompInfo, nullptr, &mDownPipeline));

    VkShaderModule upModule = shaders.GetOrLoad("shaders/bloom_upsample.comp.spv");
    VkComputePipelineCreateInfo upCompInfo{};
    upCompInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    upCompInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    upCompInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    upCompInfo.stage.module = upModule;
    upCompInfo.stage.pName  = "main";
    upCompInfo.layout       = mUpPipeLayout;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &upCompInfo, nullptr, &mUpPipeline));

    CreateImages(device, allocator, mWidth, mHeight);
    CreateDescriptors(device);

    LOG_INFO("Bloom initialized");
}

void Bloom::Shutdown(VkDevice device, VmaAllocator allocator) {
    DestroyImages(device, allocator);

    if (mDescPool)      { vkDestroyDescriptorPool(device, mDescPool, nullptr);      mDescPool = VK_NULL_HANDLE; }
    if (mDownPipeline)  { vkDestroyPipeline(device, mDownPipeline, nullptr);        mDownPipeline = VK_NULL_HANDLE; }
    if (mUpPipeline)    { vkDestroyPipeline(device, mUpPipeline, nullptr);          mUpPipeline = VK_NULL_HANDLE; }
    if (mDownPipeLayout){ vkDestroyPipelineLayout(device, mDownPipeLayout, nullptr); mDownPipeLayout = VK_NULL_HANDLE; }
    if (mUpPipeLayout)  { vkDestroyPipelineLayout(device, mUpPipeLayout, nullptr);   mUpPipeLayout = VK_NULL_HANDLE; }
    if (mDownLayout)    { vkDestroyDescriptorSetLayout(device, mDownLayout, nullptr); mDownLayout = VK_NULL_HANDLE; }
    if (mUpLayout)      { vkDestroyDescriptorSetLayout(device, mUpLayout, nullptr);   mUpLayout = VK_NULL_HANDLE; }
    if (mLinearSampler)  { vkDestroySampler(device, mLinearSampler, nullptr);         mLinearSampler = VK_NULL_HANDLE; }
}

void Bloom::Resize(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height) {
    mWidth  = std::max(width >> 1, 1u);
    mHeight = std::max(height >> 1, 1u);
    DestroyImages(device, allocator);
    CreateImages(device, allocator, mWidth, mHeight);
    CreateDescriptors(device);
}

void Bloom::CreateImages(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h) {
    VkImageCreateInfo imgInfo{};
    imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType    = VK_IMAGE_TYPE_2D;
    imgInfo.format       = VK_FORMAT_R16G16B16A16_SFLOAT;
    imgInfo.extent       = { w, h, 1 };
    imgInfo.mipLevels    = MIP_COUNT;
    imgInfo.arrayLayers  = 1;
    imgInfo.samples      = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling       = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage        = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VK_CHECK(vmaCreateImage(allocator, &imgInfo, &allocInfo, &mBloomImage, &mBloomAlloc, nullptr));

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = mBloomImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = VK_FORMAT_R16G16B16A16_SFLOAT;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount   = MIP_COUNT;
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &mBloomFullView));

    mMipViews.resize(MIP_COUNT);
    for (uint32_t i = 0; i < MIP_COUNT; i++) {
        viewInfo.subresourceRange.baseMipLevel = i;
        viewInfo.subresourceRange.levelCount   = 1;
        VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &mMipViews[i]));
    }
}

void Bloom::DestroyImages(VkDevice device, VmaAllocator allocator) {
    for (auto& view : mMipViews)
        if (view) { vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
    mMipViews.clear();

    if (mBloomFullView) { vkDestroyImageView(device, mBloomFullView, nullptr); mBloomFullView = VK_NULL_HANDLE; }
    if (mBloomImage)    { vmaDestroyImage(allocator, mBloomImage, mBloomAlloc); mBloomImage = VK_NULL_HANDLE; mBloomAlloc = VK_NULL_HANDLE; }
}

void Bloom::CreateDescriptors(VkDevice device) {
    if (mDescPool) {
        vkDestroyDescriptorPool(device, mDescPool, nullptr);
        mDescPool = VK_NULL_HANDLE;
    }
    mDownSets.clear();
    mUpSets.clear();

    if (mMipViews.empty()) return;

    // Pool: MIP_COUNT downsample + (MIP_COUNT-1) upsample sets
    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = MIP_COUNT + (MIP_COUNT - 1);
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = MIP_COUNT + (MIP_COUNT - 1);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets       = MIP_COUNT + (MIP_COUNT - 1);
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes    = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &mDescPool));

    // Allocate downsample sets
    std::vector<VkDescriptorSetLayout> downLayouts(MIP_COUNT, mDownLayout);
    VkDescriptorSetAllocateInfo downAllocInfo{};
    downAllocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    downAllocInfo.descriptorPool     = mDescPool;
    downAllocInfo.descriptorSetCount = MIP_COUNT;
    downAllocInfo.pSetLayouts        = downLayouts.data();
    mDownSets.resize(MIP_COUNT);
    VK_CHECK(vkAllocateDescriptorSets(device, &downAllocInfo, mDownSets.data()));

    // Allocate upsample sets
    std::vector<VkDescriptorSetLayout> upLayouts(MIP_COUNT - 1, mUpLayout);
    VkDescriptorSetAllocateInfo upAllocInfo{};
    upAllocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    upAllocInfo.descriptorPool     = mDescPool;
    upAllocInfo.descriptorSetCount = MIP_COUNT - 1;
    upAllocInfo.pSetLayouts        = upLayouts.data();
    mUpSets.resize(MIP_COUNT - 1);
    VK_CHECK(vkAllocateDescriptorSets(device, &upAllocInfo, mUpSets.data()));

    // Update downsample descriptors: sets 1..5 use mMipViews (set 0 updated per-Dispatch)
    for (uint32_t i = 1; i < MIP_COUNT; i++) {
        VkDescriptorImageInfo srcInfo{};
        srcInfo.sampler     = mLinearSampler;
        srcInfo.imageView   = mMipViews[i - 1];
        srcInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo dstInfo{};
        dstInfo.imageView   = mMipViews[i];
        dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = mDownSets[i];
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo      = &srcInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = mDownSets[i];
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo      = &dstInfo;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }

    // Downsample set 0: dst only (src = hdrView updated in Dispatch)
    {
        VkDescriptorImageInfo dstInfo{};
        dstInfo.imageView   = mMipViews[0];
        dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = mDownSets[0];
        write.dstBinding      = 1;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo      = &dstInfo;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    // Update upsample descriptors
    for (uint32_t i = 0; i < MIP_COUNT - 1; i++) {
        VkDescriptorImageInfo srcInfo{};
        srcInfo.sampler     = mLinearSampler;
        srcInfo.imageView   = mMipViews[i + 1];
        srcInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo dstInfo{};
        dstInfo.imageView   = mMipViews[i];
        dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = mUpSets[i];
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo      = &srcInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = mUpSets[i];
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo      = &dstInfo;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }
}

void Bloom::Dispatch(VkCommandBuffer cmd, VkImageView hdrView, VkSampler hdrSampler,
                     uint32_t srcWidth, uint32_t srcHeight) {
    if (mMipViews.empty() || hdrView == VK_NULL_HANDLE) return;

    VkSampler samplerToUse = (hdrSampler != VK_NULL_HANDLE) ? hdrSampler : mLinearSampler;

    VkDescriptorImageInfo srcInfo{};
    srcInfo.sampler     = samplerToUse;
    srcInfo.imageView   = hdrView;
    srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = mDownSets[0];
    write.dstBinding      = 0;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo      = &srcInfo;
    vkUpdateDescriptorSets(mDevice, 1, &write, 0, nullptr);

    // 1. Transition all mips to GENERAL
    TransitionImage(cmd, mBloomImage,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, MIP_COUNT, 0, 1);

    // 2. Downsample chain (6 passes)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mDownPipeline);

    for (uint32_t i = 0; i < MIP_COUNT; i++) {
        uint32_t dstW = std::max(mWidth >> i, 1u);
        uint32_t dstH = std::max(mHeight >> i, 1u);

        float srcTexelU = (i == 0) ? (1.0f / static_cast<float>(srcWidth))
                                   : (1.0f / static_cast<float>(std::max(mWidth >> (i - 1), 1u)));
        float srcTexelV = (i == 0) ? (1.0f / static_cast<float>(srcHeight))
                                   : (1.0f / static_cast<float>(std::max(mHeight >> (i - 1), 1u)));

        struct PushConstants {
            float texelSize[2];
            uint32_t dstSize[2];
            uint32_t mipLevel;
        } push;
        push.texelSize[0] = srcTexelU;
        push.texelSize[1] = srcTexelV;
        push.dstSize[0]   = dstW;
        push.dstSize[1]   = dstH;
        push.mipLevel     = i;

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                mDownPipeLayout, 0, 1, &mDownSets[i], 0, nullptr);
        vkCmdPushConstants(cmd, mDownPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, &push);
        vkCmdDispatch(cmd, (dstW + 7) / 8, (dstH + 7) / 8, 1);

        if (i < MIP_COUNT - 1) {
            TransitionImage(cmd, mBloomImage,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1);
        }
    }

    // 3. Upsample chain (5 passes, from mip 5 up to mip 0)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mUpPipeline);

    for (int32_t i = static_cast<int32_t>(MIP_COUNT) - 2; i >= 0; --i) {
        uint32_t dstW = std::max(mWidth >> i, 1u);
        uint32_t dstH = std::max(mHeight >> i, 1u);
        uint32_t srcW = std::max(mWidth >> (i + 1), 1u);
        uint32_t srcH = std::max(mHeight >> (i + 1), 1u);

        float srcTexelU = 1.0f / static_cast<float>(srcW);
        float srcTexelV = 1.0f / static_cast<float>(srcH);
        float filterRadius = 1.0f;

        struct UpPushConstants {
            float texelSize[2];
            uint32_t dstSize[2];
            float filterRadius;
        } push;
        push.texelSize[0]   = srcTexelU;
        push.texelSize[1]   = srcTexelV;
        push.dstSize[0]     = dstW;
        push.dstSize[1]     = dstH;
        push.filterRadius   = filterRadius;

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                mUpPipeLayout, 0, 1, &mUpSets[i], 0, nullptr);
        vkCmdPushConstants(cmd, mUpPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, &push);
        vkCmdDispatch(cmd, (dstW + 7) / 8, (dstH + 7) / 8, 1);

        if (i > 0) {
            TransitionImage(cmd, mBloomImage,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                            VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1);
        }
    }

    // 4. Transition mip 0 to SHADER_READ_ONLY for tone mapping
    TransitionImage(cmd, mBloomImage,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
}
