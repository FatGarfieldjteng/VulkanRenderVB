#include "PostProcess/SSAO.h"
#include "Resource/ShaderManager.h"
#include "Core/Logger.h"
#include "RHI/VulkanUtils.h"

#include <algorithm>
#include <cstring>

namespace {

struct GTAOPushConstants {
    float invProjection[16];  // mat4
    float projInfo[4];        // vec4
    float noiseScale[2];      // vec2
    float radius;
    float bias;
    float intensity;
    float farPlane;
    uint32_t resolution[2];   // uvec2
};
static_assert(sizeof(GTAOPushConstants) == 112, "GTAO push constants size mismatch");

struct BlurPushConstants {
    float direction[2];       // vec2
    uint32_t resolution[2];   // uvec2
    float depthThreshold;
};
static_assert(sizeof(BlurPushConstants) == 20, "Blur push constants size mismatch");

constexpr float kBlurDepthThreshold = 0.02f;

} // namespace

void SSAO::Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders,
                      uint32_t width, uint32_t height) {
    mDevice = device;
    mWidth  = width;
    mHeight = height;

    // Nearest sampler for depth/AO
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter     = VK_FILTER_NEAREST;
    samplerInfo.minFilter     = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &mNearestSampler));

    // GTAO layout: binding 0 = combined_image_sampler (depth), binding 1 = storage_image r8 (ao output)
    {
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorBindingFlags gtaoBindFlags[2] = {
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo gtaoFlagsInfo{};
        gtaoFlagsInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        gtaoFlagsInfo.bindingCount  = 2;
        gtaoFlagsInfo.pBindingFlags = gtaoBindFlags;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.pNext        = &gtaoFlagsInfo;
        layoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings    = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &mGTAOLayout));

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset     = 0;
        pushRange.size       = sizeof(GTAOPushConstants);

        VkPipelineLayoutCreateInfo pipeLayoutInfo{};
        pipeLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeLayoutInfo.setLayoutCount = 1;
        pipeLayoutInfo.pSetLayouts    = &mGTAOLayout;
        pipeLayoutInfo.pushConstantRangeCount = 1;
        pipeLayoutInfo.pPushConstantRanges    = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(device, &pipeLayoutInfo, nullptr, &mGTAOPipeLayout));

        VkShaderModule compModule = shaders.GetOrLoad("shaders/gtao.comp.spv");
        VkComputePipelineCreateInfo compInfo{};
        compInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        compInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        compInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        compInfo.stage.module = compModule;
        compInfo.stage.pName  = "main";
        compInfo.layout       = mGTAOPipeLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compInfo, nullptr, &mGTAOPipeline));
    }

    // Blur layout: binding 0 = combined_image_sampler (ao input), binding 1 = combined_image_sampler (depth),
    //              binding 2 = storage_image r8 (ao output)
    {
        VkDescriptorSetLayoutBinding bindings[3]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[2].binding         = 2;
        bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorBindingFlags blurBindFlags[3] = {
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo blurFlagsInfo{};
        blurFlagsInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        blurFlagsInfo.bindingCount  = 3;
        blurFlagsInfo.pBindingFlags = blurBindFlags;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.pNext        = &blurFlagsInfo;
        layoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings    = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &mBlurLayout));

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset     = 0;
        pushRange.size       = sizeof(BlurPushConstants);

        VkPipelineLayoutCreateInfo pipeLayoutInfo{};
        pipeLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeLayoutInfo.setLayoutCount = 1;
        pipeLayoutInfo.pSetLayouts    = &mBlurLayout;
        pipeLayoutInfo.pushConstantRangeCount = 1;
        pipeLayoutInfo.pPushConstantRanges    = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(device, &pipeLayoutInfo, nullptr, &mBlurPipeLayout));

        VkShaderModule compModule = shaders.GetOrLoad("shaders/gtao_blur.comp.spv");
        VkComputePipelineCreateInfo compInfo{};
        compInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        compInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        compInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        compInfo.stage.module = compModule;
        compInfo.stage.pName  = "main";
        compInfo.layout       = mBlurPipeLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compInfo, nullptr, &mBlurPipeline));
    }

    CreateImages(device, allocator, width, height);
    CreateDescriptors(device);

    LOG_INFO("SSAO (GTAO) initialized {}x{}", width, height);
}

void SSAO::Shutdown(VkDevice device, VmaAllocator allocator) {
    DestroyImages(device, allocator);

    if (mDescPool)            { vkDestroyDescriptorPool(device, mDescPool, nullptr);            mDescPool = VK_NULL_HANDLE; }
    if (mGTAOPipeline)        { vkDestroyPipeline(device, mGTAOPipeline, nullptr);            mGTAOPipeline = VK_NULL_HANDLE; }
    if (mGTAOPipeLayout)      { vkDestroyPipelineLayout(device, mGTAOPipeLayout, nullptr);    mGTAOPipeLayout = VK_NULL_HANDLE; }
    if (mGTAOLayout)           { vkDestroyDescriptorSetLayout(device, mGTAOLayout, nullptr);    mGTAOLayout = VK_NULL_HANDLE; }
    if (mBlurPipeline)         { vkDestroyPipeline(device, mBlurPipeline, nullptr);            mBlurPipeline = VK_NULL_HANDLE; }
    if (mBlurPipeLayout)       { vkDestroyPipelineLayout(device, mBlurPipeLayout, nullptr);    mBlurPipeLayout = VK_NULL_HANDLE; }
    if (mBlurLayout)           { vkDestroyDescriptorSetLayout(device, mBlurLayout, nullptr);    mBlurLayout = VK_NULL_HANDLE; }
    if (mNearestSampler)       { vkDestroySampler(device, mNearestSampler, nullptr);           mNearestSampler = VK_NULL_HANDLE; }
}

void SSAO::Resize(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height) {
    if (mWidth == width && mHeight == height)
        return;
    mWidth  = width;
    mHeight = height;
    DestroyImages(device, allocator);
    if (mDescPool) {
        vkDestroyDescriptorPool(device, mDescPool, nullptr);
        mDescPool = VK_NULL_HANDLE;
    }
    CreateImages(device, allocator, width, height);
    CreateDescriptors(device);
}

void SSAO::CreateImages(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h) {
    w = std::max(w, 1u);
    h = std::max(h, 1u);

    VkImageCreateInfo imgInfo{};
    imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgInfo.format        = VK_FORMAT_R8_UNORM;
    imgInfo.extent        = { w, h, 1 };
    imgInfo.mipLevels     = 1;
    imgInfo.arrayLayers   = 1;
    imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VK_CHECK(vmaCreateImage(allocator, &imgInfo, &allocInfo, &mAOImage, &mAOAlloc, nullptr));
    VK_CHECK(vmaCreateImage(allocator, &imgInfo, &allocInfo, &mAOTempImage, &mAOTempAlloc, nullptr));

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = VK_FORMAT_R8_UNORM;
    viewInfo.subresourceRange.aspectMask   = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount   = 1;
    viewInfo.subresourceRange.layerCount   = 1;

    viewInfo.image = mAOImage;
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &mAOView));
    viewInfo.image = mAOTempImage;
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &mAOTempView));

    mAOLayout    = VK_IMAGE_LAYOUT_UNDEFINED;
    mAOTempLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void SSAO::DestroyImages(VkDevice device, VmaAllocator allocator) {
    if (mAOView)      { vkDestroyImageView(device, mAOView, nullptr);      mAOView = VK_NULL_HANDLE; }
    if (mAOImage)     { vmaDestroyImage(allocator, mAOImage, mAOAlloc);   mAOImage = VK_NULL_HANDLE; mAOAlloc = VK_NULL_HANDLE; }
    if (mAOTempView)  { vkDestroyImageView(device, mAOTempView, nullptr);  mAOTempView = VK_NULL_HANDLE; }
    if (mAOTempImage) { vmaDestroyImage(allocator, mAOTempImage, mAOTempAlloc); mAOTempImage = VK_NULL_HANDLE; mAOTempAlloc = VK_NULL_HANDLE; }
}

void SSAO::CreateDescriptors(VkDevice device) {
    // Pool: 3 sets, GTAO (1 sampler + 1 storage), Blur x2 (2 samplers + 1 storage each)
    // Total: 1 combined + 1 storage for GTAO; 2*(2 combined + 1 storage) for blur = 5 combined + 3 storage
    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 5;
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 3;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets       = 3;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes    = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &mDescPool));

    VkDescriptorSetLayout layouts[3] = { mGTAOLayout, mBlurLayout, mBlurLayout };
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = mDescPool;
    allocInfo.descriptorSetCount = 3;
    allocInfo.pSetLayouts        = layouts;
    VkDescriptorSet sets[3];
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, sets));
    mGTAOSet   = sets[0];
    mBlurHSet  = sets[1];
    mBlurVSet  = sets[2];
}

void SSAO::Dispatch(VkCommandBuffer cmd, VkImageView depthView,
                    const float* invProjection, const float* projInfo,
                    float radius, float bias, float intensity, float farPlane,
                    uint32_t width, uint32_t height) {
    if (!mAOImage || !depthView) return;
    width  = std::max(width, 1u);
    height = std::max(height, 1u);

    // ---- 1. Update GTAO descriptor set: depth + mAOImage (output) ----
    {
        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler     = mNearestSampler;
        depthInfo.imageView   = depthView;
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo aoOutInfo{};
        aoOutInfo.imageView   = mAOView;
        aoOutInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = mGTAOSet;
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo      = &depthInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = mGTAOSet;
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo      = &aoOutInfo;

        vkUpdateDescriptorSets(mDevice, 2, writes, 0, nullptr);
    }

    // ---- 2. Transition mAOImage to GENERAL ----
    TransitionImage(cmd, mAOImage,
                    (mAOLayout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    (mAOLayout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    mAOLayout, VK_IMAGE_LAYOUT_GENERAL);
    mAOLayout = VK_IMAGE_LAYOUT_GENERAL;

    // ---- 3. Dispatch GTAO ----
    GTAOPushConstants gtaoPC{};
    std::memcpy(gtaoPC.invProjection, invProjection, sizeof(gtaoPC.invProjection));
    std::memcpy(gtaoPC.projInfo, projInfo, sizeof(gtaoPC.projInfo));
    gtaoPC.noiseScale[0] = 1.0f;
    gtaoPC.noiseScale[1] = 1.0f;
    gtaoPC.radius        = radius;
    gtaoPC.bias          = bias;
    gtaoPC.intensity     = intensity;
    gtaoPC.farPlane      = farPlane;
    gtaoPC.resolution[0] = width;
    gtaoPC.resolution[1] = height;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mGTAOPipeline);
    vkCmdPushConstants(cmd, mGTAOPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GTAOPushConstants), &gtaoPC);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mGTAOPipeLayout, 0, 1, &mGTAOSet, 0, nullptr);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);

    // ---- 4. Barrier: GTAO write complete ----
    TransitionImage(cmd, mAOImage,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    mAOLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // ---- 5. Update blur H set: input=mAOView, depth=depthView, output=mAOTempView, direction=(1,0) ----
    {
        VkDescriptorImageInfo aoInInfo{};
        aoInInfo.sampler     = mNearestSampler;
        aoInInfo.imageView   = mAOView;
        aoInInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler     = mNearestSampler;
        depthInfo.imageView   = depthView;
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo aoOutInfo{};
        aoOutInfo.imageView   = mAOTempView;
        aoOutInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[3]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = mBlurHSet;
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo      = &aoInInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = mBlurHSet;
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo      = &depthInfo;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = mBlurHSet;
        writes[2].dstBinding      = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].pImageInfo      = &aoOutInfo;

        vkUpdateDescriptorSets(mDevice, 3, writes, 0, nullptr);
    }

    // ---- 6. Transition mAOTempImage to GENERAL ----
    TransitionImage(cmd, mAOTempImage,
                    (mAOTempLayout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    (mAOTempLayout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    mAOTempLayout, VK_IMAGE_LAYOUT_GENERAL);
    mAOTempLayout = VK_IMAGE_LAYOUT_GENERAL;

    // ---- 7. Dispatch blur H ----
    BlurPushConstants blurPC{};
    blurPC.direction[0]     = 1.0f;
    blurPC.direction[1]     = 0.0f;
    blurPC.resolution[0]    = width;
    blurPC.resolution[1]    = height;
    blurPC.depthThreshold  = kBlurDepthThreshold;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mBlurPipeline);
    vkCmdPushConstants(cmd, mBlurPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurPushConstants), &blurPC);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mBlurPipeLayout, 0, 1, &mBlurHSet, 0, nullptr);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);

    // ---- 8. Barrier: blur H write complete ----
    TransitionImage(cmd, mAOTempImage,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    mAOTempLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // ---- 9. Update blur V set: input=mAOTempView, depth=depthView, output=mAOView, direction=(0,1) ----
    {
        VkDescriptorImageInfo aoInInfo{};
        aoInInfo.sampler     = mNearestSampler;
        aoInInfo.imageView   = mAOTempView;
        aoInInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler     = mNearestSampler;
        depthInfo.imageView   = depthView;
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo aoOutInfo{};
        aoOutInfo.imageView   = mAOView;
        aoOutInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[3]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = mBlurVSet;
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo      = &aoInInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = mBlurVSet;
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo      = &depthInfo;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = mBlurVSet;
        writes[2].dstBinding      = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].pImageInfo      = &aoOutInfo;

        vkUpdateDescriptorSets(mDevice, 3, writes, 0, nullptr);
    }

    // ---- 10. Transition mAOImage to GENERAL for blur V output ----
    TransitionImage(cmd, mAOImage,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    mAOLayout, VK_IMAGE_LAYOUT_GENERAL);
    mAOLayout = VK_IMAGE_LAYOUT_GENERAL;

    // ---- 11. Dispatch blur V ----
    blurPC.direction[0] = 0.0f;
    blurPC.direction[1] = 1.0f;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mBlurPipeline);
    vkCmdPushConstants(cmd, mBlurPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurPushConstants), &blurPC);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mBlurPipeLayout, 0, 1, &mBlurVSet, 0, nullptr);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);

    // ---- 12. Final barrier: mAOImage to SHADER_READ_ONLY for downstream use ----
    TransitionImage(cmd, mAOImage,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    mAOLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}
