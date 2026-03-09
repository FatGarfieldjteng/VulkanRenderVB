#include "PostProcess/AutoExposure.h"
#include "Resource/ShaderManager.h"
#include "Core/Logger.h"
#include "RHI/VulkanUtils.h"

#include <cstring>

namespace {

inline void TransitionBuffer(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset,
                             VkDeviceSize size,
                             VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                             VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkBufferMemoryBarrier2 barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask        = srcStage;
    barrier.srcAccessMask       = srcAccess;
    barrier.dstStageMask        = dstStage;
    barrier.dstAccessMask       = dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer              = buffer;
    barrier.offset              = offset;
    barrier.size                = size;

    VkDependencyInfo dep{};
    dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers    = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep);
}

struct HistogramPushConstants {
    float minLogLum;
    float invLogLumRange;
    uint32_t inputSizeX;
    uint32_t inputSizeY;
};

struct AveragePushConstants {
    float minLogLum;
    float logLumRange;
    float deltaTime;
    float adaptSpeed;
    uint32_t pixelCount;
};

} // namespace

void AutoExposure::Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders) {
    mDevice = device;

    // Create linear sampler for HDR reading
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

    // Histogram descriptor set layout: binding 0 = combined_image_sampler, binding 1 = storage_buffer
    VkDescriptorSetLayoutBinding histogramBindings[2]{};
    histogramBindings[0].binding         = 0;
    histogramBindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    histogramBindings[0].descriptorCount = 1;
    histogramBindings[0].stageFlags     = VK_SHADER_STAGE_COMPUTE_BIT;
    histogramBindings[1].binding         = 1;
    histogramBindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    histogramBindings[1].descriptorCount = 1;
    histogramBindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorBindingFlags histBindFlags[2] = {
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo histFlagsInfo{};
    histFlagsInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    histFlagsInfo.bindingCount  = 2;
    histFlagsInfo.pBindingFlags = histBindFlags;

    VkDescriptorSetLayoutCreateInfo histogramLayoutInfo{};
    histogramLayoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    histogramLayoutInfo.pNext        = &histFlagsInfo;
    histogramLayoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    histogramLayoutInfo.bindingCount = 2;
    histogramLayoutInfo.pBindings    = histogramBindings;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &histogramLayoutInfo, nullptr, &mHistogramLayout));

    // Average descriptor set layout: binding 0 = storage_buffer (histogram), binding 1 = storage_buffer (exposure)
    VkDescriptorSetLayoutBinding averageBindings[2]{};
    averageBindings[0].binding         = 0;
    averageBindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    averageBindings[0].descriptorCount = 1;
    averageBindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    averageBindings[1].binding         = 1;
    averageBindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    averageBindings[1].descriptorCount = 1;
    averageBindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo averageLayoutInfo{};
    averageLayoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    averageLayoutInfo.bindingCount = 2;
    averageLayoutInfo.pBindings    = averageBindings;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &averageLayoutInfo, nullptr, &mAverageLayout));

    // Histogram pipeline layout with push constants
    VkPushConstantRange histogramPushRange{};
    histogramPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    histogramPushRange.offset     = 0;
    histogramPushRange.size      = sizeof(HistogramPushConstants);

    VkPipelineLayoutCreateInfo histogramPipeLayoutInfo{};
    histogramPipeLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    histogramPipeLayoutInfo.setLayoutCount         = 1;
    histogramPipeLayoutInfo.pSetLayouts            = &mHistogramLayout;
    histogramPipeLayoutInfo.pushConstantRangeCount = 1;
    histogramPipeLayoutInfo.pPushConstantRanges    = &histogramPushRange;
    VK_CHECK(vkCreatePipelineLayout(device, &histogramPipeLayoutInfo, nullptr, &mHistogramPipeLayout));

    // Average pipeline layout with push constants
    VkPushConstantRange averagePushRange{};
    averagePushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    averagePushRange.offset     = 0;
    averagePushRange.size      = sizeof(AveragePushConstants);

    VkPipelineLayoutCreateInfo averagePipeLayoutInfo{};
    averagePipeLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    averagePipeLayoutInfo.setLayoutCount         = 1;
    averagePipeLayoutInfo.pSetLayouts            = &mAverageLayout;
    averagePipeLayoutInfo.pushConstantRangeCount = 1;
    averagePipeLayoutInfo.pPushConstantRanges    = &averagePushRange;
    VK_CHECK(vkCreatePipelineLayout(device, &averagePipeLayoutInfo, nullptr, &mAveragePipeLayout));

    // Create compute pipelines
    VkShaderModule histogramModule = shaders.GetOrLoad("shaders/exposure_histogram.comp.spv");
    VkComputePipelineCreateInfo histogramCompInfo{};
    histogramCompInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    histogramCompInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    histogramCompInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    histogramCompInfo.stage.module = histogramModule;
    histogramCompInfo.stage.pName  = "main";
    histogramCompInfo.layout       = mHistogramPipeLayout;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &histogramCompInfo, nullptr,
                                      &mHistogramPipeline));

    VkShaderModule averageModule = shaders.GetOrLoad("shaders/exposure_average.comp.spv");
    VkComputePipelineCreateInfo averageCompInfo{};
    averageCompInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    averageCompInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    averageCompInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    averageCompInfo.stage.module = averageModule;
    averageCompInfo.stage.pName  = "main";
    averageCompInfo.layout       = mAveragePipeLayout;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &averageCompInfo, nullptr,
                                      &mAveragePipeline));

    // Descriptor pool: 1 combined_image_sampler, 2 storage_buffers for histogram set;
    // 2 storage_buffers for average set
    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 4;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets       = 2;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes    = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &mDescPool));

    // Allocate descriptor sets
    VkDescriptorSetLayout layouts[] = { mHistogramLayout, mAverageLayout };
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = mDescPool;
    allocInfo.descriptorSetCount = 2;
    allocInfo.pSetLayouts        = layouts;
    VkDescriptorSet sets[2];
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, sets));
    mHistogramSet = sets[0];
    mAverageSet   = sets[1];

    // Create histogram buffer: 256 * sizeof(uint32_t)
    constexpr VkDeviceSize histogramSize = 256 * sizeof(uint32_t);
    VkBufferCreateInfo histogramBufInfo{};
    histogramBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    histogramBufInfo.size  = histogramSize;
    histogramBufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo histogramAllocInfo{};
    histogramAllocInfo.usage          = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    histogramAllocInfo.flags         = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VK_CHECK(vmaCreateBuffer(allocator, &histogramBufInfo, &histogramAllocInfo,
                             &mHistogramBuffer, &mHistogramAlloc, nullptr));

    // Create exposure buffer: sizeof(float)
    constexpr VkDeviceSize exposureSize = sizeof(float);
    VkBufferCreateInfo exposureBufInfo{};
    exposureBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    exposureBufInfo.size  = exposureSize;
    exposureBufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo exposureAllocInfo{};
    exposureAllocInfo.usage  = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    exposureAllocInfo.flags  = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VK_CHECK(vmaCreateBuffer(allocator, &exposureBufInfo, &exposureAllocInfo,
                             &mExposureBuffer, &mExposureAlloc, nullptr));

    // Initialize exposure to 1.0 via host mapping (if possible) or defer to first Dispatch
    VmaAllocationInfo exposureAllocInfoOut{};
    vmaGetAllocationInfo(allocator, mExposureAlloc, &exposureAllocInfoOut);
    if (exposureAllocInfoOut.pMappedData) {
        const float initialExposure = 1.0f;
        std::memcpy(exposureAllocInfoOut.pMappedData, &initialExposure, sizeof(float));
    }

    // Update average descriptor set (histogram + exposure buffers are fixed)
    VkDescriptorBufferInfo histogramBufDesc{};
    histogramBufDesc.buffer = mHistogramBuffer;
    histogramBufDesc.offset = 0;
    histogramBufDesc.range  = histogramSize;

    VkDescriptorBufferInfo exposureBufDesc{};
    exposureBufDesc.buffer = mExposureBuffer;
    exposureBufDesc.offset = 0;
    exposureBufDesc.range  = exposureSize;

    VkWriteDescriptorSet averageWrites[2]{};
    averageWrites[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    averageWrites[0].dstSet          = mAverageSet;
    averageWrites[0].dstBinding      = 0;
    averageWrites[0].descriptorCount  = 1;
    averageWrites[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    averageWrites[0].pBufferInfo      = &histogramBufDesc;

    averageWrites[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    averageWrites[1].dstSet          = mAverageSet;
    averageWrites[1].dstBinding      = 1;
    averageWrites[1].descriptorCount  = 1;
    averageWrites[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    averageWrites[1].pBufferInfo      = &exposureBufDesc;

    vkUpdateDescriptorSets(device, 2, averageWrites, 0, nullptr);

    LOG_INFO("AutoExposure initialized");
}

void AutoExposure::Shutdown(VkDevice device, VmaAllocator allocator) {
    if (mLinearSampler) {
        vkDestroySampler(device, mLinearSampler, nullptr);
        mLinearSampler = VK_NULL_HANDLE;
    }
    if (mHistogramBuffer) {
        vmaDestroyBuffer(allocator, mHistogramBuffer, mHistogramAlloc);
        mHistogramBuffer  = VK_NULL_HANDLE;
        mHistogramAlloc   = VK_NULL_HANDLE;
    }
    if (mExposureBuffer) {
        vmaDestroyBuffer(allocator, mExposureBuffer, mExposureAlloc);
        mExposureBuffer   = VK_NULL_HANDLE;
        mExposureAlloc    = VK_NULL_HANDLE;
    }
    if (mDescPool) {
        vkDestroyDescriptorPool(device, mDescPool, nullptr);
        mDescPool = VK_NULL_HANDLE;
    }
    if (mHistogramPipeline) {
        vkDestroyPipeline(device, mHistogramPipeline, nullptr);
        mHistogramPipeline = VK_NULL_HANDLE;
    }
    if (mAveragePipeline) {
        vkDestroyPipeline(device, mAveragePipeline, nullptr);
        mAveragePipeline = VK_NULL_HANDLE;
    }
    if (mHistogramPipeLayout) {
        vkDestroyPipelineLayout(device, mHistogramPipeLayout, nullptr);
        mHistogramPipeLayout = VK_NULL_HANDLE;
    }
    if (mAveragePipeLayout) {
        vkDestroyPipelineLayout(device, mAveragePipeLayout, nullptr);
        mAveragePipeLayout = VK_NULL_HANDLE;
    }
    if (mHistogramLayout) {
        vkDestroyDescriptorSetLayout(device, mHistogramLayout, nullptr);
        mHistogramLayout = VK_NULL_HANDLE;
    }
    if (mAverageLayout) {
        vkDestroyDescriptorSetLayout(device, mAverageLayout, nullptr);
        mAverageLayout = VK_NULL_HANDLE;
    }
    mDevice = VK_NULL_HANDLE;
}

void AutoExposure::Dispatch(VkCommandBuffer cmd, VkImageView hdrView, VkSampler hdrSampler,
                            uint32_t width, uint32_t height,
                            float minLogLum, float maxLogLum,
                            float deltaTime, float adaptSpeed) {
    static bool sFirstDispatch = true;

    // Update histogram descriptor with provided hdrView + hdrSampler
    VkSampler samplerToUse = (hdrSampler != VK_NULL_HANDLE) ? hdrSampler : mLinearSampler;
    VkDescriptorImageInfo hdrImageInfo{};
    hdrImageInfo.sampler     = samplerToUse;
    hdrImageInfo.imageView   = hdrView;
    hdrImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo histogramBufInfo{};
    histogramBufInfo.buffer = mHistogramBuffer;
    histogramBufInfo.offset = 0;
    histogramBufInfo.range  = 256 * sizeof(uint32_t);

    VkWriteDescriptorSet histogramWrites[2]{};
    histogramWrites[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    histogramWrites[0].dstSet          = mHistogramSet;
    histogramWrites[0].dstBinding       = 0;
    histogramWrites[0].descriptorCount  = 1;
    histogramWrites[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    histogramWrites[0].pImageInfo       = &hdrImageInfo;

    histogramWrites[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    histogramWrites[1].dstSet          = mHistogramSet;
    histogramWrites[1].dstBinding       = 1;
    histogramWrites[1].descriptorCount  = 1;
    histogramWrites[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    histogramWrites[1].pBufferInfo      = &histogramBufInfo;

    vkUpdateDescriptorSets(mDevice, 2, histogramWrites, 0, nullptr);

    // Initialize exposure to 1.0 on first dispatch (in case allocation wasn't host-visible)
    if (sFirstDispatch) {
        vkCmdFillBuffer(cmd, mExposureBuffer, 0, sizeof(float), 0x3F800000u); // 1.0f as uint
        TransitionBuffer(cmd, mExposureBuffer, 0, sizeof(float),
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        sFirstDispatch = false;
    }

    // Clear histogram buffer
    vkCmdFillBuffer(cmd, mHistogramBuffer, 0, 256 * sizeof(uint32_t), 0);

    // Barrier: transfer write -> compute read/write for histogram
    TransitionBuffer(cmd, mHistogramBuffer, 0, 256 * sizeof(uint32_t),
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // Dispatch histogram (local_size 16x16)
    const float logLumRange = maxLogLum - minLogLum;
    const float invLogLumRange = (logLumRange > 1e-6f) ? (1.0f / logLumRange) : 1.0f;

    HistogramPushConstants histogramPC{};
    histogramPC.minLogLum      = minLogLum;
    histogramPC.invLogLumRange = invLogLumRange;
    histogramPC.inputSizeX     = width;
    histogramPC.inputSizeY    = height;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mHistogramPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            mHistogramPipeLayout, 0, 1, &mHistogramSet, 0, nullptr);
    vkCmdPushConstants(cmd, mHistogramPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(HistogramPushConstants), &histogramPC);

    const uint32_t groupsX = (width + 15) / 16;
    const uint32_t groupsY = (height + 15) / 16;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);

    // Barrier: histogram write -> average read
    TransitionBuffer(cmd, mHistogramBuffer, 0, 256 * sizeof(uint32_t),
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    TransitionBuffer(cmd, mExposureBuffer, 0, sizeof(float),
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // Dispatch average (local_size 256, 1 group)
    const uint32_t pixelCount = width * height;

    AveragePushConstants averagePC{};
    averagePC.minLogLum   = minLogLum;
    averagePC.logLumRange = logLumRange;
    averagePC.deltaTime   = deltaTime;
    averagePC.adaptSpeed = adaptSpeed;
    averagePC.pixelCount  = pixelCount;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mAveragePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            mAveragePipeLayout, 0, 1, &mAverageSet, 0, nullptr);
    vkCmdPushConstants(cmd, mAveragePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(AveragePushConstants), &averagePC);

    vkCmdDispatch(cmd, 1, 1, 1);

    // Barrier: average write -> subsequent read
    TransitionBuffer(cmd, mExposureBuffer, 0, sizeof(float),
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE);
}
