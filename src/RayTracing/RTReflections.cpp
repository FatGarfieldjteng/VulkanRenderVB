#include "RayTracing/RTReflections.h"
#include "Core/Logger.h"

static void CreateR16FImage(VmaAllocator allocator, VkDevice device,
                            uint32_t w, uint32_t h, VkFormat format, VulkanImage& out) {
    VkImageCreateInfo imgCI{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = format;
    imgCI.extent        = {w, h, 1};
    imgCI.mipLevels     = 1;
    imgCI.arrayLayers   = 1;
    imgCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgCI.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkImage image;
    VmaAllocation alloc;
    VK_CHECK(vmaCreateImage(allocator, &imgCI, &allocCI, &image, &alloc, nullptr));

    VkImageViewCreateInfo viewCI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewCI.image    = image;
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format   = format;
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view;
    VK_CHECK(vkCreateImageView(device, &viewCI, nullptr, &view));

    out.SetHandles(image, view, alloc);
}

void RTReflections::CreateImages(uint32_t width, uint32_t height) {
    for (int i = 0; i < 2; i++)
        CreateR16FImage(mAllocator, mDevice, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, mReflImage[i]);
}

void RTReflections::Initialize(VkDevice device, VmaAllocator allocator,
                                ShaderManager& shaders, uint32_t width, uint32_t height) {
    mDevice    = device;
    mAllocator = allocator;
    mWidth     = width;
    mHeight    = height;
    mDescriptorsDirty = true;

    CreateImages(width, height);

    VkSamplerCreateInfo samplerCI{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerCI.magFilter    = VK_FILTER_LINEAR;
    samplerCI.minFilter    = VK_FILTER_LINEAR;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device, &samplerCI, nullptr, &mSampler));

    CreateDescriptors();

    // Trace pipeline
    {
        VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TracePushConstants)};
        VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutCI.setLayoutCount         = 1;
        layoutCI.pSetLayouts            = &mTraceDescLayout;
        layoutCI.pushConstantRangeCount = 1;
        layoutCI.pPushConstantRanges    = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(device, &layoutCI, nullptr, &mTracePipeLayout));

        VkShaderModule mod = shaders.GetOrLoad("shaders/rt_reflections.comp.spv");
        VkComputePipelineCreateInfo pipeCI{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeCI.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeCI.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeCI.stage.module = mod;
        pipeCI.stage.pName  = "main";
        pipeCI.layout       = mTracePipeLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &mTracePipeline));
    }

    // Denoise pipeline
    {
        VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DenoisePushConstants)};
        VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutCI.setLayoutCount         = 1;
        layoutCI.pSetLayouts            = &mDenoiseDescLayout;
        layoutCI.pushConstantRangeCount = 1;
        layoutCI.pPushConstantRanges    = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(device, &layoutCI, nullptr, &mDenoisePipeLayout));

        VkShaderModule mod = shaders.GetOrLoad("shaders/rt_reflect_denoise.comp.spv");
        VkComputePipelineCreateInfo pipeCI{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeCI.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeCI.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeCI.stage.module = mod;
        pipeCI.stage.pName  = "main";
        pipeCI.layout       = mDenoisePipeLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &mDenoisePipeline));
    }

    LOG_INFO("RTReflections initialized ({}x{})", width, height);
}

void RTReflections::CreateDescriptors() {
    // Trace: TLAS, output, depth (3 bindings, no prev image)
    {
        VkDescriptorSetLayoutBinding bindings[3] = {};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT};

        VkDescriptorSetLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutCI.bindingCount = 3;
        layoutCI.pBindings    = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(mDevice, &layoutCI, nullptr, &mTraceDescLayout));

        VkDescriptorPoolSize poolSizes[] = {
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        };
        VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolCI.maxSets       = 1;
        poolCI.poolSizeCount = 3;
        poolCI.pPoolSizes    = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(mDevice, &poolCI, nullptr, &mTraceDescPool));

        VkDescriptorSetAllocateInfo allocCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocCI.descriptorPool     = mTraceDescPool;
        allocCI.descriptorSetCount = 1;
        allocCI.pSetLayouts        = &mTraceDescLayout;
        VK_CHECK(vkAllocateDescriptorSets(mDevice, &allocCI, &mTraceDescSet));
    }

    // Denoise: input, output, depth
    {
        VkDescriptorSetLayoutBinding bindings[3] = {};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT};

        VkDescriptorSetLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutCI.bindingCount = 3;
        layoutCI.pBindings    = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(mDevice, &layoutCI, nullptr, &mDenoiseDescLayout));

        VkDescriptorPoolSize poolSizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
        };
        VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolCI.maxSets       = 3;
        poolCI.poolSizeCount = 2;
        poolCI.pPoolSizes    = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(mDevice, &poolCI, nullptr, &mDenoiseDescPool));

        for (int i = 0; i < 3; i++) {
            VkDescriptorSetAllocateInfo allocCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocCI.descriptorPool     = mDenoiseDescPool;
            allocCI.descriptorSetCount = 1;
            allocCI.pSetLayouts        = &mDenoiseDescLayout;
            VK_CHECK(vkAllocateDescriptorSets(mDevice, &allocCI, &mDenoiseDescSets[i]));
        }
    }
}

void RTReflections::UpdateTraceDescriptors(VkAccelerationStructureKHR tlas,
                                            VkImageView depthView, VkSampler depthSampler) {
    VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
    asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures    = &tlas;

    VkDescriptorImageInfo outputInfo{VK_NULL_HANDLE, mReflImage[0].GetView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo depthInfo{depthSampler, depthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    VkWriteDescriptorSet writes[3] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &asWrite;
    writes[0].dstSet = mTraceDescSet; writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mTraceDescSet,
                  1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mTraceDescSet,
                  2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo};

    vkUpdateDescriptorSets(mDevice, 3, writes, 0, nullptr);
}

void RTReflections::UpdateDenoiseDescriptors(VkImageView depthView, VkSampler depthSampler) {
    VkDescriptorImageInfo depthInfo{depthSampler, depthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    for (int iter = 0; iter < 3; iter++) {
        int srcIdx = (iter % 2 == 0) ? 0 : 1;
        int dstIdx = 1 - srcIdx;

        VkDescriptorImageInfo srcInfo{VK_NULL_HANDLE, mReflImage[srcIdx].GetView(), VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo dstInfo{VK_NULL_HANDLE, mReflImage[dstIdx].GetView(), VK_IMAGE_LAYOUT_GENERAL};

        VkWriteDescriptorSet writes[3] = {};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mDenoiseDescSets[iter],
                      0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &srcInfo};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mDenoiseDescSets[iter],
                      1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dstInfo};
        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mDenoiseDescSets[iter],
                      2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo};
        vkUpdateDescriptorSets(mDevice, 3, writes, 0, nullptr);
    }
}

void RTReflections::Dispatch(VkCommandBuffer cmd, VkAccelerationStructureKHR tlas,
                              VkImageView depthView, VkSampler depthSampler,
                              const glm::mat4& invViewProj,
                              const glm::vec3& cameraPos, float roughness) {
    if (mDescriptorsDirty) {
        UpdateTraceDescriptors(tlas, depthView, depthSampler);
        UpdateDenoiseDescriptors(depthView, depthSampler);
        mDescriptorsDirty = false;
    }

    // Transition both images to GENERAL (discard old content each frame)
    VkImageMemoryBarrier2 barriers[2] = {};
    for (int i = 0; i < 2; i++) {
        barriers[i].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[i].srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barriers[i].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barriers[i].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
        barriers[i].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[i].newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barriers[i].image         = mReflImage[i].GetImage();
        barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 2;
    dep.pImageMemoryBarriers    = barriers;
    vkCmdPipelineBarrier2(cmd, &dep);

    TracePushConstants pc{};
    pc.invViewProj = invViewProj;
    pc.cameraPos   = glm::vec4(cameraPos, 0.0f);
    pc.resolution  = {mWidth, mHeight};
    pc.roughness   = roughness;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mTracePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mTracePipeLayout, 0, 1, &mTraceDescSet, 0, nullptr);
    vkCmdPushConstants(cmd, mTracePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (mWidth + 7) / 8, (mHeight + 7) / 8, 1);

    {
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        VkDependencyInfo dep2{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep2.memoryBarrierCount = 1;
        dep2.pMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep2);
    }
}

void RTReflections::Denoise(VkCommandBuffer cmd, VkImageView depthView, VkSampler depthSampler,
                              const glm::mat4& invViewProj) {

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mDenoisePipeline);

    constexpr int kIterations = 3;
    int stepSizes[] = {1, 2, 4};
    for (int iter = 0; iter < kIterations; iter++) {
        DenoisePushConstants pc{};
        pc.invViewProj = invViewProj;
        pc.resolution  = {mWidth, mHeight};
        pc.stepSize    = stepSizes[iter];
        pc.depthSigma  = 0.01f;
        pc.normalSigma = 128.0f;
        pc.colorSigma  = 0.5f;

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mDenoisePipeLayout,
                                0, 1, &mDenoiseDescSets[iter], 0, nullptr);
        vkCmdPushConstants(cmd, mDenoisePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (mWidth + 7) / 8, (mHeight + 7) / 8, 1);

        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    mOutputIdx = (kIterations % 2 == 0) ? 0 : 1;
}

void RTReflections::Resize(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height) {
    if (width == mWidth && height == mHeight) return;
    mDescriptorsDirty = true;
    for (int i = 0; i < 2; i++)
        mReflImage[i].Destroy(allocator, device);
    mWidth  = width;
    mHeight = height;
    CreateImages(width, height);
}

void RTReflections::Shutdown(VkDevice device, VmaAllocator allocator) {
    for (int i = 0; i < 2; i++)
        mReflImage[i].Destroy(allocator, device);

    if (mSampler)             { vkDestroySampler(device, mSampler, nullptr);                       mSampler = VK_NULL_HANDLE; }
    if (mTracePipeline)       { vkDestroyPipeline(device, mTracePipeline, nullptr);                mTracePipeline = VK_NULL_HANDLE; }
    if (mTracePipeLayout)     { vkDestroyPipelineLayout(device, mTracePipeLayout, nullptr);        mTracePipeLayout = VK_NULL_HANDLE; }
    if (mTraceDescPool)       { vkDestroyDescriptorPool(device, mTraceDescPool, nullptr);          mTraceDescPool = VK_NULL_HANDLE; }
    if (mTraceDescLayout)     { vkDestroyDescriptorSetLayout(device, mTraceDescLayout, nullptr);   mTraceDescLayout = VK_NULL_HANDLE; }
    if (mDenoisePipeline)     { vkDestroyPipeline(device, mDenoisePipeline, nullptr);              mDenoisePipeline = VK_NULL_HANDLE; }
    if (mDenoisePipeLayout)   { vkDestroyPipelineLayout(device, mDenoisePipeLayout, nullptr);      mDenoisePipeLayout = VK_NULL_HANDLE; }
    if (mDenoiseDescPool)     { vkDestroyDescriptorPool(device, mDenoiseDescPool, nullptr);        mDenoiseDescPool = VK_NULL_HANDLE; }
    if (mDenoiseDescLayout)   { vkDestroyDescriptorSetLayout(device, mDenoiseDescLayout, nullptr); mDenoiseDescLayout = VK_NULL_HANDLE; }
    mTraceDescSet = VK_NULL_HANDLE;
    for (int i = 0; i < 3; i++) mDenoiseDescSets[i] = VK_NULL_HANDLE;
    mDescriptorsDirty = true;
}
