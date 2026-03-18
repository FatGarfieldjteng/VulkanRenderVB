#include "RayTracing/PathTracer.h"
#include "Resource/TransferManager.h"
#include "Resource/DescriptorManager.h"
#include "Core/Logger.h"

#include <cmath>
#include <cstring>
#include <algorithm>

void PathTracer::Initialize(VkDevice device, VmaAllocator allocator,
                             ShaderManager& shaders,
                             const TransferManager& transfer,
                             const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rtProps,
                             uint32_t width, uint32_t height) {
    mDevice    = device;
    mAllocator = allocator;
    mTransfer  = &transfer;
    mShaders   = &shaders;
    mRTProps   = rtProps;
    mWidth     = width;
    mHeight    = height;

    CreateImages(width, height);
    CreateDescriptors();

    LOG_INFO("PathTracer initialized: {}x{}", width, height);
}

void PathTracer::Shutdown(VkDevice device, VmaAllocator allocator) {
    mPipeline.Destroy(device);
    mSBT.Destroy(allocator);

    mColorOutput.Destroy(allocator, device);
    mAlbedoOutput.Destroy(allocator, device);
    mNormalOutput.Destroy(allocator, device);
    mDepthOutput.Destroy(allocator, device);
    mMotionOutput.Destroy(allocator, device);
    mAccumBuffer.Destroy(allocator, device);
    mInstanceInfoBuffer.Destroy(allocator);
    mFrameUBO.Destroy(allocator);

    if (mPipelineLayout)    { vkDestroyPipelineLayout(device, mPipelineLayout, nullptr);    mPipelineLayout = VK_NULL_HANDLE; }
    if (mSceneDescLayout)   { vkDestroyDescriptorSetLayout(device, mSceneDescLayout, nullptr); mSceneDescLayout = VK_NULL_HANDLE; }
    if (mDescPool)          { vkDestroyDescriptorPool(device, mDescPool, nullptr);          mDescPool = VK_NULL_HANDLE; }

    mSceneDescSet = VK_NULL_HANDLE;
    mSceneDirty = true;
    mAccumFrames = 0;
}

void PathTracer::Resize(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h) {
    if (w == mWidth && h == mHeight) return;
    mWidth = w; mHeight = h;

    mColorOutput.Destroy(allocator, device);
    mAlbedoOutput.Destroy(allocator, device);
    mNormalOutput.Destroy(allocator, device);
    mDepthOutput.Destroy(allocator, device);
    mMotionOutput.Destroy(allocator, device);
    mAccumBuffer.Destroy(allocator, device);

    CreateImages(w, h);
    UpdateImageDescriptors();
    mAccumFrames = 0;
}

void PathTracer::UpdateImageDescriptors() {
    if (mSceneDescSet == VK_NULL_HANDLE) return;

    VkDescriptorImageInfo colorInfo{VK_NULL_HANDLE, mColorOutput.GetView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo accumInfo{VK_NULL_HANDLE, mAccumBuffer.GetView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo normalInfo{VK_NULL_HANDLE, mNormalOutput.GetView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo albedoInfo{VK_NULL_HANDLE, mAlbedoOutput.GetView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo depthInfo{VK_NULL_HANDLE, mDepthOutput.GetView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo motionInfo{VK_NULL_HANDLE, mMotionOutput.GetView(), VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet writes[6] = {};
    auto makeWrite = [&](int idx, uint32_t binding, VkDescriptorImageInfo* info) {
        writes[idx] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[idx].dstSet = mSceneDescSet; writes[idx].dstBinding = binding;
        writes[idx].descriptorCount = 1; writes[idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[idx].pImageInfo = info;
    };
    makeWrite(0, 1, &colorInfo);
    makeWrite(1, 2, &accumInfo);
    makeWrite(2, 3, &normalInfo);
    makeWrite(3, 4, &albedoInfo);
    makeWrite(4, 5, &depthInfo);
    makeWrite(5, 13, &motionInfo);

    vkUpdateDescriptorSets(mDevice, 6, writes, 0, nullptr);
}

void PathTracer::CreateImages(uint32_t w, uint32_t h) {
    mColorOutput.CreateStorageImage(mAllocator, mDevice, w, h, VK_FORMAT_R32G32B32A32_SFLOAT);
    mAlbedoOutput.CreateStorageImage(mAllocator, mDevice, w, h, VK_FORMAT_R8G8B8A8_UNORM);
    mNormalOutput.CreateStorageImage(mAllocator, mDevice, w, h, VK_FORMAT_R16G16B16A16_SFLOAT);
    mDepthOutput.CreateStorageImage(mAllocator, mDevice, w, h, VK_FORMAT_R32_SFLOAT);
    mMotionOutput.CreateStorageImage(mAllocator, mDevice, w, h, VK_FORMAT_R32G32_SFLOAT);
    mAccumBuffer.CreateStorageImage(mAllocator, mDevice, w, h, VK_FORMAT_R32G32B32A32_SFLOAT);
}

void PathTracer::CreateDescriptors() {
    // Scene descriptor set layout (set 0)
    // Bindings 0-5: TLAS, color, accum, normal, albedo, depth
    // Bindings 6-9: vertex, index, material, instance SSBOs
    // Binding 10: env map, 11: BRDF LUT, 12: irradiance
    // Binding 13: motion output
    // Binding 14: frame UBO (viewProj + prevViewProj)
    VkDescriptorSetLayoutBinding bindings[] = {
        {0,  VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr},
        {1,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr},
        {2,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr},
        {3,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr},
        {4,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr},
        {5,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr},
        {6,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR, nullptr},
        {7,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR, nullptr},
        {8,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR, nullptr},
        {9,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR, nullptr},
        {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_MISS_BIT_KHR, nullptr},
        {11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr},
        {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr},
        {13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr},
        {14, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr},
    };

    VkDescriptorSetLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutCI.bindingCount = static_cast<uint32_t>(std::size(bindings));
    layoutCI.pBindings    = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(mDevice, &layoutCI, nullptr, &mSceneDescLayout));

    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
    };
    VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.maxSets       = 1;
    poolCI.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
    poolCI.pPoolSizes    = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(mDevice, &poolCI, nullptr, &mDescPool));

    VkDescriptorSetAllocateInfo allocCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocCI.descriptorPool     = mDescPool;
    allocCI.descriptorSetCount = 1;
    allocCI.pSetLayouts        = &mSceneDescLayout;
    VK_CHECK(vkAllocateDescriptorSets(mDevice, &allocCI, &mSceneDescSet));

    // Create the per-frame UBO (host-visible for easy updates)
    mFrameUBO.CreateHostVisible(mAllocator, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                sizeof(FrameUBOData));
}

void PathTracer::CreatePipeline(ShaderManager& shaders) {
    mPipeline.Destroy(mDevice);
    mSBT.Destroy(mAllocator);

    auto raygenMod  = shaders.GetOrLoad("shaders/pt_raygen.rgen.spv");
    auto missMod    = shaders.GetOrLoad("shaders/pt_miss.rmiss.spv");
    auto shadowMiss = shaders.GetOrLoad("shaders/pt_shadow_miss.rmiss.spv");
    auto chitMod    = shaders.GetOrLoad("shaders/pt_closesthit.rchit.spv");
    auto ahitMod    = shaders.GetOrLoad("shaders/pt_anyhit.rahit.spv");

    uint32_t raygenIdx    = mPipeline.AddStage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, raygenMod);
    uint32_t missIdx      = mPipeline.AddStage(VK_SHADER_STAGE_MISS_BIT_KHR, missMod);
    uint32_t shadowMissIdx = mPipeline.AddStage(VK_SHADER_STAGE_MISS_BIT_KHR, shadowMiss);
    uint32_t chitIdx      = mPipeline.AddStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, chitMod);
    uint32_t ahitIdx      = mPipeline.AddStage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR, ahitMod);

    mPipeline.AddRayGenGroup(raygenIdx);
    mPipeline.AddMissGroup(missIdx);
    mPipeline.AddMissGroup(shadowMissIdx);

    // Single hit group pair for all materials (SBT simplification)
    mPipeline.AddHitGroup(chitIdx, ahitIdx);                    // primary
    mPipeline.AddHitGroup(VK_SHADER_UNUSED_KHR, ahitIdx);      // shadow

    if (mPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(mDevice, mPipelineLayout, nullptr);
        mPipelineLayout = VK_NULL_HANDLE;
    }
    {
        VkDescriptorSetLayout layouts[] = {mSceneDescLayout, mBindlessDescLayout};
        VkPushConstantRange pcRange{
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
            0, sizeof(PushConstants)
        };
        VkPipelineLayoutCreateInfo plCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plCI.setLayoutCount         = mBindlessDescLayout ? 2u : 1u;
        plCI.pSetLayouts            = layouts;
        plCI.pushConstantRangeCount = 1;
        plCI.pPushConstantRanges    = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(mDevice, &plCI, nullptr, &mPipelineLayout));
    }

    mPipeline.Build(mDevice, mPipelineLayout, 1);

    // Single hit group pair — no per-material SBT data needed
    mSBT.Build(mDevice, mAllocator, mPipeline, mRTProps,
               1, 2, 2, 0, 0);
    mSBT.UploadToGPU(mAllocator, mDevice, *mTransfer);

    LOG_INFO("PathTracer pipeline created: single hit group pair (SBT simplified)");
}

void PathTracer::UpdateScene(VkDevice device, VmaAllocator allocator,
                              const TransferManager& transfer,
                              VkAccelerationStructureKHR tlas,
                              const MeshPool& meshPool,
                              const std::vector<RTInstanceInfo>& instanceInfos,
                              VkBuffer materialSSBO, VkDeviceSize materialSSBOSize,
                              VkDescriptorSet bindlessTexSet,
                              VkDescriptorSetLayout bindlessTexLayout,
                              VkImageView envCubeView, VkSampler cubeSampler,
                              VkImageView irradianceView,
                              VkImageView brdfLutView, VkSampler lutSampler) {
    mBindlessDescLayout = bindlessTexLayout;
    mBindlessDescSet    = bindlessTexSet;

    // Upload instance info buffer
    mInstanceInfoBuffer.Destroy(allocator);
    if (!instanceInfos.empty()) {
        mInstanceInfoBuffer.CreateDeviceLocal(allocator, transfer,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            instanceInfos.data(), instanceInfos.size() * sizeof(RTInstanceInfo));
    }

    // Build pipeline once (no per-material rebuild needed)
    if (mShaders && mPipeline.GetPipeline() == VK_NULL_HANDLE) {
        CreatePipeline(*mShaders);
    }

    // Update descriptor set
    VkWriteDescriptorSetAccelerationStructureKHR asWrite{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures    = &tlas;

    VkDescriptorImageInfo colorInfo{VK_NULL_HANDLE, mColorOutput.GetView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo accumInfo{VK_NULL_HANDLE, mAccumBuffer.GetView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo normalInfo{VK_NULL_HANDLE, mNormalOutput.GetView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo albedoInfo{VK_NULL_HANDLE, mAlbedoOutput.GetView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo depthInfo{VK_NULL_HANDLE, mDepthOutput.GetView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo motionInfo{VK_NULL_HANDLE, mMotionOutput.GetView(), VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorBufferInfo vertBufInfo{meshPool.GetVertexBuffer(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo idxBufInfo{meshPool.GetIndexBuffer(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo matBufInfo{materialSSBO, 0, materialSSBOSize};
    VkDescriptorBufferInfo instBufInfo{mInstanceInfoBuffer.GetHandle(), 0, VK_WHOLE_SIZE};

    VkDescriptorImageInfo envInfo{cubeSampler, envCubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo brdfInfo{lutSampler, brdfLutView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo irrInfo{cubeSampler, irradianceView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    VkDescriptorBufferInfo uboInfo{mFrameUBO.GetHandle(), 0, sizeof(FrameUBOData)};

    VkWriteDescriptorSet writes[16] = {};

    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[0].pNext = &asWrite;
    writes[0].dstSet = mSceneDescSet; writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    auto makeImageWrite = [&](int idx, uint32_t binding, VkDescriptorImageInfo* info) {
        writes[idx] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[idx].dstSet = mSceneDescSet; writes[idx].dstBinding = binding;
        writes[idx].descriptorCount = 1; writes[idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[idx].pImageInfo = info;
    };

    makeImageWrite(1, 1, &colorInfo);
    makeImageWrite(2, 2, &accumInfo);
    makeImageWrite(3, 3, &normalInfo);
    makeImageWrite(4, 4, &albedoInfo);
    makeImageWrite(5, 5, &depthInfo);
    makeImageWrite(6, 13, &motionInfo);

    auto makeBufferWrite = [&](int idx, uint32_t binding, VkDescriptorBufferInfo* info) {
        writes[idx] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[idx].dstSet = mSceneDescSet; writes[idx].dstBinding = binding;
        writes[idx].descriptorCount = 1; writes[idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[idx].pBufferInfo = info;
    };

    makeBufferWrite(7, 6, &vertBufInfo);
    makeBufferWrite(8, 7, &idxBufInfo);
    makeBufferWrite(9, 8, &matBufInfo);
    makeBufferWrite(10, 9, &instBufInfo);

    auto makeSamplerWrite = [&](int idx, uint32_t binding, VkDescriptorImageInfo* info) {
        writes[idx] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[idx].dstSet = mSceneDescSet; writes[idx].dstBinding = binding;
        writes[idx].descriptorCount = 1; writes[idx].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[idx].pImageInfo = info;
    };

    makeSamplerWrite(11, 10, &envInfo);
    makeSamplerWrite(12, 11, &brdfInfo);
    makeSamplerWrite(13, 12, &irrInfo);

    // UBO write
    writes[14] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[14].dstSet = mSceneDescSet; writes[14].dstBinding = 14;
    writes[14].descriptorCount = 1; writes[14].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[14].pBufferInfo = &uboInfo;

    vkUpdateDescriptorSets(device, 15, writes, 0, nullptr);

    mSceneDirty = false;
    mAccumFrames = 0;
}

void PathTracer::Trace(VkCommandBuffer cmd,
                        const glm::mat4& invViewProj,
                        const glm::mat4& viewProj,
                        const glm::vec3& cameraPos,
                        const glm::vec3& sunDir,
                        const glm::vec3& sunColor,
                        float sunIntensity,
                        float lightRadius,
                        bool /*denoiserEnabled*/) {
    if (mPipeline.GetPipeline() == VK_NULL_HANDLE) return;

    mAccumReset = false;
    // Discard temporal info on camera move/zoom: any viewProj change invalidates history
    // Use 1e-4 (not 1e-5) to avoid spurious resets from floating-point drift; NRD needs stable history for temporal denoising
    float maxDiff = 0.0f;
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            maxDiff = std::max(maxDiff, std::abs(viewProj[c][r] - mPrevViewProj[c][r]));
    if (maxDiff > 1e-4f) {
        mAccumFrames = 0;
        mAccumReset  = true;
    }

    // Update frame UBO with current and previous view-projection matrices
    // First frame: mPrevViewProj is identity → use viewProj to avoid wrong motion vectors
    {
        FrameUBOData uboData{};
        uboData.viewProj     = viewProj;
        glm::mat4 prevVP     = (mPrevViewProj == glm::mat4(1.0f)) ? viewProj : mPrevViewProj;
        uboData.prevViewProj = prevVP;
        std::memcpy(mFrameUBO.GetMappedData(), &uboData, sizeof(uboData));
    }

    // Transition images to GENERAL (use correct oldLayout: UNDEFINED on first frame, GENERAL thereafter)
    VkImageLayout oldLayout = (mAccumFrames > 0) ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageMemoryBarrier2 barriers[6]{};
    auto makeBarrier = [oldLayout](VkImage image) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        b.srcAccessMask = (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) ? 0u : (VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT);
        b.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        b.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
        b.oldLayout     = oldLayout;
        b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        b.image         = image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        return b;
    };
    barriers[0] = makeBarrier(mColorOutput.GetImage());
    barriers[1] = makeBarrier(mAccumBuffer.GetImage());
    barriers[2] = makeBarrier(mNormalOutput.GetImage());
    barriers[3] = makeBarrier(mAlbedoOutput.GetImage());
    barriers[4] = makeBarrier(mDepthOutput.GetImage());
    barriers[5] = makeBarrier(mMotionOutput.GetImage());

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 6;
    dep.pImageMemoryBarriers    = barriers;
    vkCmdPipelineBarrier2(cmd, &dep);

    if (mAccumReset) {
        VkClearColorValue clearVal = {{0.0f, 0.0f, 0.0f, 0.0f}};
        VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, mAccumBuffer.GetImage(), VK_IMAGE_LAYOUT_GENERAL,
                             &clearVal, 1, &range);

        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo clearDep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        clearDep.memoryBarrierCount = 1;
        clearDep.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cmd, &clearDep);
    }

    PushConstants pc{};
    pc.invViewProj       = invViewProj;
    pc.cameraPosAndFrame = glm::vec4(cameraPos, float(mSampleOffset));
    pc.sunDirAndRadius   = glm::vec4(sunDir, lightRadius);
    pc.sunColorIntensity = glm::vec4(sunColor, sunIntensity);
    pc.params            = glm::uvec4(maxBounces, mSampleOffset, enableMIS ? 1 : 0, mAccumFrames);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, mPipeline.GetPipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            mPipelineLayout, 0, 1, &mSceneDescSet, 0, nullptr);
    if (mBindlessDescSet != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                mPipelineLayout, 1, 1, &mBindlessDescSet, 0, nullptr);
    }
    vkCmdPushConstants(cmd, mPipelineLayout,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
        0, sizeof(PushConstants), &pc);

    auto rayGenRegion  = mSBT.GetRayGenRegion();
    auto missRegion    = mSBT.GetMissRegion();
    auto hitRegion     = mSBT.GetHitRegion();
    auto callRegion    = mSBT.GetCallableRegion();

    vkCmdTraceRaysKHR(cmd, &rayGenRegion, &missRegion, &hitRegion, &callRegion,
                      mWidth, mHeight, 1);

    if (progressive) mAccumFrames++;
    mSampleOffset++;
    mPrevViewProj = viewProj;
}
