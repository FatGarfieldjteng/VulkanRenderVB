#include "RayTracing/NRDDenoiser.h"
#include "Core/Logger.h"
#include "RHI/VulkanUtils.h"

#include <volk.h>

void NRDDenoiser::TransitionOutputForSampling(VkCommandBuffer cmd) const {
    TransitionImage(cmd, mOutput.GetImage(),
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
#include <cstring>
#include <array>

// NRD + NRI (include order: NRIRayTracing before NRIWrapperVK)
#include "NRI.h"
#include "Extensions/NRIHelper.h"
#include "Extensions/NRIRayTracing.h"
#include "Extensions/NRIWrapperVK.h"
#include "NRD.h"
#include "NRDDescs.h"
#include "NRDSettings.h"
#include "NRDIntegration.hpp"

NRDDenoiser::NRDDenoiser() = default;

NRDDenoiser::~NRDDenoiser() {
    if (mNRD) {
        mNRD->Destroy();
        delete mNRD;
        mNRD = nullptr;
    }
}

void NRDDenoiser::Initialize(VkInstance instance, VkDevice device, VmaAllocator allocator,
                             ShaderManager& shaders,
                             VkPhysicalDevice physicalDevice,
                             VkQueue queue, uint32_t queueFamilyIndex,
                             uint32_t width, uint32_t height) {
    mInstance         = instance;
    mDevice           = device;
    mAllocator        = allocator;
    mPhysicalDevice   = physicalDevice;
    mQueue            = queue;
    mQueueFamilyIndex = queueFamilyIndex;
    mShaders          = &shaders;
    mWidth            = width;
    mHeight           = height;

    // Create NRI device from Vulkan (instance required for NRI to resolve vkGetDeviceProcAddr)
    nri::DeviceCreationVKDesc deviceCreationVKDesc = {};
    deviceCreationVKDesc.vkDevice          = device;
    deviceCreationVKDesc.vkPhysicalDevice   = physicalDevice;
    deviceCreationVKDesc.vkInstance         = instance;
    deviceCreationVKDesc.minorVersion       = 3;  // Vulkan 1.3 (required for extendedDynamicState path)

    nri::QueueFamilyVKDesc queueFamilyVKDesc = {};
    queueFamilyVKDesc.queueNum    = 1;
    queueFamilyVKDesc.queueType   = nri::QueueType::GRAPHICS;
    queueFamilyVKDesc.familyIndex = queueFamilyIndex;

    deviceCreationVKDesc.queueFamilies = &queueFamilyVKDesc;
    deviceCreationVKDesc.queueFamilyNum = 1;

    nri::Device* nriDevice = nullptr;
    if (nri::nriCreateDeviceFromVKDevice(deviceCreationVKDesc, nriDevice) != nri::Result::SUCCESS) {
        LOG_ERROR("NRD: Failed to create NRI device from Vulkan");
        return;
    }
    mNriDevice = nriDevice;

    if (!mNRD)
        mNRD = new nrd::Integration();

    // NRD instance: REBLUR_DIFFUSE
    mReblurDiffuseId = 1;
    nrd::DenoiserDesc denoiserDesc = {};
    denoiserDesc.identifier = mReblurDiffuseId;
    denoiserDesc.denoiser   = nrd::Denoiser::REBLUR_DIFFUSE;

    nrd::InstanceCreationDesc instanceCreationDesc = {};
    instanceCreationDesc.denoisers     = &denoiserDesc;
    instanceCreationDesc.denoisersNum  = 1;

    nrd::IntegrationCreationDesc integrationCreationDesc = {};
    strncpy(integrationCreationDesc.name, "NRD", sizeof(integrationCreationDesc.name) - 1);
    integrationCreationDesc.resourceWidth  = static_cast<uint16_t>(width);
    integrationCreationDesc.resourceHeight = static_cast<uint16_t>(height);
    integrationCreationDesc.queuedFrameNum = 3;
    integrationCreationDesc.enableWholeLifetimeDescriptorCaching = false;
    integrationCreationDesc.autoWaitForIdle = true;

    nrd::Result result = mNRD->RecreateVK(integrationCreationDesc, instanceCreationDesc, deviceCreationVKDesc);
    if (result != nrd::Result::SUCCESS) {
        LOG_ERROR("NRD: RecreateVK failed");
        nri::nriDestroyDevice(static_cast<nri::Device*>(mNriDevice));
        mNriDevice = nullptr;
        return;
    }

    CreatePrepackResources(width, height);
    CreatePrepackPipeline(shaders);
    CreatePrepackDescriptors();

    // NRD REBLUR internal format is RGBA16F (YCoCg in .xyz). Use same format to avoid packing artifacts.
    mOutput.CreateStorageImage(mAllocator, mDevice, width, height, VK_FORMAT_R16G16B16A16_SFLOAT);

    LOG_INFO("NRDDenoiser initialized: {}x{}", width, height);
}

void NRDDenoiser::Shutdown(VkDevice device, VmaAllocator allocator) {
    if (mNRD) {
        mNRD->Destroy();
        delete mNRD;
        mNRD = nullptr;
    }
    if (mNriDevice) {
        nri::nriDestroyDevice(static_cast<nri::Device*>(mNriDevice));
        mNriDevice = nullptr;
    }

    mPrepackRadianceHitDist.Destroy(allocator, device);
    mPrepackNormalRoughness.Destroy(allocator, device);
    mPrepackViewZ.Destroy(allocator, device);
    mPrepackMotion.Destroy(allocator, device);
    mOutput.Destroy(allocator, device);

    if (mPrepackPipeline)   { vkDestroyPipeline(device, mPrepackPipeline, nullptr);   mPrepackPipeline   = VK_NULL_HANDLE; }
    if (mPrepackPipeLayout) { vkDestroyPipelineLayout(device, mPrepackPipeLayout, nullptr); mPrepackPipeLayout = VK_NULL_HANDLE; }
    if (mPrepackDescLayout) { vkDestroyDescriptorSetLayout(device, mPrepackDescLayout, nullptr); mPrepackDescLayout = VK_NULL_HANDLE; }
    if (mPrepackDescPool)   { vkDestroyDescriptorPool(device, mPrepackDescPool, nullptr);   mPrepackDescPool   = VK_NULL_HANDLE; }
    mPrepackDescSet = VK_NULL_HANDLE;
}

void NRDDenoiser::Resize(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h) {
    if (w == mWidth && h == mHeight) return;
    if (!mNriDevice || !mShaders) return;

    Shutdown(device, allocator);
    mWidth = w;
    mHeight = h;
    mHistoryValid = false;

    // Re-initialize with new dimensions
    Initialize(mInstance, device, allocator, *mShaders, mPhysicalDevice, mQueue, mQueueFamilyIndex, w, h);
}

void NRDDenoiser::CreatePrepackResources(uint32_t w, uint32_t h) {
    mPrepackRadianceHitDist.CreateStorageImage(mAllocator, mDevice, w, h, VK_FORMAT_R16G16B16A16_SFLOAT);
    mPrepackNormalRoughness.CreateStorageImage(mAllocator, mDevice, w, h, VK_FORMAT_R8G8B8A8_UNORM);
    mPrepackViewZ.CreateStorageImage(mAllocator, mDevice, w, h, VK_FORMAT_R16_SFLOAT);
    mPrepackMotion.CreateStorageImage(mAllocator, mDevice, w, h, VK_FORMAT_R16G16_SFLOAT);
}

void NRDDenoiser::CreatePrepackPipeline(ShaderManager& shaders) {
    VkDescriptorSetLayoutBinding bindings[9] = {};
    for (int i = 0; i < 9; i++) {
        bindings[i] = {uint32_t(i), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }

    VkDescriptorBindingFlags bindFlags[9] = {};
    for (int i = 0; i < 9; i++)
        bindFlags[i] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    flagsCI.bindingCount  = 9;
    flagsCI.pBindingFlags = bindFlags;

    VkDescriptorSetLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutCI.pNext        = &flagsCI;
    layoutCI.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutCI.bindingCount = 9;
    layoutCI.pBindings    = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(mDevice, &layoutCI, nullptr, &mPrepackDescLayout));

    struct PrepackPC { glm::vec3 hitDistParams; float _pad; };
    VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PrepackPC)};
    VkPipelineLayoutCreateInfo plCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount         = 1;
    plCI.pSetLayouts            = &mPrepackDescLayout;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges    = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(mDevice, &plCI, nullptr, &mPrepackPipeLayout));

    VkShaderModule mod = shaders.GetOrLoad("shaders/nrd_prepack.comp.spv");
    VkComputePipelineCreateInfo pipeCI{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeCI.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeCI.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeCI.stage.module = mod;
    pipeCI.stage.pName  = "main";
    pipeCI.layout       = mPrepackPipeLayout;
    VK_CHECK(vkCreateComputePipelines(mDevice, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &mPrepackPipeline));
}

void NRDDenoiser::CreatePrepackDescriptors() {
    VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 9}};
    VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolCI.maxSets       = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(mDevice, &poolCI, nullptr, &mPrepackDescPool));

    VkDescriptorSetAllocateInfo allocCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocCI.descriptorPool     = mPrepackDescPool;
    allocCI.descriptorSetCount = 1;
    allocCI.pSetLayouts        = &mPrepackDescLayout;
    VK_CHECK(vkAllocateDescriptorSets(mDevice, &allocCI, &mPrepackDescSet));
}

void NRDDenoiser::UpdatePrepackDescriptors(VkImageView colorView, VkImageView normalView,
                                            VkImageView depthView, VkImageView motionView,
                                            VkImageView albedoView) {
    if (!mDescriptorsDirty) return;
    mDescriptorsDirty = false;

    VkDescriptorImageInfo infos[9] = {};
    infos[0].imageView = colorView;
    infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    infos[1].imageView = normalView;
    infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    infos[2].imageView = depthView;
    infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    infos[3].imageView = motionView;
    infos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    infos[4].imageView = mPrepackRadianceHitDist.GetView();
    infos[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    infos[5].imageView = mPrepackNormalRoughness.GetView();
    infos[5].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    infos[6].imageView = mPrepackViewZ.GetView();
    infos[6].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    infos[7].imageView = mPrepackMotion.GetView();
    infos[7].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    infos[8].imageView = albedoView;
    infos[8].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[9] = {};
    for (int i = 0; i < 9; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = mPrepackDescSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo = &infos[i];
    }
    vkUpdateDescriptorSets(mDevice, 9, writes, 0, nullptr);
}

void NRDDenoiser::RunPrepack(VkCommandBuffer cmd) {
    struct PrepackPC {
        glm::vec3 hitDistParams;
        float _pad;
    } pc;
    pc.hitDistParams = glm::vec3(3.0f, 0.1f, 20.0f);  // ReblurHitDistanceParameters default
    pc._pad = 0.0f;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mPrepackPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mPrepackPipeLayout, 0, 1, &mPrepackDescSet, 0, nullptr);
    vkCmdPushConstants(cmd, mPrepackPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (mWidth + 7) / 8, (mHeight + 7) / 8, 1);
}

void NRDDenoiser::SetupNRDCommonSettings(nrd::CommonSettings& common, uint32_t frameIndex,
                                          const glm::mat4& viewMat, const glm::mat4& projMat,
                                          const glm::mat4& viewMatPrev, const glm::mat4& projMatPrev) {
    common.frameIndex = frameIndex;
    common.resourceSize[0] = static_cast<uint16_t>(mWidth);
    common.resourceSize[1] = static_cast<uint16_t>(mHeight);
    common.resourceSizePrev[0] = static_cast<uint16_t>(mWidth);
    common.resourceSizePrev[1] = static_cast<uint16_t>(mHeight);
    common.rectSize[0] = static_cast<uint16_t>(mWidth);
    common.rectSize[1] = static_cast<uint16_t>(mHeight);
    common.rectSizePrev[0] = static_cast<uint16_t>(mWidth);
    common.rectSizePrev[1] = static_cast<uint16_t>(mHeight);
    // NRD sample: use CLEAR_AND_RESTART when history invalid (clears garbage from zoom/camera move)
    common.accumulationMode = mHistoryValid ? nrd::AccumulationMode::CONTINUE : nrd::AccumulationMode::CLEAR_AND_RESTART;
    // Use NRD internal motion from view matrices (zero MV) - more stable for zoom/FOV changes
    // Per-pixel MV can be wrong when projection changes; internal motion handles it correctly
    common.isMotionVectorInWorldSpace = true;
    common.motionVectorScale[0] = 0.0f;
    common.motionVectorScale[1] = 0.0f;
    common.motionVectorScale[2] = 0.0f;
    common.denoisingRange = 500000.0f;
    common.disocclusionThreshold = 0.01f;  // NRD default: reject bad history more aggressively

    // NRD requires these for correct temporal reprojection
    std::memcpy(common.viewToClipMatrix, &projMat[0][0], 16 * sizeof(float));
    std::memcpy(common.viewToClipMatrixPrev, &projMatPrev[0][0], 16 * sizeof(float));
    std::memcpy(common.worldToViewMatrix, &viewMat[0][0], 16 * sizeof(float));
    std::memcpy(common.worldToViewMatrixPrev, &viewMatPrev[0][0], 16 * sizeof(float));
}

void NRDDenoiser::SetupNRDReblurSettings() {
    nrd::ReblurSettings settings = {};
    settings.hitDistanceParameters.A = 3.0f;
    settings.hitDistanceParameters.B = 0.1f;
    settings.hitDistanceParameters.C = 20.0f;
    settings.maxAccumulatedFrameNum = 30;
    settings.diffusePrepassBlurRadius = 0.0f;
    settings.minBlurRadius = 1.0f;
    settings.maxBlurRadius = 22.0f;  // Slightly reduced from 30 to limit dark halos at sharp albedo edges (checkerboard)
    // Use NRD default convergence (s=1, b=0.2) - our s=2 caused insufficient denoising
    settings.convergenceSettings.s = 1.0f;
    settings.convergenceSettings.b = 0.2f;
    mNRD->SetDenoiserSettings(mReblurDiffuseId, &settings);
}

void NRDDenoiser::Denoise(VkCommandBuffer cmd,
                          VkImageView noisyColorView,
                          VkImageView normalView,
                          VkImageView depthView,
                          VkImageView motionView,
                          VkImageView albedoView,
                          VkImage /*depthImage*/,
                          VkImage /*normalImage*/,
                          const glm::mat4& /*invViewProj*/,
                          const glm::mat4& viewProj,
                          const glm::mat4& viewMat,
                          const glm::mat4& projMat,
                          const glm::mat4& viewMatPrev,
                          const glm::mat4& projMatPrev,
                          bool cameraMoved) {
    if (!mNriDevice) return;

    if (cameraMoved)
        mHistoryValid = false;

    mDescriptorsDirty = true;
    UpdatePrepackDescriptors(noisyColorView, normalView, depthView, motionView, albedoView);

    // Transition prepack images to GENERAL (from UNDEFINED on first frame, SHADER_READ_ONLY_OPTIMAL after NRD)
    VkImageLayout prepackOldLayout = mFrameIndex == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    auto makePrepackBarrier = [&](VkImage img) {
        TransitionImage(cmd, img,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, prepackOldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_2_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
            prepackOldLayout, VK_IMAGE_LAYOUT_GENERAL);
    };
    makePrepackBarrier(mPrepackRadianceHitDist.GetImage());
    makePrepackBarrier(mPrepackNormalRoughness.GetImage());
    makePrepackBarrier(mPrepackViewZ.GetImage());
    makePrepackBarrier(mPrepackMotion.GetImage());

    // Transition output to GENERAL (UNDEFINED first frame; SHADER_READ_ONLY after composite sampled it)
    VkImageLayout outOldLayout = mFrameIndex == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    TransitionImage(cmd, mOutput.GetImage(),
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, outOldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
        outOldLayout, VK_IMAGE_LAYOUT_GENERAL);

    RunPrepack(cmd);

    // Transition prepack images GENERAL -> SHADER_READ_ONLY_OPTIMAL for NRD
    auto makePrepackToReadBarrier = [&](VkImage img) {
        TransitionImage(cmd, img,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    };
    makePrepackToReadBarrier(mPrepackRadianceHitDist.GetImage());
    makePrepackToReadBarrier(mPrepackNormalRoughness.GetImage());
    makePrepackToReadBarrier(mPrepackViewZ.GetImage());
    makePrepackToReadBarrier(mPrepackMotion.GetImage());

    mNRD->NewFrame();
    nrd::CommonSettings commonSettings = {};
    SetupNRDCommonSettings(commonSettings, mFrameIndex++, viewMat, projMat, viewMatPrev, projMatPrev);
    mNRD->SetCommonSettings(commonSettings);
    SetupNRDReblurSettings();

    nrd::ResourceSnapshot resourceSnapshot = {};
    resourceSnapshot.restoreInitialState = false;  // We handle layout transitions ourselves

    nrd::Resource resInRadiance = {};
    resInRadiance.vk.image  = reinterpret_cast<uint64_t>(mPrepackRadianceHitDist.GetImage());
    resInRadiance.vk.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    resInRadiance.state = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER};
    resourceSnapshot.SetResource(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, resInRadiance);

    nrd::Resource resInNormal = {};
    resInNormal.vk.image  = reinterpret_cast<uint64_t>(mPrepackNormalRoughness.GetImage());
    resInNormal.vk.format = VK_FORMAT_R8G8B8A8_UNORM;
    resInNormal.state = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER};
    resourceSnapshot.SetResource(nrd::ResourceType::IN_NORMAL_ROUGHNESS, resInNormal);

    nrd::Resource resInViewZ = {};
    resInViewZ.vk.image  = reinterpret_cast<uint64_t>(mPrepackViewZ.GetImage());
    resInViewZ.vk.format = VK_FORMAT_R16_SFLOAT;
    resInViewZ.state = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER};
    resourceSnapshot.SetResource(nrd::ResourceType::IN_VIEWZ, resInViewZ);

    nrd::Resource resInMV = {};
    resInMV.vk.image  = reinterpret_cast<uint64_t>(mPrepackMotion.GetImage());
    resInMV.vk.format = VK_FORMAT_R16G16_SFLOAT;
    resInMV.state = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER};
    resourceSnapshot.SetResource(nrd::ResourceType::IN_MV, resInMV);

    nrd::Resource resOut = {};
    resOut.vk.image  = reinterpret_cast<uint64_t>(mOutput.GetImage());
    resOut.vk.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    resOut.state = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::GENERAL, nri::StageBits::COMPUTE_SHADER};
    resourceSnapshot.SetResource(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, resOut);

    nri::CommandBufferVKDesc cmdDesc = {};
    cmdDesc.vkCommandBuffer = cmd;

    const nrd::Identifier denoisers[] = {mReblurDiffuseId};
    mNRD->DenoiseVK(denoisers, 1, cmdDesc, resourceSnapshot);

    mHistoryValid = true;
    mViewProjPrev = viewProj;
}
