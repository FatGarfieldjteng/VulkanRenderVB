#include "IBL/IBLProcessor.h"
#include "Resource/TransferManager.h"
#include "RHI/VulkanUtils.h"
#include "Core/Logger.h"

#include <stb_image.h>

#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <filesystem>

static VkShaderModule LoadShaderSPV(VkDevice device, const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open shader: {}", path);
        return VK_NULL_HANDLE;
    }
    size_t sz = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> code(sz);
    file.read(code.data(), static_cast<std::streamsize>(sz));

    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = sz;
    ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule mod = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &mod));
    return mod;
}

static VkImageView CreateArrayView(VkDevice device, VkImage image, VkFormat format,
                                   uint32_t baseMip, uint32_t mipCount, uint32_t layers) {
    VkImageViewCreateInfo vi{};
    vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image    = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vi.format   = format;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, 0, layers };
    VkImageView v = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(device, &vi, nullptr, &v));
    return v;
}

static VkImageView CreateCubeView(VkDevice device, VkImage image, VkFormat format,
                                  uint32_t mipCount) {
    VkImageViewCreateInfo vi{};
    vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image    = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    vi.format   = format;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mipCount, 0, 6 };
    VkImageView v = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(device, &vi, nullptr, &v));
    return v;
}

// =========================================================================
void IBLProcessor::Initialize(VmaAllocator allocator, VkDevice device,
                              const TransferManager& transfer, VkPipelineCache pipelineCache) {
    mAllocator     = allocator;
    mDevice        = device;
    mTransfer      = &transfer;
    mPipelineCache = pipelineCache;
}

void IBLProcessor::Process(const char* hdrPath) {
    CreateCubemapImages();
    CreateSamplers();

    bool loaded = false;
    if (hdrPath && std::filesystem::exists(hdrPath)) {
        int w, h, ch;
        float* data = stbi_loadf(hdrPath, &w, &h, &ch, 4);
        if (data) {
            UploadEquirectangular(data, static_cast<uint32_t>(w), static_cast<uint32_t>(h));
            stbi_image_free(data);
            loaded = true;
            LOG_INFO("Loaded HDR environment: {} ({}x{})", hdrPath, w, h);
        }
    }

    if (!loaded) {
        const char* searchPaths[] = {
            "assets/environment.hdr", "assets/sky.hdr",
            "assets/venice_sunset.hdr", "assets/studio.hdr",
        };
        for (const char* p : searchPaths) {
            if (std::filesystem::exists(p)) {
                int w, h, ch;
                float* data = stbi_loadf(p, &w, &h, &ch, 4);
                if (data) {
                    UploadEquirectangular(data, static_cast<uint32_t>(w), static_cast<uint32_t>(h));
                    stbi_image_free(data);
                    loaded = true;
                    LOG_INFO("Loaded HDR environment: {} ({}x{})", p, w, h);
                    break;
                }
            }
        }
    }

    if (!loaded) {
        LOG_INFO("No HDR file found, generating procedural sky");
        GenerateProceduralSky();
    }

    BakeIBL();

    if (mEquirectView)  { vkDestroyImageView(mDevice, mEquirectView, nullptr); mEquirectView = VK_NULL_HANDLE; }
    if (mEquirectImage) { vmaDestroyImage(mAllocator, mEquirectImage, mEquirectAlloc); mEquirectImage = VK_NULL_HANDLE; }

    mReady = true;
    LOG_INFO("IBL processing complete (env={}x{}, irr={}x{}, prefilter={}x{} x{} mips, brdf={}x{})",
             ENV_SIZE, ENV_SIZE, IRR_SIZE, IRR_SIZE, PREFILTER_SIZE, PREFILTER_SIZE,
             PREFILTER_MIP_LEVELS, BRDF_SIZE, BRDF_SIZE);
}

// =========================================================================
void IBLProcessor::CreateCubemapImages() {
    auto makeCube = [&](uint32_t size, uint32_t mips, VkImageUsageFlags extra,
                        VkImage& img, VmaAllocation& alloc) {
        VkImageCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType     = VK_IMAGE_TYPE_2D;
        ci.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
        ci.extent        = { size, size, 1 };
        ci.mipLevels     = mips;
        ci.arrayLayers   = 6;
        ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ci.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | extra;
        ci.flags         = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VK_CHECK(vmaCreateImage(mAllocator, &ci, &ai, &img, &alloc, nullptr));
    };

    makeCube(ENV_SIZE,      1,                  0, mEnvCubemap,        mEnvCubemapAlloc);
    makeCube(IRR_SIZE,      1,                  0, mIrradianceCubemap, mIrradianceCubemapAlloc);
    makeCube(PREFILTER_SIZE, PREFILTER_MIP_LEVELS, 0, mPrefilterCubemap, mPrefilterCubemapAlloc);

    mEnvCubeView        = CreateCubeView(mDevice, mEnvCubemap,        VK_FORMAT_R16G16B16A16_SFLOAT, 1);
    mIrradianceCubeView = CreateCubeView(mDevice, mIrradianceCubemap, VK_FORMAT_R16G16B16A16_SFLOAT, 1);
    mPrefilterCubeView  = CreateCubeView(mDevice, mPrefilterCubemap,  VK_FORMAT_R16G16B16A16_SFLOAT, PREFILTER_MIP_LEVELS);

    {
        VkImageCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType     = VK_IMAGE_TYPE_2D;
        ci.format        = VK_FORMAT_R16G16_SFLOAT;
        ci.extent        = { BRDF_SIZE, BRDF_SIZE, 1 };
        ci.mipLevels     = 1;
        ci.arrayLayers   = 1;
        ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ci.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VK_CHECK(vmaCreateImage(mAllocator, &ci, &ai, &mBRDFLut, &mBRDFLutAlloc, nullptr));

        VkImageViewCreateInfo vi{};
        vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image    = mBRDFLut;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format   = VK_FORMAT_R16G16_SFLOAT;
        vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(mDevice, &vi, nullptr, &mBRDFLutView));
    }
}

void IBLProcessor::CreateSamplers() {
    VkSamplerCreateInfo si{};
    si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter    = VK_FILTER_LINEAR;
    si.minFilter    = VK_FILTER_LINEAR;
    si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod       = static_cast<float>(PREFILTER_MIP_LEVELS);
    VK_CHECK(vkCreateSampler(mDevice, &si, nullptr, &mCubeSampler));

    si.maxLod = 0.0f;
    VK_CHECK(vkCreateSampler(mDevice, &si, nullptr, &mLutSampler));
}

// =========================================================================
void IBLProcessor::UploadEquirectangular(const float* pixels, uint32_t w, uint32_t h) {
    VkDeviceSize imgSize = static_cast<VkDeviceSize>(w) * h * 4 * sizeof(float);

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = imgSize;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo bai{};
    bai.usage = VMA_MEMORY_USAGE_AUTO;
    bai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    VmaAllocationInfo info{};
    vmaCreateBuffer(mAllocator, &bci, &bai, &staging, &stagingAlloc, &info);
    std::memcpy(info.pMappedData, pixels, static_cast<size_t>(imgSize));

    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
    ici.extent        = { w, h, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo iai{};
    iai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VK_CHECK(vmaCreateImage(mAllocator, &ici, &iai, &mEquirectImage, &mEquirectAlloc, nullptr));

    mTransfer->ImmediateSubmit([&](VkCommandBuffer cmd) {
        TransitionImage(cmd, mEquirectImage,
            VK_PIPELINE_STAGE_2_NONE, 0,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy region{};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent      = { w, h, 1 };
        vkCmdCopyBufferToImage(cmd, staging, mEquirectImage,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        TransitionImage(cmd, mEquirectImage,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    vmaDestroyBuffer(mAllocator, staging, stagingAlloc);

    VkImageViewCreateInfo vi{};
    vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image    = mEquirectImage;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format   = VK_FORMAT_R32G32B32A32_SFLOAT;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(mDevice, &vi, nullptr, &mEquirectView));
}

// =========================================================================
void IBLProcessor::GenerateProceduralSky() {
    constexpr uint32_t W = 512, H = 256;
    std::vector<float> pixels(W * H * 4);
    const float pi = 3.14159265359f;

    for (uint32_t y = 0; y < H; y++) {
        float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(H);
        float elevation = (0.5f - v) * pi;
        float sinEl = std::sin(elevation);

        for (uint32_t x = 0; x < W; x++) {
            uint32_t idx = (y * W + x) * 4;

            if (sinEl > 0.0f) {
                float t = sinEl;
                float horizon = std::max(0.0f, 1.0f - t * 3.0f);
                pixels[idx + 0] = 0.15f + 0.55f * horizon + 0.05f * t;
                pixels[idx + 1] = 0.25f + 0.45f * horizon + 0.10f * t;
                pixels[idx + 2] = 0.50f + 0.30f * horizon + 0.45f * t;
            } else {
                float t = -sinEl;
                pixels[idx + 0] = 0.10f * (1.0f - t) + 0.03f * t;
                pixels[idx + 1] = 0.08f * (1.0f - t) + 0.02f * t;
                pixels[idx + 2] = 0.06f * (1.0f - t) + 0.01f * t;
            }
            pixels[idx + 3] = 1.0f;
        }
    }

    UploadEquirectangular(pixels.data(), W, H);
}

// =========================================================================
void IBLProcessor::BakeIBL() {
    // --- Descriptor set layouts ---
    VkDescriptorSetLayoutBinding inputOutputBindings[2]{};
    inputOutputBindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    inputOutputBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo iolci{};
    iolci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    iolci.bindingCount = 2;
    iolci.pBindings    = inputOutputBindings;
    VkDescriptorSetLayout inputOutputLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(mDevice, &iolci, nullptr, &inputOutputLayout));

    VkDescriptorSetLayoutBinding outputBinding = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    VkDescriptorSetLayoutCreateInfo olci{};
    olci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    olci.bindingCount = 1;
    olci.pBindings    = &outputBinding;
    VkDescriptorSetLayout outputOnlyLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(mDevice, &olci, nullptr, &outputOnlyLayout));

    // --- Descriptor pool ---
    // equirect(IO) + irradiance(IO) + 5 prefilter(IO) + brdf(O) = 8 sets
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 7 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          8 },
    };
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 8;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes    = poolSizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(mDevice, &dpci, nullptr, &pool));

    auto allocSet = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &layout;
        VkDescriptorSet s = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateDescriptorSets(mDevice, &ai, &s));
        return s;
    };

    // --- Create temporary array views for compute write ---
    VkImageView envArrayView = CreateArrayView(mDevice, mEnvCubemap, VK_FORMAT_R16G16B16A16_SFLOAT, 0, 1, 6);
    VkImageView irrArrayView = CreateArrayView(mDevice, mIrradianceCubemap, VK_FORMAT_R16G16B16A16_SFLOAT, 0, 1, 6);
    VkImageView prefMipViews[PREFILTER_MIP_LEVELS];
    for (uint32_t m = 0; m < PREFILTER_MIP_LEVELS; m++)
        prefMipViews[m] = CreateArrayView(mDevice, mPrefilterCubemap, VK_FORMAT_R16G16B16A16_SFLOAT, m, 1, 6);

    // --- Allocate and write descriptor sets ---
    VkDescriptorSet equirectSet = allocSet(inputOutputLayout);
    {
        VkDescriptorImageInfo sampler{ mCubeSampler, mEquirectView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo storage{ VK_NULL_HANDLE, envArrayView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w[2]{};
        w[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, equirectSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sampler, nullptr, nullptr };
        w[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, equirectSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &storage, nullptr, nullptr };
        vkUpdateDescriptorSets(mDevice, 2, w, 0, nullptr);
    }

    VkDescriptorSet irrSet = allocSet(inputOutputLayout);
    {
        VkDescriptorImageInfo sampler{ mCubeSampler, mEnvCubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo storage{ VK_NULL_HANDLE, irrArrayView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w[2]{};
        w[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, irrSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sampler, nullptr, nullptr };
        w[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, irrSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &storage, nullptr, nullptr };
        vkUpdateDescriptorSets(mDevice, 2, w, 0, nullptr);
    }

    VkDescriptorSet prefSets[PREFILTER_MIP_LEVELS];
    for (uint32_t m = 0; m < PREFILTER_MIP_LEVELS; m++) {
        prefSets[m] = allocSet(inputOutputLayout);
        VkDescriptorImageInfo sampler{ mCubeSampler, mEnvCubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo storage{ VK_NULL_HANDLE, prefMipViews[m], VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w[2]{};
        w[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, prefSets[m], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sampler, nullptr, nullptr };
        w[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, prefSets[m], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &storage, nullptr, nullptr };
        vkUpdateDescriptorSets(mDevice, 2, w, 0, nullptr);
    }

    VkDescriptorSet brdfSet = allocSet(outputOnlyLayout);
    {
        VkDescriptorImageInfo storage{ VK_NULL_HANDLE, mBRDFLutView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w{};
        w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, brdfSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &storage, nullptr, nullptr };
        vkUpdateDescriptorSets(mDevice, 1, &w, 0, nullptr);
    }

    // --- Pipeline layouts ---
    VkPipelineLayoutCreateInfo plci{};
    plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts    = &inputOutputLayout;
    VkPipelineLayout ioLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(mDevice, &plci, nullptr, &ioLayout));

    VkPushConstantRange prefPC{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) + sizeof(uint32_t) };
    VkPipelineLayoutCreateInfo prefLci{};
    prefLci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    prefLci.setLayoutCount         = 1;
    prefLci.pSetLayouts            = &inputOutputLayout;
    prefLci.pushConstantRangeCount = 1;
    prefLci.pPushConstantRanges    = &prefPC;
    VkPipelineLayout prefLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(mDevice, &prefLci, nullptr, &prefLayout));

    plci.pSetLayouts = &outputOnlyLayout;
    VkPipelineLayout oLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(mDevice, &plci, nullptr, &oLayout));

    // --- Compute pipelines ---
    auto makeCompute = [&](const char* spvPath, VkPipelineLayout layout) -> VkPipeline {
        VkShaderModule mod = LoadShaderSPV(mDevice, spvPath);
        VkComputePipelineCreateInfo ci{};
        ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage  = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                      VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr };
        ci.layout = layout;
        VkPipeline p = VK_NULL_HANDLE;
        VK_CHECK(vkCreateComputePipelines(mDevice, mPipelineCache, 1, &ci, nullptr, &p));
        vkDestroyShaderModule(mDevice, mod, nullptr);
        return p;
    };

    VkPipeline equirectPipe  = makeCompute("shaders/equirect_to_cube.comp.spv", ioLayout);
    VkPipeline irrPipe       = makeCompute("shaders/irradiance.comp.spv",       ioLayout);
    VkPipeline prefPipe      = makeCompute("shaders/prefilter.comp.spv",        prefLayout);
    VkPipeline brdfPipe      = makeCompute("shaders/brdf_lut.comp.spv",         oLayout);

    // --- Dispatch all compute work ---
    mTransfer->ImmediateSubmit([&](VkCommandBuffer cmd) {
        // 1) Equirect → Cubemap
        TransitionImage(cmd, mEnvCubemap,
            VK_PIPELINE_STAGE_2_NONE, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, equirectPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ioLayout, 0, 1, &equirectSet, 0, nullptr);
        vkCmdDispatch(cmd, ENV_SIZE / 16, ENV_SIZE / 16, 6);

        TransitionImage(cmd, mEnvCubemap,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);

        // 2) Irradiance convolution
        TransitionImage(cmd, mIrradianceCubemap,
            VK_PIPELINE_STAGE_2_NONE, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, irrPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ioLayout, 0, 1, &irrSet, 0, nullptr);
        vkCmdDispatch(cmd, std::max(1u, IRR_SIZE / 16), std::max(1u, IRR_SIZE / 16), 6);

        TransitionImage(cmd, mIrradianceCubemap,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);

        // 3) Prefilter per mip
        TransitionImage(cmd, mPrefilterCubemap,
            VK_PIPELINE_STAGE_2_NONE, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_ASPECT_COLOR_BIT, 0, PREFILTER_MIP_LEVELS, 0, 6);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefPipe);

        for (uint32_t m = 0; m < PREFILTER_MIP_LEVELS; m++) {
            uint32_t mipSize = PREFILTER_SIZE >> m;
            struct { float roughness; uint32_t mipSize; } pc;
            pc.roughness = static_cast<float>(m) / static_cast<float>(PREFILTER_MIP_LEVELS - 1);
            pc.mipSize   = mipSize;

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefLayout, 0, 1, &prefSets[m], 0, nullptr);
            vkCmdPushConstants(cmd, prefLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, std::max(1u, mipSize / 16), std::max(1u, mipSize / 16), 6);

            if (m + 1 < PREFILTER_MIP_LEVELS) {
                TransitionImage(cmd, mPrefilterCubemap,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, 6);
            }
        }

        TransitionImage(cmd, mPrefilterCubemap,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, 0, PREFILTER_MIP_LEVELS, 0, 6);

        // 4) BRDF LUT
        TransitionImage(cmd, mBRDFLut,
            VK_PIPELINE_STAGE_2_NONE, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, brdfPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, oLayout, 0, 1, &brdfSet, 0, nullptr);
        vkCmdDispatch(cmd, BRDF_SIZE / 16, BRDF_SIZE / 16, 1);

        TransitionImage(cmd, mBRDFLut,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    // --- Cleanup baking resources ---
    vkDestroyPipeline(mDevice, equirectPipe, nullptr);
    vkDestroyPipeline(mDevice, irrPipe,      nullptr);
    vkDestroyPipeline(mDevice, prefPipe,     nullptr);
    vkDestroyPipeline(mDevice, brdfPipe,     nullptr);

    vkDestroyPipelineLayout(mDevice, ioLayout,   nullptr);
    vkDestroyPipelineLayout(mDevice, prefLayout, nullptr);
    vkDestroyPipelineLayout(mDevice, oLayout,    nullptr);

    vkDestroyDescriptorPool(mDevice, pool, nullptr);
    vkDestroyDescriptorSetLayout(mDevice, inputOutputLayout, nullptr);
    vkDestroyDescriptorSetLayout(mDevice, outputOnlyLayout, nullptr);

    vkDestroyImageView(mDevice, envArrayView, nullptr);
    vkDestroyImageView(mDevice, irrArrayView, nullptr);
    for (uint32_t m = 0; m < PREFILTER_MIP_LEVELS; m++)
        vkDestroyImageView(mDevice, prefMipViews[m], nullptr);
}

// =========================================================================
void IBLProcessor::Shutdown(VmaAllocator allocator, VkDevice device) {
    auto destroyCube = [&](VkImage& img, VmaAllocation& alloc, VkImageView& view) {
        if (view) { vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
        if (img)  { vmaDestroyImage(allocator, img, alloc); img = VK_NULL_HANDLE; alloc = VK_NULL_HANDLE; }
    };

    destroyCube(mEnvCubemap,        mEnvCubemapAlloc,       mEnvCubeView);
    destroyCube(mIrradianceCubemap, mIrradianceCubemapAlloc, mIrradianceCubeView);
    destroyCube(mPrefilterCubemap,  mPrefilterCubemapAlloc,  mPrefilterCubeView);

    if (mBRDFLutView) { vkDestroyImageView(device, mBRDFLutView, nullptr); mBRDFLutView = VK_NULL_HANDLE; }
    if (mBRDFLut)     { vmaDestroyImage(allocator, mBRDFLut, mBRDFLutAlloc); mBRDFLut = VK_NULL_HANDLE; }

    if (mEquirectView)  { vkDestroyImageView(device, mEquirectView, nullptr); mEquirectView = VK_NULL_HANDLE; }
    if (mEquirectImage) { vmaDestroyImage(allocator, mEquirectImage, mEquirectAlloc); mEquirectImage = VK_NULL_HANDLE; }

    if (mCubeSampler) { vkDestroySampler(device, mCubeSampler, nullptr); mCubeSampler = VK_NULL_HANDLE; }
    if (mLutSampler)  { vkDestroySampler(device, mLutSampler, nullptr);  mLutSampler  = VK_NULL_HANDLE; }

    mReady = false;
    LOG_INFO("IBL resources destroyed");
}
