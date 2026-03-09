#include "PostProcess/PostProcessStack.h"
#include "PostProcess/AutoExposure.h"
#include "PostProcess/SSAO.h"
#include "PostProcess/Bloom.h"
#include "PostProcess/ToneMapping.h"
#include "PostProcess/ColorGrading.h"
#include "Resource/ShaderManager.h"
#include "RHI/VulkanUtils.h"
#include "Core/Logger.h"

#include <memory>

PostProcessStack::PostProcessStack() = default;
PostProcessStack::~PostProcessStack() = default;

void PostProcessStack::Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders,
                                   VkFormat swapchainFormat, uint32_t width, uint32_t height) {
    mWidth  = width;
    mHeight = height;
    mSwapFormat = swapchainFormat;

    CreateHDRImage(device, allocator, width, height);
    CreateLDRImage(device, allocator, width, height, swapchainFormat);
    CreatePlaceholders(device, allocator);

    mAutoExposure = std::make_unique<AutoExposure>();
    mAutoExposure->Initialize(device, allocator, shaders);

    mSSAO = std::make_unique<SSAO>();
    mSSAO->Initialize(device, allocator, shaders, width, height);

    mBloom = std::make_unique<Bloom>();
    mBloom->Initialize(device, allocator, shaders, width, height);

    mToneMapping = std::make_unique<ToneMapping>();
    mToneMapping->Initialize(device, shaders, swapchainFormat);

    mColorGrading = std::make_unique<ColorGrading>();
    mColorGrading->Initialize(device, allocator, shaders, swapchainFormat);

    LOG_INFO("PostProcessStack initialized ({}x{})", width, height);
}

void PostProcessStack::Shutdown(VkDevice device, VmaAllocator allocator) {
    if (mColorGrading) mColorGrading->Shutdown(device, allocator);
    if (mToneMapping)  mToneMapping->Shutdown(device);
    if (mBloom)        mBloom->Shutdown(device, allocator);
    if (mSSAO)         mSSAO->Shutdown(device, allocator);
    if (mAutoExposure) mAutoExposure->Shutdown(device, allocator);

    mColorGrading.reset();
    mToneMapping.reset();
    mBloom.reset();
    mSSAO.reset();
    mAutoExposure.reset();

    DestroyMSAAImages(device, allocator);
    DestroyPlaceholders(device, allocator);
    DestroyLDRImage(device, allocator);
    DestroyHDRImage(device, allocator);
}

void PostProcessStack::Resize(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height) {
    mWidth  = width;
    mHeight = height;

    DestroyHDRImage(device, allocator);
    CreateHDRImage(device, allocator, width, height);

    DestroyLDRImage(device, allocator);
    CreateLDRImage(device, allocator, width, height, mSwapFormat);

    DestroyMSAAImages(device, allocator);
    CreateMSAAImages(device, allocator);

    if (mSSAO) mSSAO->Resize(device, allocator, width, height);
    if (mBloom) mBloom->Resize(device, allocator, width, height);
}

void PostProcessStack::CreateHDRImage(VkDevice device, VmaAllocator allocator,
                                       uint32_t width, uint32_t height) {
    VkImageCreateInfo imgInfo{};
    imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgInfo.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
    imgInfo.extent        = { width, height, 1 };
    imgInfo.mipLevels     = 1;
    imgInfo.arrayLayers   = 1;
    imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                          | VK_IMAGE_USAGE_STORAGE_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VK_CHECK(vmaCreateImage(allocator, &imgInfo, &allocCI, &mHDRImage, &mHDRAlloc, nullptr));

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = mHDRImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = VK_FORMAT_R16G16B16A16_SFLOAT;
    viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &mHDRView));
}

void PostProcessStack::DestroyHDRImage(VkDevice device, VmaAllocator allocator) {
    if (mHDRView)  { vkDestroyImageView(device, mHDRView, nullptr); mHDRView = VK_NULL_HANDLE; }
    if (mHDRImage) { vmaDestroyImage(allocator, mHDRImage, mHDRAlloc); mHDRImage = VK_NULL_HANDLE; }
}

void PostProcessStack::CreateLDRImage(VkDevice device, VmaAllocator allocator,
                                       uint32_t width, uint32_t height, VkFormat fmt) {
    VkImageCreateInfo imgInfo{};
    imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgInfo.format        = fmt;
    imgInfo.extent        = { width, height, 1 };
    imgInfo.mipLevels     = 1;
    imgInfo.arrayLayers   = 1;
    imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VK_CHECK(vmaCreateImage(allocator, &imgInfo, &allocCI, &mLDRImage, &mLDRAlloc, nullptr));

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = mLDRImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = fmt;
    viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &mLDRView));
}

void PostProcessStack::DestroyLDRImage(VkDevice device, VmaAllocator allocator) {
    if (mLDRView)  { vkDestroyImageView(device, mLDRView, nullptr); mLDRView = VK_NULL_HANDLE; }
    if (mLDRImage) { vmaDestroyImage(allocator, mLDRImage, mLDRAlloc); mLDRImage = VK_NULL_HANDLE; }
}

static void CreatePlaceholder1x1(VkDevice device, VmaAllocator allocator,
                                  VkFormat format, uint8_t r, uint8_t g, uint8_t b, uint8_t a,
                                  VkImage& outImage, VkImageView& outView, VmaAllocation& outAlloc) {
    VkImageCreateInfo imgInfo{};
    imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgInfo.format        = format;
    imgInfo.extent        = { 1, 1, 1 };
    imgInfo.mipLevels     = 1;
    imgInfo.arrayLayers   = 1;
    imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VK_CHECK(vmaCreateImage(allocator, &imgInfo, &allocCI, &outImage, &outAlloc, nullptr));

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = outImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = format;
    viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &outView));
}

void PostProcessStack::CreatePlaceholders(VkDevice device, VmaAllocator allocator) {
    CreatePlaceholder1x1(device, allocator, VK_FORMAT_R8_UNORM, 255, 255, 255, 255,
                         mWhitePlaceholder, mWhitePlaceholderView, mWhitePlaceholderAlloc);
    CreatePlaceholder1x1(device, allocator, VK_FORMAT_R16G16B16A16_SFLOAT, 0, 0, 0, 0,
                         mBlackPlaceholder, mBlackPlaceholderView, mBlackPlaceholderAlloc);
}

void PostProcessStack::DestroyPlaceholders(VkDevice device, VmaAllocator allocator) {
    if (mWhitePlaceholderView) { vkDestroyImageView(device, mWhitePlaceholderView, nullptr); mWhitePlaceholderView = VK_NULL_HANDLE; }
    if (mWhitePlaceholder)     { vmaDestroyImage(allocator, mWhitePlaceholder, mWhitePlaceholderAlloc); mWhitePlaceholder = VK_NULL_HANDLE; }
    if (mBlackPlaceholderView) { vkDestroyImageView(device, mBlackPlaceholderView, nullptr); mBlackPlaceholderView = VK_NULL_HANDLE; }
    if (mBlackPlaceholder)     { vmaDestroyImage(allocator, mBlackPlaceholder, mBlackPlaceholderAlloc); mBlackPlaceholder = VK_NULL_HANDLE; }
}

void PostProcessStack::CreateMSAAImages(VkDevice device, VmaAllocator allocator) {
    if (mMSAASamples == VK_SAMPLE_COUNT_1_BIT || mWidth == 0 || mHeight == 0)
        return;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType     = VK_IMAGE_TYPE_2D;
        imgInfo.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
        imgInfo.extent        = { mWidth, mHeight, 1 };
        imgInfo.mipLevels     = 1;
        imgInfo.arrayLayers   = 1;
        imgInfo.samples       = mMSAASamples;
        imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vmaCreateImage(allocator, &imgInfo, &allocCI, &mMSAAColorImage, &mMSAAColorAlloc, nullptr));

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image    = mMSAAColorImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format   = VK_FORMAT_R16G16B16A16_SFLOAT;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &mMSAAColorView));
    }

    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType     = VK_IMAGE_TYPE_2D;
        imgInfo.format        = VK_FORMAT_D32_SFLOAT;
        imgInfo.extent        = { mWidth, mHeight, 1 };
        imgInfo.mipLevels     = 1;
        imgInfo.arrayLayers   = 1;
        imgInfo.samples       = mMSAASamples;
        imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vmaCreateImage(allocator, &imgInfo, &allocCI, &mMSAADepthImage, &mMSAADepthAlloc, nullptr));

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image    = mMSAADepthImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format   = VK_FORMAT_D32_SFLOAT;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &mMSAADepthView));
    }

    LOG_INFO("MSAA images created ({}x{}, {}x samples)", mWidth, mHeight, static_cast<int>(mMSAASamples));
}

void PostProcessStack::DestroyMSAAImages(VkDevice device, VmaAllocator allocator) {
    if (mMSAAColorView)  { vkDestroyImageView(device, mMSAAColorView, nullptr);  mMSAAColorView = VK_NULL_HANDLE; }
    if (mMSAAColorImage) { vmaDestroyImage(allocator, mMSAAColorImage, mMSAAColorAlloc); mMSAAColorImage = VK_NULL_HANDLE; }
    if (mMSAADepthView)  { vkDestroyImageView(device, mMSAADepthView, nullptr);  mMSAADepthView = VK_NULL_HANDLE; }
    if (mMSAADepthImage) { vmaDestroyImage(allocator, mMSAADepthImage, mMSAADepthAlloc); mMSAADepthImage = VK_NULL_HANDLE; }
}

void PostProcessStack::SetMSAASampleCount(VkDevice device, VmaAllocator allocator, VkSampleCountFlagBits samples) {
    if (samples == mMSAASamples) return;
    DestroyMSAAImages(device, allocator);
    mMSAASamples = samples;
    CreateMSAAImages(device, allocator);
}

void PostProcessStack::TransitionPlaceholders(VkCommandBuffer cmd) {
    if (mPlaceholdersReady) return;

    VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    // White placeholder (AO fallback): must be 1.0 so hdr *= ao is identity
    TransitionImage(cmd, mWhitePlaceholder,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkClearColorValue white{};
    white.float32[0] = 1.0f;
    white.float32[1] = 1.0f;
    white.float32[2] = 1.0f;
    white.float32[3] = 1.0f;
    vkCmdClearColorImage(cmd, mWhitePlaceholder, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &white, 1, &range);

    TransitionImage(cmd, mWhitePlaceholder,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Black placeholder (bloom fallback): must be 0.0 so bloom additive is identity
    TransitionImage(cmd, mBlackPlaceholder,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkClearColorValue black{};
    black.float32[0] = 0.0f;
    black.float32[1] = 0.0f;
    black.float32[2] = 0.0f;
    black.float32[3] = 0.0f;
    vkCmdClearColorImage(cmd, mBlackPlaceholder, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &black, 1, &range);

    TransitionImage(cmd, mBlackPlaceholder,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    mPlaceholdersReady = true;
}
