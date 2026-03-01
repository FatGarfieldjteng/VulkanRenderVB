#include "Resource/VulkanImage.h"
#include "Resource/TransferManager.h"
#include "RHI/VulkanUtils.h"
#include "Core/Logger.h"

#include <cmath>
#include <cstring>
#include <algorithm>

void VulkanImage::CreateTexture2D(VmaAllocator allocator, VkDevice device,
                                  const TransferManager& transfer,
                                  uint32_t width, uint32_t height,
                                  VkFormat format, const void* pixels)
{
    mWidth     = width;
    mHeight    = height;
    mMipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;

    VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4; // RGBA

    // --- staging buffer ---
    VkBufferCreateInfo stagingBufInfo{};
    stagingBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingBufInfo.size  = imageSize;
    stagingBufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo{};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                             VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer      stagingBuffer     = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation = VK_NULL_HANDLE;
    VmaAllocationInfo stagingInfo{};
    vmaCreateBuffer(allocator, &stagingBufInfo, &stagingAllocInfo,
                    &stagingBuffer, &stagingAllocation, &stagingInfo);

    std::memcpy(stagingInfo.pMappedData, pixels, static_cast<size_t>(imageSize));

    // --- create image ---
    VkImageCreateInfo imgInfo{};
    imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgInfo.format        = format;
    imgInfo.extent        = { width, height, 1 };
    imgInfo.mipLevels     = mMipLevels;
    imgInfo.arrayLayers   = 1;
    imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo imgAllocInfo{};
    imgAllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VK_CHECK(vmaCreateImage(allocator, &imgInfo, &imgAllocInfo,
                            &mImage, &mAllocation, nullptr));

    // --- copy staging buffer to mip 0 ---
    transfer.ImmediateSubmit([&](VkCommandBuffer cmd) {
        TransitionImage(cmd, mImage,
                        VK_PIPELINE_STAGE_2_NONE, 0,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_ASPECT_COLOR_BIT, 0, mMipLevels);

        VkBufferImageCopy region{};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent      = { width, height, 1 };
        vkCmdCopyBufferToImage(cmd, stagingBuffer, mImage,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    });

    vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);

    // --- generate mipmaps (transitions to SHADER_READ_ONLY_OPTIMAL) ---
    GenerateMipmaps(transfer, format);

    // --- image view ---
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = mImage;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = format;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = mMipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &mView));

    LOG_INFO("Texture2D created: {}x{}, {} mip levels", width, height, mMipLevels);
}

void VulkanImage::CreateDepth(VmaAllocator allocator, VkDevice device,
                              uint32_t width, uint32_t height, VkFormat format)
{
    mWidth     = width;
    mHeight    = height;
    mMipLevels = 1;

    VkImageCreateInfo imgInfo{};
    imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgInfo.format        = format;
    imgInfo.extent        = { width, height, 1 };
    imgInfo.mipLevels     = 1;
    imgInfo.arrayLayers   = 1;
    imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VK_CHECK(vmaCreateImage(allocator, &imgInfo, &allocInfo, &mImage, &mAllocation, nullptr));

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = mImage;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = format;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &mView));
}

void VulkanImage::GenerateMipmaps(const TransferManager& transfer, VkFormat /*format*/)
{
    if (mMipLevels <= 1) return;

    transfer.ImmediateSubmit([&](VkCommandBuffer cmd) {
        int32_t mipW = static_cast<int32_t>(mWidth);
        int32_t mipH = static_cast<int32_t>(mHeight);

        for (uint32_t i = 1; i < mMipLevels; i++) {
            // Transition level i-1 from TRANSFER_DST to TRANSFER_SRC
            TransitionImage(cmd, mImage,
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1);

            int32_t nextW = mipW > 1 ? mipW / 2 : 1;
            int32_t nextH = mipH > 1 ? mipH / 2 : 1;

            VkImageBlit2 blit{};
            blit.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
            blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1 };
            blit.srcOffsets[0]  = { 0, 0, 0 };
            blit.srcOffsets[1]  = { mipW, mipH, 1 };
            blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 };
            blit.dstOffsets[0]  = { 0, 0, 0 };
            blit.dstOffsets[1]  = { nextW, nextH, 1 };

            VkBlitImageInfo2 blitInfo{};
            blitInfo.sType          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
            blitInfo.srcImage       = mImage;
            blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            blitInfo.dstImage       = mImage;
            blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            blitInfo.regionCount    = 1;
            blitInfo.pRegions       = &blit;
            blitInfo.filter         = VK_FILTER_LINEAR;

            vkCmdBlitImage2(cmd, &blitInfo);

            // Transition level i-1 from TRANSFER_SRC to SHADER_READ_ONLY
            TransitionImage(cmd, mImage,
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1);

            mipW = nextW;
            mipH = nextH;
        }

        // Transition last mip from TRANSFER_DST to SHADER_READ_ONLY
        TransitionImage(cmd, mImage,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_IMAGE_ASPECT_COLOR_BIT, mMipLevels - 1, 1);
    });
}

void VulkanImage::Destroy(VmaAllocator allocator, VkDevice device) {
    if (mView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, mView, nullptr);
        mView = VK_NULL_HANDLE;
    }
    if (mImage != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator, mImage, mAllocation);
        mImage      = VK_NULL_HANDLE;
        mAllocation = VK_NULL_HANDLE;
    }
}
