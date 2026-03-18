#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

class TransferManager;

class VulkanImage {
public:
    VulkanImage() = default;

    /// Create a 2D texture from pixel data, upload via staging, and generate mipmaps.
    void CreateTexture2D(VmaAllocator allocator, VkDevice device,
                         const TransferManager& transfer,
                         uint32_t width, uint32_t height,
                         VkFormat format, const void* pixels);

    /// Create a depth-only image (no upload needed).
    void CreateDepth(VmaAllocator allocator, VkDevice device,
                     uint32_t width, uint32_t height,
                     VkFormat format = VK_FORMAT_D32_SFLOAT);

    /// Create a storage image for compute/RT shader read/write.
    void CreateStorageImage(VmaAllocator allocator, VkDevice device,
                            uint32_t width, uint32_t height,
                            VkFormat format);

    void Destroy(VmaAllocator allocator, VkDevice device);

    VkImage     GetImage()     const { return mImage; }
    VkImageView GetView()      const { return mView; }
    uint32_t    GetMipLevels() const { return mMipLevels; }
    uint32_t    GetWidth()     const { return mWidth; }
    uint32_t    GetHeight()    const { return mHeight; }

    void SetHandles(VkImage image, VkImageView view, VmaAllocation alloc) {
        mImage = image; mView = view; mAllocation = alloc;
    }

private:
    void GenerateMipmaps(const TransferManager& transfer, VkFormat format);

    VkImage       mImage      = VK_NULL_HANDLE;
    VkImageView   mView       = VK_NULL_HANDLE;
    VmaAllocation mAllocation = VK_NULL_HANDLE;
    uint32_t      mWidth      = 0;
    uint32_t      mHeight     = 0;
    uint32_t      mMipLevels  = 1;
};
