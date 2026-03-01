#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

class TransferManager;

class VulkanBuffer {
public:
    VulkanBuffer() = default;

    /// Create a device-local buffer and upload data via a staging buffer.
    void CreateDeviceLocal(VmaAllocator allocator,
                           const TransferManager& transfer,
                           VkBufferUsageFlags usage,
                           const void* data, VkDeviceSize size);

    /// Create a host-visible, persistently mapped buffer.
    void CreateHostVisible(VmaAllocator allocator,
                           VkBufferUsageFlags usage,
                           VkDeviceSize size);

    /// Create a device-local buffer without uploading initial data.
    void CreateDeviceLocalEmpty(VmaAllocator allocator,
                                VkBufferUsageFlags usage,
                                VkDeviceSize size);

    void Destroy(VmaAllocator allocator);

    VkBuffer     GetHandle()     const { return mBuffer; }
    VkDeviceSize GetSize()       const { return mSize; }
    void*        GetMappedData() const { return mMappedData; }

private:
    VkBuffer      mBuffer     = VK_NULL_HANDLE;
    VmaAllocation mAllocation = VK_NULL_HANDLE;
    VkDeviceSize  mSize       = 0;
    void*         mMappedData = nullptr;
};
