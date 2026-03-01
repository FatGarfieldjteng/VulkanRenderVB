#include "Resource/VulkanBuffer.h"
#include "Resource/TransferManager.h"
#include "Core/Logger.h"

#include <cstring>

void VulkanBuffer::CreateDeviceLocal(VmaAllocator allocator,
                                     const TransferManager& transfer,
                                     VkBufferUsageFlags usage,
                                     const void* data, VkDeviceSize size)
{
    mSize = size;

    // --- staging buffer (host-visible) ---
    VkBufferCreateInfo stagingBufInfo{};
    stagingBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingBufInfo.size  = size;
    stagingBufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo{};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                             VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer      stagingBuffer     = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation = VK_NULL_HANDLE;
    VmaAllocationInfo stagingInfo{};
    VK_CHECK(vmaCreateBuffer(allocator, &stagingBufInfo, &stagingAllocInfo,
                             &stagingBuffer, &stagingAllocation, &stagingInfo));

    std::memcpy(stagingInfo.pMappedData, data, static_cast<size_t>(size));

    // --- device-local buffer ---
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size  = size;
    bufInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo deviceAllocInfo{};
    deviceAllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VK_CHECK(vmaCreateBuffer(allocator, &bufInfo, &deviceAllocInfo,
                             &mBuffer, &mAllocation, nullptr));

    // --- copy ---
    transfer.ImmediateSubmit([&](VkCommandBuffer cmd) {
        VkBufferCopy region{};
        region.size = size;
        vkCmdCopyBuffer(cmd, stagingBuffer, mBuffer, 1, &region);
    });

    vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
}

void VulkanBuffer::CreateHostVisible(VmaAllocator allocator,
                                     VkBufferUsageFlags usage,
                                     VkDeviceSize size)
{
    mSize = size;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size  = size;
    bufInfo.usage = usage;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                      VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(allocator, &bufInfo, &allocInfo,
                             &mBuffer, &mAllocation, &info));
    mMappedData = info.pMappedData;
}

void VulkanBuffer::CreateDeviceLocalEmpty(VmaAllocator allocator,
                                           VkBufferUsageFlags usage,
                                           VkDeviceSize size) {
    mSize = size;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size  = size;
    bufInfo.usage = usage;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VK_CHECK(vmaCreateBuffer(allocator, &bufInfo, &allocInfo,
                             &mBuffer, &mAllocation, nullptr));
}

void VulkanBuffer::Destroy(VmaAllocator allocator) {
    if (mBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, mBuffer, mAllocation);
        mBuffer     = VK_NULL_HANDLE;
        mAllocation = VK_NULL_HANDLE;
        mMappedData = nullptr;
        mSize       = 0;
    }
}
