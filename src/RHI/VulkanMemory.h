#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

class VulkanMemory {
public:
    void Initialize(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device);
    void Shutdown();

    VmaAllocator GetAllocator() const { return mAllocator; }

    /// Logs VMA allocation statistics (debug builds only).
    void LogStats() const;

private:
    VmaAllocator mAllocator = VK_NULL_HANDLE;
};
