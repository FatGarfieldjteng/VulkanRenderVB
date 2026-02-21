#include "RHI/VulkanMemory.h"
#include "Core/Logger.h"

void VulkanMemory::Initialize(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device) {
    VmaVulkanFunctions vulkanFunctions{};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorInfo.instance         = instance;
    allocatorInfo.physicalDevice   = physicalDevice;
    allocatorInfo.device           = device;
    allocatorInfo.pVulkanFunctions = &vulkanFunctions;

    VK_CHECK(vmaCreateAllocator(&allocatorInfo, &mAllocator));
    LOG_INFO("VMA allocator created");
}

void VulkanMemory::Shutdown() {
    if (mAllocator != VK_NULL_HANDLE) {
        LogStats();
        vmaDestroyAllocator(mAllocator);
        mAllocator = VK_NULL_HANDLE;
        LOG_INFO("VMA allocator destroyed");
    }
}

void VulkanMemory::LogStats() const {
#ifdef VRB_DEBUG
    if (mAllocator == VK_NULL_HANDLE) return;

    VmaTotalStatistics stats{};
    vmaCalculateStatistics(mAllocator, &stats);

    LOG_INFO("VMA stats: {} allocations, {} bytes used, {} bytes total",
             stats.total.statistics.allocationCount,
             stats.total.statistics.allocationBytes,
             stats.total.statistics.blockBytes);
#endif
}
