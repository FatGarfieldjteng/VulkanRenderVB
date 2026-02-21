#pragma once

#include <volk.h>
#include <vector>
#include <cstdint>

class VulkanCommandBuffer {
public:
    void Initialize(VkDevice device, uint32_t graphicsQueueFamily, uint32_t count);
    void Shutdown(VkDevice device);

    /// Resets the command pool for the given index and begins a new command buffer.
    VkCommandBuffer Begin(VkDevice device, uint32_t index);
    void            End(uint32_t index);

    VkCommandBuffer Get(uint32_t index) const { return mCommandBuffers[index]; }

private:
    std::vector<VkCommandPool>   mCommandPools;
    std::vector<VkCommandBuffer> mCommandBuffers;
};
