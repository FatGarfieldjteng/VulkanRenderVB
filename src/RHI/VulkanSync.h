#pragma once

#include <volk.h>
#include <vector>
#include <cstdint>

/// All sync objects are sized to the swapchain image count.
/// Acquire semaphores are indexed by a rolling counter.
/// Render-finished semaphores and fences are indexed by the acquired image index.
class VulkanSync {
public:
    void Initialize(VkDevice device, uint32_t swapchainImageCount);
    void Shutdown(VkDevice device);

    VkSemaphore GetImageAvailableSemaphore(uint32_t index) const { return mImageAvailableSemaphores[index]; }
    VkSemaphore GetRenderFinishedSemaphore(uint32_t index) const { return mRenderFinishedSemaphores[index]; }
    VkFence     GetFence(uint32_t index)                   const { return mFences[index]; }

    uint32_t GetCount() const { return static_cast<uint32_t>(mFences.size()); }

private:
    std::vector<VkSemaphore> mImageAvailableSemaphores;
    std::vector<VkSemaphore> mRenderFinishedSemaphores;
    std::vector<VkFence>     mFences;
};
