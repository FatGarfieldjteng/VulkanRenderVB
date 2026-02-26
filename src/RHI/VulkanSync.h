#pragma once

#include <volk.h>
#include <vector>
#include <cstdint>

/// Acquire semaphores and fences are per-frame-in-flight, indexed by mFrameIndex.
/// Render-finished semaphores are per-swapchain-image, indexed by imageIndex,
/// because the presentation engine holds them until the image is re-acquired.
class VulkanSync {
public:
    void Initialize(VkDevice device, uint32_t framesInFlight, uint32_t swapchainImageCount);
    void Shutdown(VkDevice device);

    VkSemaphore GetImageAvailableSemaphore(uint32_t frameIndex) const { return mImageAvailableSemaphores[frameIndex]; }
    VkSemaphore GetRenderFinishedSemaphore(uint32_t imageIndex) const { return mRenderFinishedSemaphores[imageIndex]; }
    VkFence     GetFence(uint32_t frameIndex)                   const { return mFences[frameIndex]; }

    uint32_t GetFrameCount() const { return static_cast<uint32_t>(mFences.size()); }

private:
    std::vector<VkSemaphore> mImageAvailableSemaphores; // per frame-in-flight
    std::vector<VkSemaphore> mRenderFinishedSemaphores; // per swapchain image
    std::vector<VkFence>     mFences;                   // per frame-in-flight
};
