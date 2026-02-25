#pragma once

#include <volk.h>
#include <vector>
#include <cstdint>

/// All sync objects are sized to framesInFlight.
/// Acquire semaphores, render-finished semaphores, and fences are all indexed by mFrameIndex.
class VulkanSync {
public:
    void Initialize(VkDevice device, uint32_t framesInFlight);
    void Shutdown(VkDevice device);

    VkSemaphore GetImageAvailableSemaphore(uint32_t frameIndex) const { return mImageAvailableSemaphores[frameIndex]; }
    VkSemaphore GetRenderFinishedSemaphore(uint32_t frameIndex) const { return mRenderFinishedSemaphores[frameIndex]; }
    VkFence     GetFence(uint32_t frameIndex)                   const { return mFences[frameIndex]; }

    uint32_t GetFrameCount() const { return static_cast<uint32_t>(mFences.size()); }

private:
    std::vector<VkSemaphore> mImageAvailableSemaphores;
    std::vector<VkSemaphore> mRenderFinishedSemaphores;
    std::vector<VkFence>     mFences;
};
