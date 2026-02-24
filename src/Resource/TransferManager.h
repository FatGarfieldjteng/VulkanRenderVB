#pragma once

#include <volk.h>
#include <functional>

class TransferManager {
public:
    void Initialize(VkDevice device, uint32_t queueFamily, VkQueue queue);
    void Shutdown();

    /// Records and submits a one-shot command buffer, then waits for completion.
    void ImmediateSubmit(std::function<void(VkCommandBuffer)> fn) const;

    VkQueue GetQueue() const { return mQueue; }

private:
    VkDevice      mDevice      = VK_NULL_HANDLE;
    VkQueue       mQueue       = VK_NULL_HANDLE;
    VkCommandPool mCommandPool = VK_NULL_HANDLE;
    VkFence       mFence       = VK_NULL_HANDLE;
};
