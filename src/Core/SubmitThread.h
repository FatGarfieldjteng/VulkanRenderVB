#pragma once

#include <volk.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdint>

struct FramePacket {
    VkCommandBuffer  cmd             = VK_NULL_HANDLE;
    VkSemaphore      waitSemaphore   = VK_NULL_HANDLE;
    VkSemaphore      signalSemaphore = VK_NULL_HANDLE;
    VkFence          fence           = VK_NULL_HANDLE;
    VkSwapchainKHR   swapchain       = VK_NULL_HANDLE;
    uint32_t         imageIndex      = 0;
};

class SubmitThread {
public:
    void Initialize(VkQueue graphicsQueue, VkQueue presentQueue);
    void Shutdown();

    void Submit(const FramePacket& packet);
    void Drain();

private:
    void WorkerLoop();

    std::thread             mThread;
    std::mutex              mMutex;
    std::condition_variable mCondition;
    std::condition_variable mDrainCondition;

    VkQueue      mGraphicsQueue = VK_NULL_HANDLE;
    VkQueue      mPresentQueue  = VK_NULL_HANDLE;

    FramePacket  mPendingPacket{};
    bool         mHasWork    = false;
    bool         mStopping   = false;
    bool         mIdle       = true;
};
