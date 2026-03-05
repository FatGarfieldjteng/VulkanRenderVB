#include "Core/SubmitThread.h"
#include "Core/Logger.h"
#include "RHI/VulkanUtils.h"

void SubmitThread::Initialize(VkQueue graphicsQueue, VkQueue presentQueue) {
    mGraphicsQueue = graphicsQueue;
    mPresentQueue  = presentQueue;
    mStopping = false;
    mThread = std::thread(&SubmitThread::WorkerLoop, this);
    LOG_INFO("SubmitThread started");
}

void SubmitThread::Shutdown() {
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mStopping = true;
    }
    mCondition.notify_one();
    if (mThread.joinable()) mThread.join();
    LOG_INFO("SubmitThread shut down");
}

void SubmitThread::Submit(const FramePacket& packet) {
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mPendingPacket = packet;
        mHasWork = true;
        mIdle    = false;
    }
    mCondition.notify_one();
}

void SubmitThread::Drain() {
    std::unique_lock<std::mutex> lock(mMutex);
    mDrainCondition.wait(lock, [this] { return mIdle; });
}

void SubmitThread::WorkerLoop() {
    for (;;) {
        FramePacket packet;
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mCondition.wait(lock, [this] { return mHasWork || mStopping; });
            if (mStopping && !mHasWork) return;
            packet = mPendingPacket;
            mHasWork = false;
        }

        if (packet.cmd == VK_NULL_HANDLE) {
            std::lock_guard<std::mutex> lock(mMutex);
            mIdle = true;
            mDrainCondition.notify_one();
            continue;
        }

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{};
        submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount   = 1;
        submitInfo.pWaitSemaphores      = &packet.waitSemaphore;
        submitInfo.pWaitDstStageMask    = &waitStage;
        submitInfo.commandBufferCount   = 1;
        submitInfo.pCommandBuffers      = &packet.cmd;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores    = &packet.signalSemaphore;

        VK_CHECK(vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, packet.fence));

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores    = &packet.signalSemaphore;
        presentInfo.swapchainCount     = 1;
        presentInfo.pSwapchains        = &packet.swapchain;
        presentInfo.pImageIndices      = &packet.imageIndex;
        vkQueuePresentKHR(mPresentQueue, &presentInfo);

        {
            std::lock_guard<std::mutex> lock(mMutex);
            mIdle = true;
        }
        mDrainCondition.notify_one();
    }
}
