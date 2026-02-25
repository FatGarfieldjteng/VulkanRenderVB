#include "RHI/VulkanSync.h"
#include "Core/Logger.h"

void VulkanSync::Initialize(VkDevice device, uint32_t framesInFlight) {
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    mImageAvailableSemaphores.resize(framesInFlight);
    mRenderFinishedSemaphores.resize(framesInFlight);
    mFences.resize(framesInFlight);

    for (uint32_t i = 0; i < framesInFlight; i++) {
        VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &mImageAvailableSemaphores[i]));
        VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &mRenderFinishedSemaphores[i]));
        VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &mFences[i]));
    }

    LOG_INFO("Sync objects created ({} frames-in-flight)", framesInFlight);
}

void VulkanSync::Shutdown(VkDevice device) {
    for (auto f : mFences)
        if (f) vkDestroyFence(device, f, nullptr);
    for (auto s : mImageAvailableSemaphores)
        if (s) vkDestroySemaphore(device, s, nullptr);
    for (auto s : mRenderFinishedSemaphores)
        if (s) vkDestroySemaphore(device, s, nullptr);

    mFences.clear();
    mImageAvailableSemaphores.clear();
    mRenderFinishedSemaphores.clear();
    LOG_INFO("Sync objects destroyed");
}
