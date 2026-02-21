#include "RHI/VulkanSync.h"
#include "Core/Logger.h"

void VulkanSync::Initialize(VkDevice device, uint32_t swapchainImageCount) {
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    mImageAvailableSemaphores.resize(swapchainImageCount);
    mRenderFinishedSemaphores.resize(swapchainImageCount);
    mFences.resize(swapchainImageCount);

    for (uint32_t i = 0; i < swapchainImageCount; i++) {
        VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &mImageAvailableSemaphores[i]));
        VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &mRenderFinishedSemaphores[i]));
        VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &mFences[i]));
    }

    LOG_INFO("Sync objects created ({} per swapchain image)", swapchainImageCount);
}

void VulkanSync::Shutdown(VkDevice device) {
    for (size_t i = 0; i < mFences.size(); i++) {
        if (mFences[i])                   vkDestroyFence(device, mFences[i], nullptr);
        if (mRenderFinishedSemaphores[i]) vkDestroySemaphore(device, mRenderFinishedSemaphores[i], nullptr);
        if (mImageAvailableSemaphores[i]) vkDestroySemaphore(device, mImageAvailableSemaphores[i], nullptr);
    }
    mFences.clear();
    mRenderFinishedSemaphores.clear();
    mImageAvailableSemaphores.clear();
    LOG_INFO("Sync objects destroyed");
}
