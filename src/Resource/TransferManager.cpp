#include "Resource/TransferManager.h"
#include "Core/Logger.h"

void TransferManager::Initialize(VkDevice device, uint32_t queueFamily, VkQueue queue) {
    mDevice = device;
    mQueue  = queue;

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &mCommandPool));

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &mFence));

    LOG_INFO("TransferManager initialized");
}

void TransferManager::Shutdown() {
    if (mDevice == VK_NULL_HANDLE) return;
    if (mFence)       vkDestroyFence(mDevice, mFence, nullptr);
    if (mCommandPool) vkDestroyCommandPool(mDevice, mCommandPool, nullptr);
    mFence       = VK_NULL_HANDLE;
    mCommandPool = VK_NULL_HANDLE;
    LOG_INFO("TransferManager destroyed");
}

void TransferManager::ImmediateSubmit(std::function<void(VkCommandBuffer)> fn) const {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = mCommandPool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(mDevice, &allocInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    fn(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    VK_CHECK(vkQueueSubmit(mQueue, 1, &submitInfo, mFence));
    VK_CHECK(vkWaitForFences(mDevice, 1, &mFence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(mDevice, 1, &mFence));
    vkResetCommandPool(mDevice, mCommandPool, 0);
}
