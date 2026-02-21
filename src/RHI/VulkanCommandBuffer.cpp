#include "RHI/VulkanCommandBuffer.h"
#include "Core/Logger.h"

void VulkanCommandBuffer::Initialize(VkDevice device, uint32_t graphicsQueueFamily, uint32_t count) {
    mCommandPools.resize(count);
    mCommandBuffers.resize(count);

    for (uint32_t i = 0; i < count; i++) {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = graphicsQueueFamily;
        VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &mCommandPools[i]));

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool        = mCommandPools[i];
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &mCommandBuffers[i]));
    }
    LOG_INFO("Command buffers created ({})", count);
}

void VulkanCommandBuffer::Shutdown(VkDevice device) {
    for (auto pool : mCommandPools) {
        if (pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, pool, nullptr);
        }
    }
    mCommandPools.clear();
    mCommandBuffers.clear();
    LOG_INFO("Command buffers destroyed");
}

VkCommandBuffer VulkanCommandBuffer::Begin(VkDevice device, uint32_t index) {
    VK_CHECK(vkResetCommandPool(device, mCommandPools[index], 0));

    auto cmd = mCommandBuffers[index];
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    return cmd;
}

void VulkanCommandBuffer::End(uint32_t index) {
    VK_CHECK(vkEndCommandBuffer(mCommandBuffers[index]));
}
