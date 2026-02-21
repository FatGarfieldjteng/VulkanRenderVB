#pragma once

#include "Core/Window.h"
#include "RHI/VulkanInstance.h"
#include "RHI/VulkanDevice.h"
#include "RHI/VulkanSwapchain.h"
#include "RHI/VulkanCommandBuffer.h"
#include "RHI/VulkanSync.h"
#include "RHI/VulkanMemory.h"

class Application {
public:
    void Run();

private:
    void InitWindow();
    void InitVulkan();
    void CreateTriangleResources();
    void CreatePipeline();
    void MainLoop();
    void DrawFrame();
    void RecordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    void RecreateSwapchain();
    void CleanupVulkan();

    VkShaderModule CreateShaderModule(const std::string& filepath) const;

    static constexpr uint32_t WINDOW_WIDTH  = 1280;
    static constexpr uint32_t WINDOW_HEIGHT = 720;

    Window              mWindow;
    VulkanInstance      mVulkanInstance;
    VkSurfaceKHR        mSurface = VK_NULL_HANDLE;
    VulkanDevice        mDevice;
    VulkanMemory        mMemory;
    VulkanSwapchain     mSwapchain;
    VulkanSync          mSync;
    VulkanCommandBuffer mCommandBuffers;

    VkPipelineLayout mPipelineLayout   = VK_NULL_HANDLE;
    VkPipeline       mGraphicsPipeline = VK_NULL_HANDLE;

    VkBuffer       mVertexBuffer          = VK_NULL_HANDLE;
    VmaAllocation  mVertexBufferAllocation = VK_NULL_HANDLE;

    uint32_t mAcquireSemaphoreIndex = 0;
    bool     mFramebufferResized    = false;
};
