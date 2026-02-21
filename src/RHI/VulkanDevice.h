#pragma once

#include <volk.h>
#include <cstdint>

struct QueueFamilyIndices {
    uint32_t graphicsFamily = UINT32_MAX;
    uint32_t presentFamily  = UINT32_MAX;
    uint32_t transferFamily = UINT32_MAX;
    uint32_t computeFamily  = UINT32_MAX;

    bool IsComplete() const {
        return graphicsFamily != UINT32_MAX && presentFamily != UINT32_MAX;
    }
};

class VulkanDevice {
public:
    void Initialize(VkInstance instance, VkSurfaceKHR surface);
    void Shutdown();

    VkDevice         GetHandle()         const { return mDevice; }
    VkPhysicalDevice GetPhysicalDevice() const { return mPhysicalDevice; }

    VkQueue GetGraphicsQueue() const { return mGraphicsQueue; }
    VkQueue GetPresentQueue()  const { return mPresentQueue; }
    VkQueue GetTransferQueue() const { return mTransferQueue; }
    VkQueue GetComputeQueue()  const { return mComputeQueue; }

    const QueueFamilyIndices& GetQueueFamilyIndices() const { return mQueueFamilyIndices; }

    void WaitIdle() const;

private:
    void PickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface);
    void CreateLogicalDevice();
    QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) const;
    bool CheckDeviceExtensionSupport(VkPhysicalDevice device) const;

    VkPhysicalDevice   mPhysicalDevice = VK_NULL_HANDLE;
    VkDevice           mDevice         = VK_NULL_HANDLE;

    VkQueue mGraphicsQueue = VK_NULL_HANDLE;
    VkQueue mPresentQueue  = VK_NULL_HANDLE;
    VkQueue mTransferQueue = VK_NULL_HANDLE;
    VkQueue mComputeQueue  = VK_NULL_HANDLE;

    QueueFamilyIndices mQueueFamilyIndices;
    VkSurfaceKHR       mSurface = VK_NULL_HANDLE;
};
