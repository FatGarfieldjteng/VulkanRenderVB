#include "RHI/VulkanDevice.h"
#include "Core/Logger.h"

#include <vector>
#include <set>
#include <string>

static const std::vector<const char*> kRequiredDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

void VulkanDevice::Initialize(VkInstance instance, VkSurfaceKHR surface) {
    mSurface = surface;
    PickPhysicalDevice(instance, surface);
    CreateLogicalDevice();
}

void VulkanDevice::Shutdown() {
    if (mDevice != VK_NULL_HANDLE) {
        vkDestroyDevice(mDevice, nullptr);
        mDevice = VK_NULL_HANDLE;
        LOG_INFO("VkDevice destroyed");
    }
}

void VulkanDevice::WaitIdle() const {
    if (mDevice != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(mDevice);
    }
}

void VulkanDevice::PickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface) {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        LOG_ERROR("No Vulkan-capable GPU found");
        return;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    VkPhysicalDevice bestDevice   = VK_NULL_HANDLE;
    int              bestScore    = -1;

    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);

        if (props.apiVersion < VK_API_VERSION_1_3) {
            LOG_WARN("Skipping {} - does not support Vulkan 1.3", props.deviceName);
            continue;
        }

        auto indices = FindQueueFamilies(device, surface);
        if (!indices.IsComplete()) continue;
        if (!CheckDeviceExtensionSupport(device)) continue;

        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)   score += 1000;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 100;

        if (score > bestScore) {
            bestScore    = score;
            bestDevice   = device;
        }
    }

    if (bestDevice == VK_NULL_HANDLE) {
        LOG_ERROR("No suitable GPU found");
        return;
    }

    mPhysicalDevice    = bestDevice;
    mQueueFamilyIndices = FindQueueFamilies(mPhysicalDevice, surface);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(mPhysicalDevice, &props);
    LOG_INFO("Selected GPU: {}", props.deviceName);
    LOG_INFO("  Graphics queue family: {}", mQueueFamilyIndices.graphicsFamily);
    LOG_INFO("  Present  queue family: {}", mQueueFamilyIndices.presentFamily);
    LOG_INFO("  Transfer queue family: {}", mQueueFamilyIndices.transferFamily);
    LOG_INFO("  Compute  queue family: {}", mQueueFamilyIndices.computeFamily);
}

void VulkanDevice::CreateLogicalDevice() {
    std::set<uint32_t> uniqueFamilies = {
        mQueueFamilyIndices.graphicsFamily,
        mQueueFamilyIndices.presentFamily,
        mQueueFamilyIndices.transferFamily,
        mQueueFamilyIndices.computeFamily,
    };

    float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    for (uint32_t family : uniqueFamilies) {
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount       = 1;
        queueInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueInfo);
    }

    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.pNext = &features13;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features12;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext                   = &features2;
    createInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos       = queueCreateInfos.data();
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(kRequiredDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = kRequiredDeviceExtensions.data();

    VK_CHECK(vkCreateDevice(mPhysicalDevice, &createInfo, nullptr, &mDevice));
    volkLoadDevice(mDevice);

    vkGetDeviceQueue(mDevice, mQueueFamilyIndices.graphicsFamily, 0, &mGraphicsQueue);
    vkGetDeviceQueue(mDevice, mQueueFamilyIndices.presentFamily,  0, &mPresentQueue);
    vkGetDeviceQueue(mDevice, mQueueFamilyIndices.transferFamily, 0, &mTransferQueue);
    vkGetDeviceQueue(mDevice, mQueueFamilyIndices.computeFamily,  0, &mComputeQueue);

    LOG_INFO("VkDevice created");
}

QueueFamilyIndices VulkanDevice::FindQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) const {
    QueueFamilyIndices indices;

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    for (uint32_t i = 0; i < count; i++) {
        const auto& props = families[i];

        if (props.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (presentSupport) {
            indices.presentFamily = i;
        }

        // Prefer dedicated transfer queue (no graphics bit)
        if ((props.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(props.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            indices.transferFamily = i;
        }

        // Prefer dedicated compute queue (no graphics bit)
        if ((props.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(props.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            indices.computeFamily = i;
        }
    }

    if (indices.transferFamily == UINT32_MAX) indices.transferFamily = indices.graphicsFamily;
    if (indices.computeFamily  == UINT32_MAX) indices.computeFamily  = indices.graphicsFamily;

    return indices;
}

bool VulkanDevice::CheckDeviceExtensionSupport(VkPhysicalDevice device) const {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, available.data());

    std::set<std::string> required(kRequiredDeviceExtensions.begin(), kRequiredDeviceExtensions.end());
    for (const auto& ext : available) {
        required.erase(ext.extensionName);
    }
    return required.empty();
}
