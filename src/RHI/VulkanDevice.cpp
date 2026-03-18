#include "RHI/VulkanDevice.h"
#include "Core/Logger.h"

#include <vector>
#include <set>
#include <string>

static const std::vector<const char*> kRequiredDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,  // Required by NRI
    VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,         // Required by NRI
};

static const std::vector<const char*> kRayTracingExtensions = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
};

static const std::vector<const char*> kRTPipelineExtensions = {
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
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
    mRayTracingSupported = CheckRayTracingSupport(mPhysicalDevice);
    mRTPipelineSupported = mRayTracingSupported && CheckRTPipelineSupport(mPhysicalDevice);

    if (mRTPipelineSupported) {
        mRTPipelineProps = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &mRTPipelineProps;
        vkGetPhysicalDeviceProperties2(mPhysicalDevice, &props2);
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(mPhysicalDevice, &props);
    LOG_INFO("Selected GPU: {}", props.deviceName);
    LOG_INFO("  Graphics queue family: {}", mQueueFamilyIndices.graphicsFamily);
    LOG_INFO("  Present  queue family: {}", mQueueFamilyIndices.presentFamily);
    LOG_INFO("  Transfer queue family: {}", mQueueFamilyIndices.transferFamily);
    LOG_INFO("  Compute  queue family: {}", mQueueFamilyIndices.computeFamily);
    LOG_INFO("  Ray tracing support:   {}", mRayTracingSupported ? "YES" : "NO");
    LOG_INFO("  RT Pipeline support:   {}", mRTPipelineSupported ? "YES" : "NO");
    if (mRTPipelineSupported) {
        LOG_INFO("    shaderGroupHandleSize:      {}", mRTPipelineProps.shaderGroupHandleSize);
        LOG_INFO("    maxRayRecursionDepth:        {}", mRTPipelineProps.maxRayRecursionDepth);
        LOG_INFO("    shaderGroupBaseAlignment:    {}", mRTPipelineProps.shaderGroupBaseAlignment);
        LOG_INFO("    shaderGroupHandleAlignment:  {}", mRTPipelineProps.shaderGroupHandleAlignment);
    }
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

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{};
    rtPipelineFeatures.sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtPipelineFeatures.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
    rayQueryFeatures.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rayQueryFeatures.rayQuery = VK_TRUE;
    rayQueryFeatures.pNext    = mRTPipelineSupported ? (void*)&rtPipelineFeatures : nullptr;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures{};
    accelFeatures.sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelFeatures.accelerationStructure = VK_TRUE;
    accelFeatures.pNext                 = &rayQueryFeatures;

    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT extendedDynamicStateFeatures{};
    extendedDynamicStateFeatures.sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT;
    extendedDynamicStateFeatures.extendedDynamicState = VK_TRUE;  // Required by NRI
    extendedDynamicStateFeatures.pNext = mRayTracingSupported ? (void*)&accelFeatures : nullptr;

    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.dynamicRendering  = VK_TRUE;
    features13.synchronization2  = VK_TRUE;
    features13.pNext             = &extendedDynamicStateFeatures;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.pNext = &features13;

    VkPhysicalDeviceVulkan11Features features11{};
    features11.sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features11.pNext                = &features12;
    features11.shaderDrawParameters = VK_TRUE;
    features12.descriptorIndexing                            = VK_TRUE;
    features12.descriptorBindingPartiallyBound               = VK_TRUE;
    features12.descriptorBindingSampledImageUpdateAfterBind   = VK_TRUE;
    features12.descriptorBindingStorageBufferUpdateAfterBind  = VK_TRUE;
    features12.descriptorBindingStorageImageUpdateAfterBind   = VK_TRUE;
    features12.descriptorBindingUniformBufferUpdateAfterBind  = VK_TRUE;
    features12.runtimeDescriptorArray                        = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing   = VK_TRUE;
    features12.descriptorBindingVariableDescriptorCount    = VK_TRUE;
    features12.drawIndirectCount                           = VK_TRUE;
    features12.samplerFilterMinmax                         = VK_TRUE;
    features12.bufferDeviceAddress                         = VK_TRUE;
    features12.scalarBlockLayout                           = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features11;
    features2.features.samplerAnisotropy      = VK_TRUE;
    features2.features.fillModeNonSolid       = VK_TRUE;
    features2.features.pipelineStatisticsQuery = VK_TRUE;
    features2.features.wideLines              = VK_TRUE;
    features2.features.sampleRateShading      = VK_TRUE;

    std::vector<const char*> enabledExtensions(kRequiredDeviceExtensions);
    if (mRayTracingSupported) {
        enabledExtensions.insert(enabledExtensions.end(),
                                 kRayTracingExtensions.begin(), kRayTracingExtensions.end());
    }
    if (mRTPipelineSupported) {
        enabledExtensions.insert(enabledExtensions.end(),
                                 kRTPipelineExtensions.begin(), kRTPipelineExtensions.end());
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext                   = &features2;
    createInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos       = queueCreateInfos.data();
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();

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

bool VulkanDevice::CheckRayTracingSupport(VkPhysicalDevice device) const {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, available.data());

    std::set<std::string> required(kRayTracingExtensions.begin(), kRayTracingExtensions.end());
    for (const auto& ext : available)
        required.erase(ext.extensionName);

    if (!required.empty())
        return false;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    VkPhysicalDeviceRayQueryFeaturesKHR rqFeatures{};
    rqFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rqFeatures.pNext = &asFeatures;
    VkPhysicalDeviceFeatures2 feats2{};
    feats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feats2.pNext = &rqFeatures;
    vkGetPhysicalDeviceFeatures2(device, &feats2);

    return asFeatures.accelerationStructure && rqFeatures.rayQuery;
}

bool VulkanDevice::CheckRTPipelineSupport(VkPhysicalDevice device) const {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, available.data());

    std::set<std::string> required(kRTPipelineExtensions.begin(), kRTPipelineExtensions.end());
    for (const auto& ext : available)
        required.erase(ext.extensionName);

    if (!required.empty())
        return false;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtpFeatures{};
    rtpFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    VkPhysicalDeviceFeatures2 feats2{};
    feats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feats2.pNext = &rtpFeatures;
    vkGetPhysicalDeviceFeatures2(device, &feats2);

    return rtpFeatures.rayTracingPipeline == VK_TRUE;
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
