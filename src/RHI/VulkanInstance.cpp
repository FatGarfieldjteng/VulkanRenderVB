#include "RHI/VulkanInstance.h"
#include "Core/Logger.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <vector>
#include <cstring>

static const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";

#ifdef VRB_DEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL DebugMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT             /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void*                                       /*userData*/)
{
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        LOG_ERROR("[Vulkan] {}", callbackData->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        LOG_WARN("[Vulkan] {}", callbackData->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        LOG_INFO("[Vulkan] {}", callbackData->pMessage);
    }
    return VK_FALSE;
}
#endif

void VulkanInstance::Initialize(const std::string& appName) {
    VK_CHECK(volkInitialize());
    LOG_INFO("Volk initialized");

    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = appName.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName        = "VulkanRenderVB";
    appInfo.engineVersion      = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_3;

    uint32_t     glfwExtCount = 0;
    const char** glfwExts     = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);

    std::vector<const char*> layers;

#ifdef VRB_DEBUG
    if (CheckValidationLayerSupport()) {
        layers.push_back(kValidationLayerName);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        LOG_INFO("Validation layers enabled");
    } else {
        LOG_WARN("Validation layers requested but not available");
    }
#endif

    VkInstanceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo        = &appInfo;
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount       = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames     = layers.data();

#ifdef VRB_DEBUG
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    debugCreateInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugCreateInfo.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugCreateInfo.pfnUserCallback = DebugMessengerCallback;
    createInfo.pNext = &debugCreateInfo;
#endif

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &mInstance));
    volkLoadInstance(mInstance);
    LOG_INFO("VkInstance created (Vulkan 1.3)");

#ifdef VRB_DEBUG
    if (!layers.empty()) {
        VK_CHECK(vkCreateDebugUtilsMessengerEXT(mInstance, &debugCreateInfo, nullptr, &mDebugMessenger));
        LOG_INFO("Debug messenger created");
    }
#endif
}

void VulkanInstance::Shutdown() {
#ifdef VRB_DEBUG
    if (mDebugMessenger != VK_NULL_HANDLE) {
        vkDestroyDebugUtilsMessengerEXT(mInstance, mDebugMessenger, nullptr);
        mDebugMessenger = VK_NULL_HANDLE;
    }
#endif
    if (mInstance != VK_NULL_HANDLE) {
        vkDestroyInstance(mInstance, nullptr);
        mInstance = VK_NULL_HANDLE;
        LOG_INFO("VkInstance destroyed");
    }
}

VkSurfaceKHR VulkanInstance::CreateSurface(GLFWwindow* window) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VK_CHECK(glfwCreateWindowSurface(mInstance, window, nullptr, &surface));
    LOG_INFO("VkSurfaceKHR created");
    return surface;
}

bool VulkanInstance::CheckValidationLayerSupport() const {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> layers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

    for (const auto& layer : layers) {
        if (std::strcmp(layer.layerName, kValidationLayerName) == 0) {
            return true;
        }
    }
    return false;
}
