#pragma once

#include <volk.h>
#include <string>

struct GLFWwindow;

class VulkanInstance {
public:
    void Initialize(const std::string& appName);
    void Shutdown();

    VkInstance   GetHandle() const { return mInstance; }
    VkSurfaceKHR CreateSurface(GLFWwindow* window) const;

private:
    bool CheckValidationLayerSupport() const;

    VkInstance                 mInstance       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT   mDebugMessenger = VK_NULL_HANDLE;
};
