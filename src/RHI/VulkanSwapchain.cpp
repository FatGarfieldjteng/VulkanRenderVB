#include "RHI/VulkanSwapchain.h"
#include "RHI/VulkanDevice.h"
#include "Core/Logger.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>

void VulkanSwapchain::Initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                                 VkSurfaceKHR surface, GLFWwindow* window,
                                 const QueueFamilyIndices& indices) {
    Create(device, physicalDevice, surface, window, indices);
}

void VulkanSwapchain::Shutdown(VkDevice device) {
    DestroyImageViews(device);
    if (mSwapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, mSwapchain, nullptr);
        mSwapchain = VK_NULL_HANDLE;
        LOG_INFO("VkSwapchainKHR destroyed");
    }
}

void VulkanSwapchain::Recreate(VkDevice device, VkPhysicalDevice physicalDevice,
                               VkSurfaceKHR surface, GLFWwindow* window,
                               const QueueFamilyIndices& indices) {
    Shutdown(device);
    Create(device, physicalDevice, surface, window, indices);
}

void VulkanSwapchain::Create(VkDevice device, VkPhysicalDevice physicalDevice,
                             VkSurfaceKHR surface, GLFWwindow* window,
                             const QueueFamilyIndices& indices) {
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());

    uint32_t modeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &modeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(modeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &modeCount, presentModes.data());

    auto surfaceFormat = ChooseSurfaceFormat(formats);
    auto presentMode   = ChoosePresentMode(presentModes);
    auto extent        = ChooseExtent(capabilities, window);

    uint32_t imageCount = DESIRED_IMAGE_COUNT;
    if (imageCount < capabilities.minImageCount) imageCount = capabilities.minImageCount;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface          = surface;
    createInfo.minImageCount    = imageCount;
    createInfo.imageFormat      = surfaceFormat.format;
    createInfo.imageColorSpace  = surfaceFormat.colorSpace;
    createInfo.imageExtent      = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.preTransform     = capabilities.currentTransform;
    createInfo.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode      = presentMode;
    createInfo.clipped          = VK_TRUE;
    createInfo.oldSwapchain     = VK_NULL_HANDLE;

    uint32_t queueFamilyIndices[] = { indices.graphicsFamily, indices.presentFamily };
    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices   = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VK_CHECK(vkCreateSwapchainKHR(device, &createInfo, nullptr, &mSwapchain));

    mImageFormat = surfaceFormat.format;
    mExtent      = extent;

    uint32_t actualCount = 0;
    vkGetSwapchainImagesKHR(device, mSwapchain, &actualCount, nullptr);
    mImages.resize(actualCount);
    vkGetSwapchainImagesKHR(device, mSwapchain, &actualCount, mImages.data());

    mImageViews.resize(actualCount);
    for (uint32_t i = 0; i < actualCount; i++) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image    = mImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format   = mImageFormat;
        viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &mImageViews[i]));
    }

    LOG_INFO("Swapchain created: {}x{}, {} images, format {}", extent.width, extent.height,
             actualCount, static_cast<int>(mImageFormat));
}

void VulkanSwapchain::DestroyImageViews(VkDevice device) {
    for (auto view : mImageViews) {
        vkDestroyImageView(device, view, nullptr);
    }
    mImageViews.clear();
    mImages.clear();
}

VkSurfaceFormatKHR VulkanSwapchain::ChooseSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& formats) const
{
    for (const auto& fmt : formats) {
        if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB &&
            fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return fmt;
        }
    }
    return formats[0];
}

VkPresentModeKHR VulkanSwapchain::ChoosePresentMode(
    const std::vector<VkPresentModeKHR>& modes) const
{
    for (auto mode : modes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) return mode;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanSwapchain::ChooseExtent(
    const VkSurfaceCapabilitiesKHR& caps, GLFWwindow* window) const
{
    if (caps.currentExtent.width != UINT32_MAX) {
        return caps.currentExtent;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    VkExtent2D extent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)
    };
    extent.width  = std::clamp(extent.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
    extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    return extent;
}
